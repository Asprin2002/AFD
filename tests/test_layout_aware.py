from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
import json
from math import exp
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from elasticafd.config import (PolicyConfig, ProjectConfig, SimulationConfig,
                               WorkloadConfig, project_config_from_dict)
from elasticafd.layout_aware import (
    GPU, PIM, LayoutAction, LayoutWorld, PimInterval, StageCost, LinkEvent,
    PreparationJob, _schedule_preparation,
    PrefixCatalog, _attention_us, _linear_us, _linear_shard_us, apply_action, choose_action, evaluate_step,
    finalize_preparations, generate_candidates, _stage_cost, _conditional_survival,
    _expected_active_tokens, _forecast_scenarios, _forecast_total, merge_attention_partials,
)
from elasticafd.partial_softmax import partial_softmax, normalize_partial_softmax, stable_attention
from elasticafd.simulator import run_simulation
from elasticafd.specs import GiB, ModelSpec, SystemSpec
from elasticafd.workload import WorkloadRequest, build_workload, load_prefix_blocks


def tiny_config(*, shared: bool = False, sparse: bool = False,
                policy: str = "layout_fixed") -> ProjectConfig:
    model = ModelSpec("tiny", 0.001, 2, 32, 4, 4, 64, tensor_parallel=8)
    rows = (
        {"request_id": "a", "arrival_us": 0, "input_tokens": 8,
         "output_tokens": 2, **({"prefix_leaf_id": "common"} if shared else {})},
        {"request_id": "b", "arrival_us": 0, "input_tokens": 8,
         "output_tokens": 2, **({"prefix_leaf_id": "common"} if shared else {})},
    )
    workload = WorkloadConfig(
        "closed_batch", requests=rows,
        prefix_blocks=({"id": "common", "parent_id": None, "tokens": 4},)
        if shared else (),
    )
    sim = SimulationConfig(
        cost_backend="layout_aware", max_batch_size=2, record_every=1,
        attention_selector="recent_blocks" if sparse else "dense",
        recent_blocks=1, policy=PolicyConfig(name=policy, decision_interval=1,
                                            planning_horizon_tokens=2),
    )
    return ProjectConfig(model, SystemSpec(), workload, sim)


class LayoutStateTests(unittest.TestCase):
    def test_json_trace_declares_prefixes_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps({
                "prefix_blocks": [{"id": "root", "parent_id": None, "tokens": 3}],
                "requests": [{"request_id": "x", "arrival_us": 0,
                              "input_tokens": 5, "output_tokens": 1,
                              "prefix_leaf_id": "root"}],
            }), encoding="utf-8")
            workload = WorkloadConfig("trace_serving", trace_file=str(path))
            self.assertEqual(build_workload(workload)[0].prefix_leaf_id, "root")
            self.assertEqual(load_prefix_blocks(workload)[0]["id"], "root")

    def test_prefix_identity_and_reference_lifetime(self) -> None:
        cfg = tiny_config(shared=True)
        world = LayoutWorld(cfg, PrefixCatalog(cfg.workload.prefix_blocks))
        for name in ("a", "b"):
            world.admit(WorkloadRequest(name, 0, 8, 2, "common"), 0)
        block = world.blocks["prefix:0:0:common"]
        self.assertEqual(block.refs, {"a", "b"})
        physical = len(world.blocks)
        world.retire("a")
        self.assertEqual(len(world.blocks), physical - cfg.model.layers)
        self.assertEqual(block.refs, {"b"})
        world.retire("b")
        self.assertEqual(world.blocks, {})

    def test_cached_prefix_uses_capacity_without_decode_work(self) -> None:
        cfg = tiny_config(shared=True)
        cfg = replace(cfg, simulation=replace(
            cfg.simulation, prefix_cache_budget_gib=1.0))
        world = LayoutWorld(cfg, PrefixCatalog(cfg.workload.prefix_blocks))
        world.admit(WorkloadRequest("a", 0, 8, 2, "common"), 0)
        world.retire("a")
        self.assertGreater(sum(world.usage()[1]), 0)
        self.assertEqual(evaluate_step(world).total_us, 0)

    def test_retired_prefix_does_not_keep_an_inflight_unused_copy(self) -> None:
        cfg = tiny_config(shared=True)
        cfg = replace(cfg, simulation=replace(
            cfg.simulation, prefix_cache_budget_gib=1.0))
        world = LayoutWorld(cfg, PrefixCatalog(cfg.workload.prefix_blocks))
        world.admit(WorkloadRequest("a", 0, 8, 2, "common"), 0)
        keys = tuple(f"prefix:0:{layer}:common" for layer in range(cfg.model.layers))
        apply_action(world, LayoutAction("copy", ("a",), keys, GPU),
                     baseline_us=1.0)
        world.retire("a")
        self.assertFalse(any(block.shared for block in world.blocks.values()))

    def test_equal_lengths_do_not_share_without_identity(self) -> None:
        cfg = tiny_config()
        world = LayoutWorld(cfg, PrefixCatalog(()))
        for name in ("a", "b"):
            world.admit(WorkloadRequest(name, 0, 8, 2), 0)
        self.assertEqual(len(world.blocks), cfg.model.layers * 2)
        self.assertFalse(any(block.shared for block in world.blocks.values()))

    def test_contiguous_interval_spans_ffn_to_next_attention(self) -> None:
        interval = PimInterval(-1, 1)
        self.assertTrue(interval.contains(0, "ffn"))
        self.assertTrue(interval.contains(1, "qkv"))
        self.assertTrue(interval.contains(1, "attention"))
        self.assertFalse(interval.contains(0, "output_projection"))
        with self.assertRaises(ValueError):
            PimInterval(-1, 4)

    def test_shared_copy_is_one_physical_preparation(self) -> None:
        cfg = tiny_config(shared=True)
        world = LayoutWorld(cfg, PrefixCatalog(cfg.workload.prefix_blocks))
        for name in ("a", "b"):
            world.admit(WorkloadRequest(name, 0, 8, 2, "common"), 0)
        keys = tuple(f"prefix:0:{layer}:common" for layer in range(cfg.model.layers))
        action = LayoutAction("copy", ("a", "b"), keys, GPU)
        first = apply_action(world, action, baseline_us=1.0)
        second = apply_action(world, action, baseline_us=1.0)
        self.assertGreater(first, 0)
        self.assertEqual(second, 0)
        self.assertEqual(world.blocks[keys[0]].refs, {"a", "b"})
        self.assertEqual(world.blocks[keys[0]].owner_at(0), PIM)
        world.clock_us = world.blocks[keys[0]].owner_ready_us
        finalize_preparations(world)
        self.assertEqual(world.blocks[keys[0]].owner, GPU)

    def test_background_copy_waits_across_old_layout_rounds(self) -> None:
        cfg = tiny_config()
        world = LayoutWorld(cfg, PrefixCatalog(()))
        world.admit(WorkloadRequest("a", 0, 8, 10), 0)
        ids = tuple(world.requests["a"].block_ids[layer][0]
                    for layer in range(cfg.model.layers))
        apply_action(world, LayoutAction("copy", ("a",), ids, GPU))
        ready = world.blocks[ids[0]].owner_ready_us
        self.assertGreater(ready, 0)
        if evaluate_step(world).total_us < ready:
            world.clock_us += evaluate_step(world).total_us
            finalize_preparations(world)
            self.assertEqual(world.blocks[ids[0]].owner, PIM)
        world.clock_us = ready
        finalize_preparations(world)
        self.assertEqual(world.blocks[ids[0]].owner, GPU)

    def test_separate_preparations_share_half_duplex_link(self) -> None:
        cfg = tiny_config()
        world = LayoutWorld(cfg, PrefixCatalog(()))
        world.admit(WorkloadRequest("a", 0, 8, 10), 0)
        first, second = world.requests["a"].block_ids[0][0], world.requests["a"].block_ids[1][0]
        duration1 = apply_action(world, LayoutAction("copy", ("a",), (first,), GPU),
                                 baseline_us=1.0)
        duration2 = apply_action(world, LayoutAction("copy", ("a",), (second,), GPU),
                                 baseline_us=1.0)
        self.assertAlmostEqual(world.blocks[first].owner_ready_us, duration1)
        self.assertAlmostEqual(world.blocks[second].owner_ready_us,
                               duration1 + duration2)

    def test_weight_preparation_is_deduplicated_and_waits_for_ready(self) -> None:
        cfg = tiny_config()
        world = LayoutWorld(cfg, PrefixCatalog(()))
        for name in ("a", "b"):
            world.admit(WorkloadRequest(name, 0, 8, 2), 0)
        interval = PimInterval(1, 3)
        prepare = apply_action(world, LayoutAction(
            "interval", ("a", "b"), interval=interval), baseline_us=1.0)
        self.assertGreater(prepare, 0)
        self.assertEqual(len(world.weights), cfg.model.layers)
        self.assertEqual(world.requests["a"].interval_at(0), PimInterval())
        world.clock_us = prepare
        finalize_preparations(world)
        self.assertEqual(world.requests["a"].interval, interval)
        self.assertEqual(world.requests["b"].interval, interval)

    def test_capacity_includes_inflight_copy(self) -> None:
        cfg = tiny_config()
        cfg = replace(cfg, system=replace(
            cfg.system, gpu=replace(cfg.system.gpu, kv_capacity_limit_gib=0.0000015)))
        world = LayoutWorld(cfg, PrefixCatalog(()))
        world.admit(WorkloadRequest("a", 0, 8, 2), 0)
        ids = tuple(world.requests["a"].block_ids[layer][0]
                    for layer in range(cfg.model.layers))
        with self.assertRaises(MemoryError):
            apply_action(world, LayoutAction("copy", ("a",), ids, GPU))

    def test_future_writer_reserves_window_growth(self) -> None:
        cfg = tiny_config()
        cfg = replace(cfg, system=replace(
            cfg.system, gpu=replace(cfg.system.gpu,
                                    kv_capacity_limit_gib=0.0000003)))
        world = LayoutWorld(cfg, PrefixCatalog(()))
        world.admit(WorkloadRequest("a", 0, 8, 2), 0)
        with self.assertRaises(MemoryError):
            apply_action(world, LayoutAction("future", ("a",), target=GPU))

    def test_capacity_recovery_switches_multiple_writers_together(self) -> None:
        cfg = tiny_config(policy="layout_greedy")
        cfg = replace(cfg, system=replace(
            cfg.system, gpu=replace(cfg.system.gpu,
                                    kv_capacity_limit_gib=256 / (1024 ** 3))))
        world = LayoutWorld(cfg, PrefixCatalog(()))
        for name in ("a", "b"):
            world.admit(WorkloadRequest(name, 0, 8, 2), 0)
        for name in ("a", "b"):
            world.requests[name].future_writer = GPU
        with self.assertRaises(MemoryError):
            world.check_capacity(world.planning_reserve_tokens())
        for name in ("a", "b"):
            candidate = deepcopy(world)
            with self.assertRaises(MemoryError):
                apply_action(candidate, LayoutAction("future", (name,), target=PIM))
        action, _, _ = choose_action(world)
        self.assertEqual(action, LayoutAction("future", ("a", "b"), target=PIM))
        apply_action(world, action)
        world.check_capacity(world.planning_reserve_tokens())


class LayoutCostTests(unittest.TestCase):
    def test_candidate_screen_reaches_all_private_requests_across_replans(self) -> None:
        cfg = tiny_config(policy="layout_greedy")
        world = LayoutWorld(cfg, PrefixCatalog(()))
        for index in range(16):
            world.admit(WorkloadRequest(f"r{index}", 0, 8, 40), 0)
        reached: set[str] = set()
        for epoch in range(8):
            world.replan_index = epoch
            screen: dict = {}
            actions = generate_candidates(world, screening=screen)
            self.assertEqual(screen["eligible_groups"], 16)
            self.assertLessEqual(len(actions), cfg.simulation.layout_max_candidates)
            self.assertEqual(screen["truncated_groups"],
                             screen["eligible_groups"] - screen["evaluated_groups"])
            reached.update(screen["evaluated_request_ids"])
        self.assertEqual(reached, {f"r{index}" for index in range(16)})

    def test_shared_ancestor_copy_is_one_physical_candidate_for_all_refs(self) -> None:
        cfg = tiny_config(policy="layout_greedy")
        catalog = PrefixCatalog((
            {"id": "root", "parent_id": None, "tokens": 4},
            {"id": "left", "parent_id": "root", "tokens": 2},
            {"id": "right", "parent_id": "root", "tokens": 2},
        ))
        world = LayoutWorld(replace(cfg, simulation=replace(
            cfg.simulation, layout_max_candidates=64)), catalog)
        world.admit(WorkloadRequest("a", 0, 8, 40, "left"), 0)
        world.admit(WorkloadRequest("b", 0, 8, 40, "right"), 0)
        actions = generate_candidates(world)
        root = [action for action in actions if action.kind == "copy"
                and "prefix:0:0:root" in action.block_ids]
        self.assertEqual(len(root), 1)
        self.assertEqual(root[0].request_ids, ("a", "b"))

    def test_interval_candidates_can_cross_merge_to_reach_projection(self) -> None:
        cfg = tiny_config(policy="layout_greedy")
        cfg = replace(cfg, simulation=replace(cfg.simulation,
                                             layout_max_candidates=64))
        world = LayoutWorld(cfg, PrefixCatalog(()))
        world.admit(WorkloadRequest("a", 0, 8, 40), 0)
        intervals = {action.interval for action in generate_candidates(world)
                     if action.kind == "interval"}
        self.assertIn(PimInterval(1, 3), intervals)

    def test_forecast_window_is_independent_of_replan_cadence(self) -> None:
        cfg = tiny_config(policy="layout_greedy")
        baselines = {}
        for horizon, cadence in ((16, 4), (20, 4), (20, 16)):
            policy = replace(cfg.simulation.policy,
                             planning_horizon_tokens=horizon,
                             decision_interval=cadence)
            configured = replace(cfg, simulation=replace(cfg.simulation,
                                                         policy=policy))
            world = LayoutWorld(configured, PrefixCatalog(()))
            world.admit(WorkloadRequest("a", 0, 8, 40), 0)
            _, _, baseline = choose_action(world)
            baselines[horizon, cadence] = baseline
        self.assertGreater(baselines[20, 4], baselines[16, 4])
        self.assertAlmostEqual(baselines[20, 4], baselines[20, 16])

    def test_explicit_merge_reconstructs_unsplit_softmax(self) -> None:
        scores = [1.0, -2.0, 3.0]
        values = [(1.0, 0.0), (0.0, 1.0), (2.0, 3.0)]
        merged = merge_attention_partials((
            partial_softmax(scores[:1], values[:1]),
            partial_softmax(scores[1:], values[1:]),
        ))
        expected = stable_attention(scores, values)
        for actual, reference in zip(normalize_partial_softmax(merged), expected):
            self.assertAlmostEqual(actual, reference)

    def test_shared_prefix_reduces_physical_scan_not_query_flops(self) -> None:
        shared = tiny_config(shared=True)
        private = tiny_config()
        states = []
        for cfg in (shared, private):
            world = LayoutWorld(cfg, PrefixCatalog(cfg.workload.prefix_blocks))
            for name in ("a", "b"):
                world.admit(WorkloadRequest(name, 0, 8, 2,
                                            "common" if cfg is shared else None), 0)
            states.append(_stage_cost(world, list(world.requests.values()), 0))
        self.assertEqual(states[0].attention_flops, states[1].attention_flops)
        self.assertLess(states[0].pim_scan_bytes, states[1].pim_scan_bytes)

    def test_sparse_partial_overlap_reads_union_within_one_subbatch(self) -> None:
        cfg = tiny_config(shared=True, sparse=True)
        cfg = replace(cfg, simulation=replace(cfg.simulation, recent_blocks=64),
                      workload=replace(cfg.workload, prefix_blocks=(
                          {"id": "common", "parent_id": None, "tokens": 100},)))
        world = LayoutWorld(cfg, PrefixCatalog(cfg.workload.prefix_blocks))
        world.admit(WorkloadRequest("a", 0, 120, 2, "common"), 0)
        world.admit(WorkloadRequest("b", 0, 140, 2, "common"), 0)

        # A reads [56, 100) and B reads [76, 100) of the shared prefix.
        # The 24-token intersection is scanned once; their private tails total 60.
        together = _stage_cost(world, list(world.requests.values()), 0, layer=0)
        separately = sum(_stage_cost(world, [request], 0, layer=0).pim_scan_bytes
                         for request in world.requests.values())
        kv_token_bytes = cfg.model.kv_bytes_per_token_per_layer
        self.assertEqual(together.pim_scan_bytes, 104 * kv_token_bytes)
        self.assertEqual(separately, 128 * kv_token_bytes)
        self.assertEqual(together.attention_flops,
                         128 * 4 * cfg.model.num_attention_heads * cfg.model.head_dim)

        survival = _conditional_survival(0, 5, 256, 1)
        forecast = _stage_cost(world, list(world.requests.values()), 0,
                               offset=5, layer=0)
        # The 19-token forecast intersection is read if either request survives.
        self.assertAlmostEqual(forecast.pim_scan_bytes,
                               (128 * survival - 19 * survival ** 2)
                               * kv_token_bytes)

        # A third request places A and C together while B stays in the other
        # subbatch: only the A/C overlap is reused in the actual two-stage step.
        world.admit(WorkloadRequest("c", 0, 130, 2, "common"), 0)
        self.assertEqual(evaluate_step(world).pim_scan_bytes,
                         158 * cfg.model.layers * kv_token_bytes)

    def test_gemm_uses_batch_rows_and_attention_uses_query_rows(self) -> None:
        cfg = tiny_config()
        self.assertGreater(_linear_us(cfg, GPU, "qkv", 2, layer=0),
                           _linear_us(cfg, GPU, "qkv", 1, layer=0))
        world = LayoutWorld(cfg, PrefixCatalog(()))
        _, flops1, _ = _attention_us(world, PIM, 1, 8)
        _, flops2, _ = _attention_us(world, PIM, 2, 8)
        self.assertEqual(flops2, 2 * flops1)

    def test_dense_shard_config_bounds(self) -> None:
        raw = {"model": "gpt-175b", "hardware": "chime_a100",
               "workload": {"type": "static_batch", "batch_size": 1,
                            "input_tokens": 8, "output_tokens": 2},
               "simulation": {"cost_backend": "layout_aware",
                              "policy": {"name": "layout_fixed"}}}
        for count in (0, 5, True):
            invalid = deepcopy(raw)
            invalid["simulation"]["layout_dense_ranksets"] = count
            with self.assertRaises(ValueError):
                project_config_from_dict(invalid)
        valid = deepcopy(raw)
        valid["simulation"]["layout_dense_ranksets"] = 4
        self.assertEqual(project_config_from_dict(valid).simulation.layout_dense_ranksets, 4)

    def test_forecast_scenario_count_bounds(self) -> None:
        raw = {"model": "gpt-175b", "hardware": "chime_a100",
               "workload": {"type": "static_batch", "batch_size": 1,
                            "input_tokens": 8, "output_tokens": 2},
               "simulation": {"cost_backend": "layout_aware",
                              "policy": {"name": "layout_fixed"}}}
        for count in (0, 65, True, 2.5):
            invalid = deepcopy(raw)
            invalid["simulation"]["layout_forecast_scenarios"] = count
            with self.assertRaises(ValueError):
                project_config_from_dict(invalid)
        valid = deepcopy(raw)
        valid["simulation"]["layout_forecast_scenarios"] = 8
        self.assertEqual(project_config_from_dict(valid).simulation.layout_forecast_scenarios, 8)

    def test_removed_exposure_fraction_is_rejected(self) -> None:
        raw = {"model": "gpt-175b", "hardware": "chime_a100",
               "workload": {"type": "static_batch", "batch_size": 1,
                            "input_tokens": 8, "output_tokens": 2},
               "simulation": {"cost_backend": "layout_aware",
                              "policy": {"name": "layout_fixed"},
                              "layout_preparation_exposed_fraction": 0.25}}
        with self.assertRaises(TypeError):
            project_config_from_dict(raw)

    def test_link_calendar_foreground_waits_for_inflight_packet(self) -> None:
        cfg = replace(tiny_config(), model=replace(tiny_config().model, layers=1),
                      simulation=replace(tiny_config().simulation,
                                         layout_boundary_overlap_fraction=0))
        world = LayoutWorld(cfg, PrefixCatalog(()))
        payload = int(4e-6 * cfg.system.link.pim_to_host_bytes_s)
        world.link_calendar.jobs[0] = PreparationJob(0, "ag", payload, -1, 0, 3)
        world.link_calendar.events.append(LinkEvent(
            -1, 3, "ag", "preparation", 0, payload))
        first = StageCost(10, 0, 5, 0, 0, 0, 0, 0, 0)
        second = StageCost(0, 0, 0, 0, 0, 0, 0, 0, 0)
        with patch("elasticafd.layout_aware._stage_cost", side_effect=[first, second]):
            predicted = evaluate_step(world)
        self.assertEqual(predicted.total_us, 18)
        self.assertEqual(predicted.foreground_wait_us, 3)
        self.assertFalse(any(event.kind == "foreground"
                             for event in world.link_calendar.events))
        with patch("elasticafd.layout_aware._stage_cost", side_effect=[first, second]):
            booked = evaluate_step(world, book_link=True)
        self.assertEqual(booked.total_us, predicted.total_us)
        self.assertEqual(world.link_busy_us, 5)
        self.assertEqual(world.foreground_wait_us, 3)

    def test_full_duplex_opposite_direction_does_not_wait(self) -> None:
        cfg = tiny_config()
        cfg = replace(cfg, model=replace(cfg.model, layers=1),
                      system=replace(cfg.system, link=replace(
                          cfg.system.link, full_duplex=True)),
                      simulation=replace(cfg.simulation,
                                         layout_boundary_overlap_fraction=0))
        world = LayoutWorld(cfg, PrefixCatalog(()))
        world.link_calendar.jobs[0] = PreparationJob(0, "ag", 4, -1, 0, 30)
        world.link_calendar.events.append(LinkEvent(-1, 30, "ag", "preparation", 0, 4))
        first = StageCost(10, 0, 5, 0, 0, 0, 0, 0, 0)
        second = StageCost(0, 0, 0, 0, 0, 0, 0, 0, 0)
        with patch("elasticafd.layout_aware._stage_cost", side_effect=[first, second]):
            cost = evaluate_step(world)
        self.assertEqual(cost.foreground_wait_us, 0)
        self.assertEqual(cost.total_us, 15)

    def test_foreground_packet_delays_unsent_preparation_and_copy_ready(self) -> None:
        cfg = tiny_config()
        cfg = replace(cfg, model=replace(cfg.model, layers=1),
                      system=replace(cfg.system, link=replace(
                          cfg.system.link, host_to_pim_gbps=0.001)),
                      simulation=replace(cfg.simulation,
                                         layout_preparation_packet_bytes=32,
                                         layout_boundary_overlap_fraction=0))
        world = LayoutWorld(cfg, PrefixCatalog(()))
        world.admit(WorkloadRequest("a", 0, 8, 2), 0)
        block_id = world.requests["a"].block_ids[0][0]
        apply_action(world, LayoutAction("copy", ("a",), (block_id,), GPU),
                     baseline_us=1)
        # Add a GA preparation to exercise packet preemption on the same link.
        _, ready, _ = _schedule_preparation(world, 128, PIM)
        first = StageCost(10, 0, 40, 0, 0, 0, 0, 0, 0)
        second = StageCost(0, 0, 0, 0, 0, 0, 0, 0, 0)
        with patch("elasticafd.layout_aware._stage_cost", side_effect=[first, second]):
            evaluate_step(world, book_link=True)
        self.assertGreater(world.link_calendar.jobs[1].ready_us, ready)

    def test_preparation_preempts_at_quantum_without_repeating_startup(self) -> None:
        cfg = tiny_config()
        cfg = replace(cfg, model=replace(cfg.model, layers=1),
                      system=replace(cfg.system, link=replace(
                          cfg.system.link, host_to_pim_gbps=0.001,
                          transfer_latency_us=2)),
                      simulation=replace(cfg.simulation,
                                         layout_preparation_packet_bytes=4,
                                         layout_boundary_overlap_fraction=0))
        world = LayoutWorld(cfg, PrefixCatalog(()))
        duration, ready, job_id = _schedule_preparation(world, 8, PIM, 3)
        self.assertEqual((duration, ready), (13, 13))
        world.clock_us = 3
        first = StageCost(10, 0, 2, 0, 0, 0, 0, 0, 0)
        second = StageCost(0, 0, 0, 0, 0, 0, 0, 0, 0)
        with patch("elasticafd.layout_aware._stage_cost", side_effect=[first, second]):
            step = evaluate_step(world, book_link=True)
        self.assertEqual(step.foreground_wait_us, 3)
        self.assertEqual(world.link_calendar.jobs[job_id].ready_us, 15)
        self.assertEqual(sum(event.startup_us for event in world.link_calendar.events
                             if event.kind == "preparation"), 2)

    def test_phase_label_uses_layer_critical_time_not_majority(self) -> None:
        cfg = tiny_config()
        cfg = replace(cfg, model=replace(cfg.model, layers=3))
        world = LayoutWorld(cfg, PrefixCatalog(()))
        rows = []
        for gpu, pim in ((2, 1), (2, 1), (0, 100)):
            rows.extend((StageCost(gpu, 0, 0, 0, 0, 0, 0, 0, 0),
                         StageCost(0, pim, 0, 0, 0, 0, 0, 0, 0)))
        with patch("elasticafd.layout_aware._stage_cost", side_effect=rows):
            cost = evaluate_step(world)
        self.assertEqual(cost.stage0_gpu_layers, 2)
        self.assertEqual(cost.stage0_pim_layers, 1)
        self.assertEqual(cost.stage0_bottleneck, PIM)
        self.assertEqual(cost.gpu_busy_us, 4)
        self.assertEqual(cost.pim_busy_us, 102)

    def test_two_layer_opposite_bottlenecks_are_mixed(self) -> None:
        world = LayoutWorld(tiny_config(), PrefixCatalog(()))
        rows = [StageCost(10, 0, 0, 0, 0, 0, 0, 0, 0),
                StageCost(0, 0, 0, 0, 0, 0, 0, 0, 0),
                StageCost(0, 0, 0, 0, 0, 0, 0, 0, 0),
                StageCost(0, 10, 0, 0, 0, 0, 0, 0, 0)]
        with patch("elasticafd.layout_aware._stage_cost", side_effect=rows):
            cost = evaluate_step(world)
        self.assertEqual(cost.total_us, 20)
        self.assertEqual(cost.stage0_bottleneck, "mixed")
        self.assertEqual((cost.stage0_gpu_layers, cost.stage0_pim_layers), (1, 1))

    def test_dense_shards_keep_full_input_per_group_and_attention_unchanged(self) -> None:
        baseline = tiny_config()
        sharded = replace(baseline, simulation=replace(
            baseline.simulation, layout_dense_ranksets=4))
        model, pim = sharded.model, sharded.system.pim
        n = model.hidden_size + 2 * model.kv_width
        n_shard = (n + 3) // 4
        bandwidth = pim.bandwidth_per_rankset_bytes_s * pim.bank_hbm_efficiency
        throughput = bandwidth * pim.dense_mac_flops_per_byte * pim.dense_compute_efficiency
        m = 2
        expected = max(2 * m * model.hidden_size * n_shard / throughput,
                       (model.hidden_size * n_shard + m * model.hidden_size
                        + m * n_shard) * model.dtype_bytes / bandwidth) * 1e6
        self.assertAlmostEqual(_linear_shard_us(sharded, "qkv", m)[0], expected)
        self.assertLess(_linear_us(sharded, PIM, "qkv", m, layer=0),
                        _linear_us(baseline, PIM, "qkv", m, layer=0))
        worlds = (LayoutWorld(baseline, PrefixCatalog(())),
                  LayoutWorld(sharded, PrefixCatalog(())))
        self.assertEqual(_attention_us(worlds[0], PIM, 2, 128),
                         _attention_us(worlds[1], PIM, 2, 128))

    def test_sharded_weights_conserve_total_bytes_and_preparation_time(self) -> None:
        for interval, operators in ((PimInterval(0, 1), ("qkv",)),
                                    (PimInterval(1, 4), ("output_projection", "ffn"))):
            states = []
            for count in (1, 4):
                cfg = tiny_config()
                cfg = replace(cfg, simulation=replace(cfg.simulation,
                                                      layout_dense_ranksets=count))
                world = LayoutWorld(cfg, PrefixCatalog(()))
                world.admit(WorkloadRequest("a", 0, 8, 2), 0)
                transfer = apply_action(world, LayoutAction(
                    "interval", ("a",), interval=interval), baseline_us=1)
                weights = list(world.weights.values())
                self.assertEqual(len(weights), cfg.model.layers * len(operators))
                self.assertTrue(all(len(weight.ranksets) == count for weight in weights))
                expected = {"qkv": cfg.model.qkv_weight_bytes_per_layer,
                            "output_projection": cfg.model.output_projection_weight_bytes_per_layer,
                            "ffn": cfg.model.ff_weight_bytes_per_layer}
                self.assertTrue(all(weight.bytes == expected[weight.operator]
                                    for weight in weights))
                states.append((world.preparation_bytes, transfer,
                               sum(world.usage()[1]), max(w.ready_us for w in weights)))
            self.assertEqual(states[0], states[1])

    def test_shard_bus_communication_follows_downstream_layout(self) -> None:
        cfg = tiny_config()
        cfg = replace(cfg, simulation=replace(cfg.simulation, layout_dense_ranksets=4))
        world = LayoutWorld(cfg, PrefixCatalog(()))
        world.admit(WorkloadRequest("a", 0, 8, 2), 0)
        request = world.requests["a"]
        h, kv, f, dtype = (cfg.model.hidden_size, cfg.model.kv_width,
                            cfg.model.ffn_hidden_size, cfg.model.dtype_bytes)
        request.interval = PimInterval(0, 1)
        qkv = _stage_cost(world, [request], 0, layer=0)
        local = (h + 2 * kv + 3) // 4
        self.assertEqual(qkv.cross_group_bytes,
                         (3 * h + h + 2 * kv - local) * dtype)
        request.interval = PimInterval(1, 3)
        output_to_gpu = _stage_cost(world, [request], 0, layer=0)
        self.assertEqual(output_to_gpu.cross_group_bytes, 3 * h * dtype)
        request.interval = PimInterval(1, 4)
        output_to_ffn = _stage_cost(world, [request], 0, layer=0)
        self.assertEqual(output_to_ffn.cross_group_bytes,
                         (6 * h + 3 * f) * dtype)
        self.assertGreater(output_to_ffn.cross_group_us, output_to_gpu.cross_group_us)
        self.assertAlmostEqual(output_to_ffn.cross_group_us,
                               output_to_ffn.cross_group_bytes /
                               cfg.system.pim.rank_bus_per_rankset_bytes_s * 1e6)

    def test_sharded_weight_capacity_is_per_rankset(self) -> None:
        cfg = tiny_config()
        cfg = replace(cfg, model=replace(cfg.model, layers=1))
        worlds = []
        for count in (1, 4):
            local = replace(cfg, simulation=replace(cfg.simulation,
                                                    layout_dense_ranksets=count))
            world = LayoutWorld(local, PrefixCatalog(()))
            world.admit(WorkloadRequest("a", 0, 8, 2), 0)
            apply_action(world, LayoutAction("interval", ("a",),
                                             interval=PimInterval(0, 1)),
                         baseline_us=1)
            worlds.append(world)
        single, sharded = (max(world.usage()[1]) for world in worlds)
        self.assertLess(sharded, single)
        rankset_limit = (single + sharded) / 2
        for world in worlds:
            world.config = replace(world.config,
                                   system=replace(world.config.system,
                                                  pim=replace(world.config.system.pim,
                                                              capacity_gib=4 * rankset_limit / GiB)))
        with self.assertRaises(MemoryError):
            worlds[0].check_capacity()
        worlds[1].check_capacity()

    def test_two_shards_deliver_cross_layer_input_to_new_group(self) -> None:
        cfg = tiny_config()
        cfg = replace(cfg, simulation=replace(cfg.simulation, layout_dense_ranksets=2,
                                              layout_rankset_sync_us=1.0))
        world = LayoutWorld(cfg, PrefixCatalog(()))
        world.admit(WorkloadRequest("a", 0, 8, 2), 0)
        request = world.requests["a"]
        request.interval = PimInterval(-1, 1)
        stage = _stage_cost(world, [request], 0, layer=0)
        one_layer = replace(cfg, model=replace(cfg.model, layers=1))
        short_world = LayoutWorld(one_layer, PrefixCatalog(()))
        short_world.admit(WorkloadRequest("a", 0, 8, 2), 0)
        short_request = short_world.requests["a"]
        short_request.interval = PimInterval(-1, 1)
        short_stage = _stage_cost(short_world, [short_request], 0, layer=0)
        hbytes = cfg.model.hidden_size * cfg.model.dtype_bytes
        # One all-gather across current FFN groups and one full copy to the
        # new group used by the next layer's QKV.
        self.assertEqual(stage.cross_group_bytes - short_stage.cross_group_bytes,
                         2 * hbytes)
        self.assertAlmostEqual(stage.cross_group_us - short_stage.cross_group_us,
                               2 * hbytes / cfg.system.pim.rank_bus_per_rankset_bytes_s
                               * 1e6 + 2.0)

    def test_gpu_cost_uses_one_data_parallel_replicas_resources(self) -> None:
        single = tiny_config()
        doubled = replace(
            single,
            model=replace(single.model, data_parallel=2),
            system=replace(single.system, gpu=replace(
                single.system.gpu, count=single.system.gpu.count * 2,
                capacity_gib=single.system.gpu.capacity_gib * 2,
                hbm_bandwidth_tbps=single.system.gpu.hbm_bandwidth_tbps * 2,
                fp16_tflops=single.system.gpu.fp16_tflops * 2)),
        )
        costs = []
        for cfg in (single, doubled):
            world = LayoutWorld(cfg, PrefixCatalog(()))
            world.admit(WorkloadRequest("a", 0, 8, 2), 0)
            costs.append(_stage_cost(world, [world.requests["a"]], 0).gpu_us)
            if cfg is doubled:
                world.admit(WorkloadRequest("b", 0, 8, 2), 1)
                parallel = _stage_cost(world, list(world.requests.values()), 0)
                self.assertAlmostEqual(parallel.gpu_us, costs[-1])
        self.assertAlmostEqual(costs[0], costs[1])

    def test_sparse_selector_changes_only_accessed_blocks(self) -> None:
        dense = tiny_config(shared=True)
        sparse = tiny_config(shared=True, sparse=True)
        values = []
        for cfg in (dense, sparse):
            world = LayoutWorld(cfg, PrefixCatalog(cfg.workload.prefix_blocks))
            for name in ("a", "b"):
                world.admit(WorkloadRequest(name, 0, 8, 2, "common"), 0)
            values.append(evaluate_step(world).pim_scan_bytes)
        self.assertLess(values[1], values[0])

    def test_sparse_window_stays_bounded_inside_growing_physical_segment(self) -> None:
        cfg = tiny_config(sparse=True)
        world = LayoutWorld(cfg, PrefixCatalog(()))
        world.admit(WorkloadRequest("a", 0, 8, 20), 0)
        initial = evaluate_step(world).pim_scan_bytes
        for _ in range(5):
            world.append_token(world.requests["a"])
        self.assertEqual(evaluate_step(world).pim_scan_bytes, initial)
        self.assertAlmostEqual(evaluate_step(world, offset=5).pim_scan_bytes,
                               initial * _conditional_survival(5, 5, 256, 1))

    def test_two_stage_objective_uses_swapped_subbatches(self) -> None:
        cfg = tiny_config()
        world = LayoutWorld(cfg, PrefixCatalog(()))
        for name in ("a", "b"):
            world.admit(WorkloadRequest(name, 0, 8, 2), 0)
        step = evaluate_step(world)
        stages = []
        for layer in range(cfg.model.layers):
            a = _stage_cost(world, [world.requests["a"]], 0, layer=layer)
            b = _stage_cost(world, [world.requests["b"]], 0, layer=layer)
            stages.append(max(a.gpu_us, b.pim_us) + max(b.gpu_us, a.pim_us))
        self.assertAlmostEqual(step.total_us, sum(stages) + step.boundary_us)

    def test_future_writer_moves_only_future_attention_work(self) -> None:
        cfg = tiny_config()
        world = LayoutWorld(cfg, PrefixCatalog(()))
        world.admit(WorkloadRequest("a", 0, 8, 20), 0)
        candidate = deepcopy(world)
        apply_action(candidate, LayoutAction("future", ("a",), target=GPU))
        self.assertEqual(candidate.preparation_bytes, 0)
        self.assertEqual(evaluate_step(world).pim_scan_bytes,
                         evaluate_step(candidate).pim_scan_bytes)
        self.assertLess(_stage_cost(candidate, list(candidate.requests.values()), 0).gpu_to_pim_us,
                        _stage_cost(world, list(world.requests.values()), 0).gpu_to_pim_us)
        self.assertGreater(evaluate_step(candidate, offset=5).gpu_scan_bytes,
                           evaluate_step(world, offset=5).gpu_scan_bytes)

    def test_pim_qkv_returns_q_for_gpu_partial_attention(self) -> None:
        cfg = tiny_config(shared=True)
        world = LayoutWorld(cfg, PrefixCatalog(cfg.workload.prefix_blocks))
        world.admit(WorkloadRequest("a", 0, 8, 2, "common"), 0)
        request = world.requests["a"]
        for layer in range(cfg.model.layers):
            private = world.blocks[request.block_ids[layer][-1]]
            world.add_copy(private, GPU, 0)
            private.owner = GPU
        gpu_qkv = _stage_cost(world, [request], 0).pim_to_gpu_us
        request.interval = PimInterval(0, 1)
        pim_qkv = _stage_cost(world, [request], 0).pim_to_gpu_us
        expected = (cfg.model.layers * cfg.model.q_bytes_per_token_per_layer
                    / cfg.system.link.pim_to_host_bytes_s * 1e6)
        self.assertAlmostEqual(pim_qkv - gpu_qkv, expected)

    def test_dependent_layers_add_even_on_different_pim_groups(self) -> None:
        cfg = tiny_config()
        world = LayoutWorld(cfg, PrefixCatalog(()))
        world.admit(WorkloadRequest("a", 0, 8, 2), 0)
        world.blocks[world.requests["a"].block_ids[0][0]].tokens = 100
        world.blocks[world.requests["a"].block_ids[1][0]].tokens = 1
        first = _attention_us(world, PIM, 1, 100)[0]
        second = _attention_us(world, PIM, 1, 1)[0]
        stage = _stage_cost(world, [world.requests["a"]], 0)
        self.assertAlmostEqual(stage.pim_us,
                               first + second + 2 * cfg.system.pim.request_setup_us)
        world.blocks[world.requests["a"].block_ids[1][0]].copies[PIM].rankset = 0
        crowded = _stage_cost(world, [world.requests["a"]], 0)
        self.assertAlmostEqual(crowded.pim_us, stage.pim_us)

    def test_independent_groups_overlap_only_within_one_layer(self) -> None:
        cfg = tiny_config()
        world = LayoutWorld(cfg, PrefixCatalog(()))
        for name in ("a", "b"):
            world.admit(WorkloadRequest(name, 0, 8, 2), 0)
        world.blocks[world.requests["b"].block_ids[0][0]].copies[PIM].rankset = 1
        parallel = _stage_cost(world, list(world.requests.values()), 0, layer=0)
        world.blocks[world.requests["b"].block_ids[0][0]].copies[PIM].rankset = 0
        serial = _stage_cost(world, list(world.requests.values()), 0, layer=0)
        self.assertGreater(serial.pim_us, parallel.pim_us)

    def test_dependent_pim_operators_do_not_overlap_across_groups(self) -> None:
        cfg = tiny_config()
        world = LayoutWorld(cfg, PrefixCatalog(()))
        world.admit(WorkloadRequest("a", 0, 8, 2), 0)
        request = world.requests["a"]
        request.interval = PimInterval(0, 1)  # QKV must precede Attention.
        block = world.blocks[request.block_ids[0][0]]
        colocated = _stage_cost(world, [request], 0, layer=0).pim_us
        block.copies[PIM].rankset = 1
        separated = _stage_cost(world, [request], 0, layer=0).pim_us
        self.assertAlmostEqual(separated, colocated)

    def test_layer_cycles_sum_two_stage_makespans(self) -> None:
        cfg = tiny_config()
        world = LayoutWorld(cfg, PrefixCatalog(()))
        for name in ("a", "b"):
            world.admit(WorkloadRequest(name, 0, 8, 2), 0)
        world.blocks[world.requests["a"].block_ids[0][0]].tokens = 10000
        step = evaluate_step(world)
        expected = 0.0
        for layer in range(cfg.model.layers):
            a = _stage_cost(world, [world.requests["a"]], 0, layer=layer)
            b = _stage_cost(world, [world.requests["b"]], 0, layer=layer)
            expected += max(a.gpu_us, b.pim_us) + max(b.gpu_us, a.pim_us)
        self.assertAlmostEqual(step.total_us, expected + step.boundary_us)

    def test_survival_is_conditional_on_observed_progress(self) -> None:
        self.assertAlmostEqual(_conditional_survival(0, 64, 256, 1), exp(-64 / 256))
        self.assertAlmostEqual(_conditional_survival(512, 64, 256, 1), exp(-64 / 256))
        self.assertGreater(_expected_active_tokens(512, 256, 256, 1), 100)
        self.assertLess(_conditional_survival(512, 64, 256, 2),
                        _conditional_survival(0, 64, 256, 2))

    def test_forecast_advances_on_candidate_timeline(self) -> None:
        world = LayoutWorld(tiny_config(), PrefixCatalog(()))
        world.admit(WorkloadRequest("a", 0, 8, 1), 0)
        world.admit(WorkloadRequest("b", 0, 8, 100), 0)
        observed = []
        def fake_step(current, at_us=None, offset=0, *, book_link=False):
            observed.append((current.clock_us, tuple(sorted(current.requests)),
                             offset, book_link))
            return type("Cost", (), {"total_us": (5.0, 7.0, 9.0)[len(observed) - 1]})()
        with patch("elasticafd.layout_aware.evaluate_step", side_effect=fake_step):
            self.assertEqual(_forecast_total(world, 3, {"a": 1, "b": 3}), 21.0)
        self.assertEqual(observed, [(0.0, ("a", "b"), 0, True),
                                    (5.0, ("b",), 0, True),
                                    (12.0, ("b",), 0, True)])
        self.assertEqual(world.requests, {})
        self.assertEqual(world.blocks, {})

    def test_scenario_lengths_ignore_hidden_target_and_are_reproducible(self) -> None:
        cfg = tiny_config(policy="layout_greedy")
        worlds = []
        for target in (1, 10000):
            world = LayoutWorld(cfg, PrefixCatalog(()))
            world.admit(WorkloadRequest("a", 0, 8, target), 0)
            world.requests["a"].generated = 32
            worlds.append(world)
        paths = [_forecast_scenarios(world, 64) for world in worlds]
        self.assertEqual(paths[0], paths[1])
        self.assertEqual(len(paths[0]), cfg.simulation.layout_forecast_scenarios)
        self.assertTrue(all(1 <= path["a"] <= 64 for path in paths[0]))
        moved = deepcopy(worlds[0])
        moved.requests["a"].future_writer = GPU
        self.assertEqual(_forecast_scenarios(moved, 64), paths[0])

    def test_end_to_end_fixed_and_greedy(self) -> None:
        for policy, ranksets in (("layout_fixed", 1), ("layout_greedy", 1),
                                 ("layout_fixed", 4), ("layout_greedy", 4)):
            config = tiny_config(shared=True, policy=policy)
            config = replace(config, simulation=replace(
                config.simulation, layout_dense_ranksets=ranksets))
            result = run_simulation(config)
            self.assertEqual(result.summary["status"], "completed")
            self.assertEqual(result.summary["output_tokens"], 4)
            self.assertEqual(result.summary["cost_backend"], "layout_aware")
            self.assertEqual(result.summary["layout_dense_ranksets"], ranksets)
            self.assertEqual(result.summary["decision_interval_tokens"],
                             config.simulation.policy.decision_interval)
            self.assertEqual(result.summary["planning_horizon_tokens"],
                             config.simulation.policy.planning_horizon_tokens)
            self.assertEqual(result.summary["output_length_prior"]["scenario_count"],
                             config.simulation.layout_forecast_scenarios)
            self.assertEqual(len(result.requests), 2)
            if policy == "layout_greedy":
                self.assertGreater(result.summary["policy_decisions"], 0)
                self.assertGreaterEqual(result.summary["policy_search_wall_time_s"], 0)
                self.assertEqual(len(result.summary["candidate_screening"]),
                                 result.summary["policy_decisions"])
                self.assertGreater(result.summary["candidate_screening"][0]["rounds"][0]
                                   ["eligible_groups"], 0)

    def test_greedy_can_apply_interval_under_a_favorable_cost_setting(self) -> None:
        cfg = tiny_config(policy="layout_greedy")
        cfg = replace(
            cfg,
            system=replace(cfg.system,
                           gpu=replace(cfg.system.gpu, hbm_bandwidth_tbps=0.001),
                           link=replace(cfg.system.link, host_to_pim_gbps=100000,
                                        pim_to_host_gbps=100000)),
            simulation=replace(cfg.simulation, layout_pim_dense_scale=0.01,
                               layout_max_candidates=20,
                               policy=replace(cfg.simulation.policy,
                                              planning_horizon_tokens=16)),
        )
        result = run_simulation(cfg)
        self.assertGreater(result.summary["interval_actions"], 0)
        self.assertGreater(result.summary["prepared_physical_gib"], 0)


if __name__ == "__main__":
    unittest.main()
