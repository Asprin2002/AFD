from __future__ import annotations

from dataclasses import replace
from math import erf, sqrt
import unittest

from elasticafd.config import project_config_from_dict
from elasticafd.cost import ChimeCostModel
from elasticafd.dense_offload import (DenseOffloadCost, DenseOffloadSimulator,
                                     DensePlan, FFNEventModel, OnlineDensePolicy,
                                     ScriptedDensePolicy, plan_menu)
from elasticafd.joint import JointAFEventModel
from elasticafd.state import Location, RequestState, make_request_view, views_for_requests


def config(layers=4, output=4):
    c = project_config_from_dict(dict(model="gpt175b", hardware="chime_a100",
        workload=dict(type="static_batch", batch_size=4, input_tokens=2048, output_tokens=output),
        simulation=dict(max_batch_size=16, pipeline_model="rankset_event", record_every=1,
                        policy=dict(name="chime", decision_interval=2, planning_horizon_tokens=2,
                                    min_switch_interval=2, min_segment_tokens=2,
                                    event_driven_replan=True, switch_execution_us=100))))
    return replace(c, model=replace(c.model, layers=layers))


class DenseOffloadTests(unittest.TestCase):
    def test_device_contiguous_pim_qkv_moves_first_cut_without_returning_q(self):
        c = config(layers=2)
        c = replace(c, simulation=replace(
            c.simulation, subbatch_progression="device_contiguous",
        ))
        requests = [RequestState(str(i), 0, 128, 4) for i in range(4)]
        views = views_for_requests(requests)
        plan = DensePlan(1, 0, 1)
        model = DenseOffloadCost(c, (DensePlan(), plan),
                                 allow_conditional_ffn=True)
        events = model.measure(views, plan, events=True).cost.events
        by_id = {event.task_id: event for event in events}
        for subbatch in (0, 1):
            prefix = f"l0:s{subbatch}"
            inbound = by_id[f"{prefix}:qkv:input_h2d"]
            qkv = by_id[f"{prefix}:qkv"]
            attention = by_id[f"{prefix}:pim_attention"]
            outbound = by_id[f"{prefix}:o_proj_d2h"]
            ffn = by_id[f"{prefix}:ffn"]
            self.assertEqual(inbound.kind, "qkv_input_h2d")
            self.assertEqual(qkv.kind, "pim_qkv")
            self.assertGreaterEqual(qkv.start_us, inbound.end_us)
            self.assertGreaterEqual(attention.start_us, qkv.end_us)
            self.assertGreaterEqual(outbound.start_us, attention.end_us)
            self.assertGreaterEqual(ffn.start_us, outbound.end_us)
            self.assertNotIn(f"{prefix}:qkv:output_d2h", by_id)
            self.assertEqual(
                by_id[f"{prefix}:pim_attention_h2d"].duration_us, 0,
            )
        self.assertIn("visit:0:done", by_id)
        self.assertLessEqual(
            by_id["visit:1:done"].end_us,
            by_id["visit:2:start"].start_us,
        )

    def test_device_contiguous_pim_qkv_rejects_gpu_kv_branch(self):
        c = config(layers=2)
        c = replace(c, simulation=replace(
            c.simulation, subbatch_progression="device_contiguous",
        ))
        requests = [RequestState(str(i), 0, 128, 4) for i in range(4)]
        requests[0].switch_tail(Location.GPU, 0)
        requests[0].append_generated_token()
        plan = DensePlan(1, 0, 1)
        model = DenseOffloadCost(c, (DensePlan(), plan),
                                 allow_conditional_ffn=True)
        with self.assertRaisesRegex(ValueError, "all Attention and new KV"):
            model.measure(views_for_requests(requests), plan)

    def test_device_contiguous_pim_qkv_only_and_partial_oproj(self):
        c = config(layers=2)
        c = replace(c, simulation=replace(
            c.simulation, subbatch_progression="device_contiguous",
        ))
        views = views_for_requests([
            RequestState(str(i), 0, 128, 4) for i in range(4)
        ])
        for plan in (DensePlan(0, 0, 1), DensePlan(.5, 0, 1)):
            model = DenseOffloadCost(c, (DensePlan(), plan),
                                     allow_conditional_ffn=True)
            events = model.measure(views, plan, events=True).cost.events
            by_id = {event.task_id: event for event in events}
            self.assertNotIn("l0:s0:qkv:output_d2h", by_id)
            self.assertGreaterEqual(
                by_id["l0:s0:gpu_o_proj"].start_us,
                by_id["l0:s0:pim_attention"].end_us,
            )
            if plan.o_proj:
                self.assertGreaterEqual(
                    by_id["l0:s0:gpu_o_proj"].start_us,
                    by_id["l0:s0:o_proj_d2h"].end_us,
                )

    def test_pim_dense_bandwidth_sensitivity_does_not_change_attention(self):
        c = config(layers=2)
        c = replace(c, simulation=replace(
            c.simulation, subbatch_progression="device_contiguous",
        ))
        views = views_for_requests([
            RequestState(str(i), 0, 128, 4) for i in range(4)
        ])
        plan = DensePlan(1, 0, 1)
        slow = DenseOffloadCost(c, (DensePlan(), plan),
                                dense_bandwidth_scale=.5,
                                dense_compute_scale=100,
                                allow_conditional_ffn=True)
        fast = DenseOffloadCost(c, (DensePlan(), plan),
                                dense_bandwidth_scale=2,
                                dense_compute_scale=100,
                                allow_conditional_ffn=True)
        self.assertEqual(
            slow.measure(views, DensePlan()).cost.total_us,
            fast.measure(views, DensePlan()).cost.total_us,
        )
        self.assertGreater(
            slow.measure(views, plan).cost.total_us,
            fast.measure(views, plan).cost.total_us,
        )

    def test_device_contiguous_split_attention_and_local_ffn(self):
        c = config(layers=2)
        c = replace(c, simulation=replace(
            c.simulation, subbatch_progression="device_contiguous",
        ))
        requests = [RequestState(str(i), 0, 2048, 4) for i in range(4)]
        for request in requests:
            request.switch_tail(Location.GPU, 0)
            request.append_generated_token()
        views = views_for_requests(requests)
        plans = (DensePlan(), DensePlan(.5, 0), DensePlan(1, .125))
        cost = DenseOffloadCost(c, plans, allow_conditional_ffn=True)
        split_events = cost.measure(
            views, DensePlan(.5, 0), events=True,
        ).cost.events
        for event in split_events:
            if event.kind in ("pim_attention_d2h", "pim_merge_d2h"):
                self.assertEqual(event.duration_us, 0)
        events = cost.measure(views, DensePlan(1, .125), events=True).cost.events
        by_id = {event.task_id: event for event in events}
        for subbatch in (0, 1):
            prefix = f"l0:s{subbatch}"
            self.assertGreaterEqual(
                by_id[f"{prefix}:pim_attention_h2d"].start_us,
                by_id[f"{prefix}:gpu_attention"].end_us,
            )
            self.assertGreaterEqual(
                by_id[f"{prefix}:pim_merge"].start_us,
                by_id[f"{prefix}:pim_attention"].end_us,
            )
            self.assertGreaterEqual(
                by_id[f"{prefix}:pim_o_proj"].start_us,
                by_id[f"{prefix}:pim_merge"].end_us,
            )
            self.assertGreaterEqual(
                by_id[f"{prefix}:ffn:up"].start_us,
                by_id[f"{prefix}:pim_o_proj"].end_us,
            )
            self.assertNotIn(f"{prefix}:ffn:h2d", by_id)
            self.assertNotIn(f"{prefix}:gpu_partial_h2d", by_id)
        self.assertIn("visit:3:done", by_id)

    def test_device_contiguous_rejects_ffn_without_full_pim_oproj(self):
        c = config()
        c = replace(c, simulation=replace(
            c.simulation, subbatch_progression="device_contiguous",
        ))
        with self.assertRaisesRegex(ValueError, "full PIM O_proj"):
            DenseOffloadCost(c, (DensePlan(), DensePlan(0, .125)),
                             allow_conditional_ffn=True)

    def test_base_attention_event_uses_the_same_device_order(self):
        c = config(layers=2)
        c = replace(c, simulation=replace(
            c.simulation, subbatch_progression="device_contiguous",
        ))
        requests = [RequestState(str(i), 0, 2048, 4) for i in range(4)]
        for request in requests:
            request.switch_tail(Location.GPU, 0)
            request.append_generated_token()
        model = ChimeCostModel(
            c.model, c.system, pipeline_model="rankset_event",
            subbatch_progression="device_contiguous",
        )
        events = model.iteration_cost(views_for_requests(requests)).events
        by_id = {event.task_id: event for event in events}
        for subbatch in (0, 1):
            prefix = f"l0:s{subbatch}"
            self.assertGreaterEqual(
                by_id[f"{prefix}:h2d"].start_us,
                by_id[f"{prefix}:gpu_attention"].end_us,
            )
            self.assertGreaterEqual(
                by_id[f"{prefix}:d2h"].start_us,
                by_id[f"{prefix}:pim_merge"].end_us,
            )

    def test_conditional_capability_gate(self):
        c = config()
        with self.assertRaisesRegex(ValueError, "conditional capability"):
            DenseOffloadCost(c, plan_menu("ffn"))
        with self.assertRaisesRegex(ValueError, "integer"):
            FFNEventModel(c.model, c.system, ffn_fraction=.12345,
                          max_o_proj_fraction=0, allow_conditional_ffn=True)

    def test_neuron_partition_numeric_identity(self):
        # x W1 -> elementwise GELU -> W2; output bias applied once.
        x = [0.1, -2., 1.3]
        w1 = [[(i-j)*.3 for j in range(8)] for i in range(3)]
        w2 = [[(i+j-4)*.17 for j in range(3)] for i in range(8)]
        def shard(start, end):
            intermediate = [sum(x[i]*w1[i][j] for i in range(3)) for j in range(start, end)]
            activated = [.5*v*(1+erf(v/sqrt(2))) for v in intermediate]
            return [sum(activated[j-start]*w2[j][h] for j in range(start,end)) for h in range(3)]
        full = shard(0, 8)
        for split in (0, 1, 2, 4, 8):
            a, b = shard(0, split), shard(split, 8)
            for h in range(3):
                self.assertAlmostEqual(full[h]+.2, a[h]+b[h]+.2, places=12)

    def test_zero_ffn_regresses_existing_joint_graph(self):
        c = config()
        views = views_for_requests([RequestState(str(i),0,2048,4) for i in range(4)])
        base = JointAFEventModel(c.model,c.system,max_o_proj_pim_fraction=1)
        adapter = DenseOffloadCost(c,plan_menu("joint"),allow_conditional_ffn=True)
        for o in (0,.25,1):
            expected = base.iteration_cost(views,o)
            actual = adapter.measure(views,DensePlan(o,0),events=True)
            self.assertEqual(expected.total_us,actual.cost.total_us)
            self.assertEqual(expected.events,actual.cost.events)

    def test_transfer_size_and_next_layer_dependencies(self):
        c = config()
        views = views_for_requests([RequestState(str(i),0,2048,4) for i in range(4)])
        cost = DenseOffloadCost(c,plan_menu("ffn"),allow_conditional_ffn=True)
        for f in (.03125,.5):
            measured = cost.measure(views,DensePlan(0,f),events=True)
            events = measured.cost.events
            by_id = {e.task_id:e for e in events}
            for e in events:
                if e.kind == "ffn_d2h":
                    expected_bytes = 2*c.model.hidden_size*4
                    self.assertAlmostEqual(e.duration_us,cost.backends[f]._transfer_us(expected_bytes,"d2h"))
                if e.kind == "gpu_qkv" and e.layer:
                    last = by_id[f"l{e.layer-1}:s{e.subbatch}:ffn"]
                    self.assertGreaterEqual(e.start_us,last.end_us)
            for resource in {r for e in events for r in e.resources}:
                used = sorted((e.start_us,e.end_us) for e in events if resource in e.resources)
                for previous,current in zip(used,used[1:]):
                    self.assertLessEqual(previous[1],current[0]+1e-8)
            self.assertGreaterEqual(measured.cost.total_us,measured.cost.gpu_busy_us)
            self.assertGreaterEqual(measured.cost.total_us,measured.cost.pim_core_busy_us)

    def test_strict_af_stage_order_with_split_and_dense_offload(self):
        c = config(layers=3)
        requests = [RequestState(str(i), 0, 2048, 4) for i in range(4)]
        for request in requests[:2]:
            request.switch_tail(Location.GPU, 0)
            request.append_generated_token()
        views = views_for_requests(requests)
        cost = DenseOffloadCost(c, plan_menu("joint"), allow_conditional_ffn=True)
        for plan in (DensePlan(), DensePlan(1.0, .125, 1.0)):
            events = cost.measure(views, plan, events=True).cost.events
            by_id = {event.task_id: event for event in events}
            pair_start = by_id["phase:3:start"].end_us
            pair_done = by_id["phase:3:done"].start_us
            next_start = by_id["phase:4:start"].end_us
            for task_id in (
                "l1:s0:pim_attention", "l1:s0:gpu_attention", "l0:s1:ffn",
            ):
                self.assertGreaterEqual(by_id[task_id].start_us, pair_start)
                self.assertLessEqual(by_id[task_id].end_us, pair_done)
            self.assertGreaterEqual(next_start, pair_done)
            for task_id in (
                "l1:s1:pim_attention", "l1:s1:gpu_attention", "l1:s0:ffn",
            ):
                self.assertGreaterEqual(by_id[task_id].start_us, next_start)
            for layer in range(c.model.layers):
                a0 = by_id[f"l{layer}:s0:attention_done"]
                a1 = by_id[f"l{layer}:s1:attention_done"]
                for event in events:
                    if event.layer == layer and event.subbatch == 1 and event.kind in (
                        "gpu_attention", "pim_attention"
                    ):
                        self.assertGreaterEqual(event.start_us, a0.end_us)
                self.assertGreaterEqual(
                    by_id[f"l{layer}:s1:ffn"].start_us,
                    by_id[f"l{layer}:s0:ffn"].end_us,
                )
                if layer:
                    prior_a1 = by_id[f"l{layer-1}:s1:attention_done"]
                    for event in events:
                        if event.layer == layer and event.subbatch == 0 and event.kind in (
                            "gpu_attention", "pim_attention"
                        ):
                            self.assertGreaterEqual(event.start_us, prior_a1.end_us)
                self.assertGreaterEqual(a1.end_us, a0.end_us)

    def test_recorded_dense_events_expose_paired_phase_boundaries(self):
        c = config(layers=2)
        c = replace(c, simulation=replace(
            c.simulation, record_pipeline_events=True,
        ))
        cost = DenseOffloadCost(c, plan_menu("control"))
        views = views_for_requests([
            RequestState(str(i), 0, 2048, 1) for i in range(2)
        ])
        events = cost.iteration_cost(views).events
        self.assertIn("phase:3:start", {event.task_id for event in events})
        self.assertIn("phase:3:done", {event.task_id for event in events})

    def test_full_ffn_has_no_gpu_matrix_but_has_return_and_cast(self):
        c = config()
        cost = DenseOffloadCost(c,(DensePlan(),DensePlan(0,1)),allow_conditional_ffn=True)
        m = cost.measure(views_for_requests([RequestState("r",0,2048,4)]),DensePlan(0,1),events=True)
        self.assertEqual(m.gpu_ffn_us,0)
        self.assertGreater(m.pim_ffn_us,0)
        self.assertIn("ffn_reduce",[e.kind for e in m.cost.events])

    def test_capacity_and_dense_scale_do_not_accelerate_attention(self):
        c = config()
        a = DenseOffloadCost(c,plan_menu("joint"),allow_conditional_ffn=True)
        b = DenseOffloadCost(c,plan_menu("joint"),dense_compute_scale=4,allow_conditional_ffn=True)
        self.assertEqual(a.pim_kv_capacity_bytes,b.pim_kv_capacity_bytes)
        self.assertLess(a.gpu_kv_capacity_bytes,c.system.gpu_kv_capacity_bytes(c.model))
        self.assertGreater(a.weight_bytes,0)
        views = views_for_requests([RequestState("r",0,2048,4)])
        self.assertEqual(a.measure(views,DensePlan()).cost,b.measure(views,DensePlan()).cost)
        self.assertGreater(a.measure(views,DensePlan(0,.25)).pim_ffn_us,
                           b.measure(views,DensePlan(0,.25)).pim_ffn_us)

    def test_exact_cache_refreshes_logical_metadata(self):
        c = config()
        cost = DenseOffloadCost(c,plan_menu("joint"),allow_conditional_ffn=True)
        for n in (2048,2049,2050,2051):
            views = views_for_requests([RequestState(str(i),0,n,4) for i in range(4)])
            cached = cost.measure(views,DensePlan(0,.125))
            fresh = cost.measure(views,DensePlan(0,.125),events=True)
            self.assertEqual(cached.cost.total_us,fresh.cost.total_us)
            self.assertEqual(cached.cost.subbatches,fresh.cost.subbatches)
        self.assertLessEqual(len(cost.measurements),256)

    def test_event_cost_cache_obeys_configured_bound(self):
        c = config()
        cost = DenseOffloadCost(
            c,
            plan_menu("control"),
            event_cost_cache_max_entries=1,
        )
        cost.measure(
            views_for_requests([RequestState("a", 0, 128, 16)]),
            DensePlan(),
        )
        cost.measure(
            views_for_requests([RequestState("b", 0, 1024, 16)]),
            DensePlan(),
        )
        self.assertEqual(cost.event_cost_cache_stats["entries"], 1)
        self.assertEqual(cost.event_cost_cache_stats["max_entries"], 1)

    def test_online_interface_ignores_future_and_keeps_history(self):
        c = config(output=32)
        cost = DenseOffloadCost(c,plan_menu("joint"),allow_conditional_ffn=True)
        policy = OnlineDensePolicy(c.simulation.policy,cost,"joint")
        requests = [RequestState(str(i),0,2048,32) for i in range(4)]
        history = [tuple((s.start_token,s.end_token,s.location) for s in r.segments) for r in requests]
        # Base policy hook deliberately ignores all pending requests.
        policy.update_runtime_context(clock_us=0,pending=[object()],max_batch_size=16,initial_location=Location.PIM)
        policy.decide(requests,0,force=True)
        self.assertEqual(history,[tuple((s.start_token,s.end_token,s.location) for s in r.segments) for r in requests])
        self.assertFalse(policy.summary_metadata()["future_arrivals_visible"])
        for d in policy.decisions:
            self.assertLess(d["objective_after_us"],d["objective_before_us"])

    def test_residence_and_joint_menu_inclusion(self):
        joint = set(plan_menu("joint"))
        self.assertTrue(set(plan_menu("oproj")) <= joint)
        self.assertTrue(set(plan_menu("ffn")) <= joint)
        self.assertTrue(set(plan_menu("qkv")) <= joint)
        self.assertEqual(set(plan_menu("f_all")), joint)
        self.assertTrue(
            {
                DensePlan(o_proj=o, ffn=f, qkv=q)
                for o in (0.0, 1.0)
                for f in (0.0, 1.0)
                for q in (0.0, 1.0)
            }
            <= joint
        )
        c = config(output=16)
        cost = DenseOffloadCost(c,plan_menu("ffn"),allow_conditional_ffn=True)
        policy = OnlineDensePolicy(c.simulation.policy,cost,"ffn")
        r = [RequestState("r",0,2048,16)]
        policy.decide(r,0,force=True)
        first = cost.plan
        policy.decide(r,1,force=True)
        self.assertEqual(first,cost.plan)

    def test_dense_gain_cannot_subsidize_tiny_irreversible_kv_gain(self):
        c = config(layers=96,output=4096)
        c = replace(c,simulation=replace(c.simulation,policy=replace(
            c.simulation.policy,decision_interval=64,planning_horizon_tokens=64,
            min_switch_interval=64,min_segment_tokens=64)))
        outcomes=[]
        for threshold in (0,.002):
            # Freeze the original O_proj/FFN menu so this remains a focused
            # test of the KV marginal guard rather than the expanded QKV menu.
            legacy = tuple(
                plan for plan in plan_menu("joint")
                if plan.qkv == 0 and plan.ffn <= 0.5
            )
            cost=DenseOffloadCost(c,legacy,allow_conditional_ffn=True)
            policy=OnlineDensePolicy(c.simulation.policy,cost,"joint",
                                     kv_marginal_gain_threshold=threshold)
            r=RequestState("r",0,2048,4096)
            policy.decide([r],0,force=True)
            outcomes.append((r.tail_location,cost.plan,policy.kv_marginal_guard_vetoes))
        self.assertEqual(outcomes[0][0],Location.GPU)
        self.assertEqual(outcomes[1][0],Location.PIM)
        self.assertEqual(outcomes[0][1],outcomes[1][1])
        self.assertGreater(outcomes[1][2],0)

    def test_lifetime_objective_uses_known_remaining_output(self):
        c = config(layers=96, output=16384)
        cost = DenseOffloadCost(
            c, plan_menu("joint"), allow_conditional_ffn=True
        )
        bounded = OnlineDensePolicy(
            c.simulation.policy,
            cost,
            "joint",
            lifetime_objective=False,
        )
        short_request = RequestState("short", 0, 8192, 16384)
        bounded.decide([short_request], 0, force=True)
        self.assertEqual(short_request.tail_location, Location.PIM)

        lifetime_cost = DenseOffloadCost(
            c, plan_menu("joint"), allow_conditional_ffn=True
        )
        lifetime = OnlineDensePolicy(
            c.simulation.policy,
            lifetime_cost,
            "joint",
            lifetime_objective=True,
        )
        request = RequestState("lifetime", 0, 8192, 16384)
        lifetime.decide([request], 0, force=True)
        self.assertEqual(request.tail_location, Location.GPU)
        self.assertEqual(lifetime_cost.plan, DensePlan(ffn=0.25))
        self.assertEqual(
            lifetime.summary_metadata()["objective"],
            "two_endpoint_event_cost_until_next_known_completion",
        )

    def test_whole_qkv_has_explicit_transfers_and_attention_dependencies(self):
        c = config(layers=2)
        requests = [RequestState(str(i), 0, 2048, 4) for i in range(4)]
        for request in requests:
            request.switch_tail(Location.GPU, 0)
            request.append_generated_token()
        views = views_for_requests(requests)
        plan = DensePlan(qkv=1.0)
        cost = DenseOffloadCost(
            c,
            (DensePlan(), plan),
            allow_conditional_ffn=True,
        )
        measured = cost.measure(views, plan, events=True)
        events = measured.cost.events
        by_id = {event.task_id: event for event in events}
        self.assertEqual(measured.gpu_qkv_us, 0.0)
        self.assertGreater(measured.pim_qkv_us, 0.0)
        self.assertNotIn("gpu_qkv", {event.kind for event in events})

        partitions = cost.partition_views(views)
        backend = cost.backends[(1.0, 0.0)]
        for layer in range(c.model.layers):
            for subbatch, batch_views in enumerate(partitions):
                prefix = f"l{layer}:s{subbatch}"
                input_event = by_id[f"{prefix}:qkv:input_h2d"]
                qkv_event = by_id[f"{prefix}:qkv"]
                output_event = by_id[f"{prefix}:qkv:output_d2h"]
                local_event = by_id[f"{prefix}:pim_attention_h2d"]
                gpu_attention = by_id[f"{prefix}:gpu_attention"]
                pim_attention = by_id[f"{prefix}:pim_attention"]
                input_bytes = (
                    len(batch_views) * c.model.output_bytes_per_token_per_layer
                )
                output_bytes = len(batch_views) * (
                    c.model.q_bytes_per_token_per_layer
                    + c.model.new_kv_bytes_per_token_per_layer
                )
                self.assertAlmostEqual(
                    input_event.duration_us,
                    backend._transfer_us(input_bytes, "h2d"),
                )
                self.assertAlmostEqual(
                    output_event.duration_us,
                    backend._transfer_us(output_bytes, "d2h"),
                )
                self.assertEqual(local_event.kind, "pim_attention_input_local")
                self.assertEqual(local_event.duration_us, 0.0)
                self.assertGreaterEqual(qkv_event.start_us, input_event.end_us)
                self.assertGreaterEqual(output_event.start_us, qkv_event.end_us)
                self.assertGreaterEqual(gpu_attention.start_us, output_event.end_us)
                self.assertGreaterEqual(pim_attention.start_us, local_event.end_us)

    def test_full_simulator_controls_capacity_and_completion(self):
        c = config(output=5)
        for mode in ("control","oproj","ffn","joint"):
            sim = DenseOffloadSimulator(c,mode=mode,allow_conditional_ffn=True)
            result = sim.run()
            self.assertEqual(result.summary["status"],"completed")
            self.assertEqual(result.summary["output_tokens"],20)
            self.assertEqual(result.summary["h2d_transfer_gib"],sum(r["h2d_bytes"] for r in result.iterations)/2**30)
            self.assertLessEqual(result.summary["peak_pim_kv_gib"],result.summary["pim_kv_capacity_gib"])
            for row in result.requests:
                self.assertTrue(row["segment_layout"].startswith("pim:0-"))
            self.assertEqual(sim.pim_used_bytes,0)
            self.assertEqual(sim.gpu_used_bytes,0)

    def test_scripted_offline_plan_executes_exact_kv_segments(self):
        c = config(output=5)
        timeline = [
            {
                "start": 0,
                "end": 5,
                "gpu_writers": 4,
                "qkv": 0.0,
                "o_proj": 0.0,
                "ffn": 0.25,
            }
        ]
        sim = DenseOffloadSimulator(
            c,
            mode="joint",
            allow_conditional_ffn=True,
            scripted_timeline=timeline,
        )
        self.assertIsInstance(sim.policy, ScriptedDensePolicy)
        result = sim.run()
        self.assertEqual(result.summary["status"], "completed")
        self.assertEqual(
            result.summary["objective"],
            "offline_precomputed_timeline",
        )
        self.assertEqual(result.summary["operator_switches"], 1)
        for request in result.requests:
            self.assertEqual(request["pim_tokens_final"], 2048)
            self.assertEqual(request["gpu_tokens_final"], 5)


if __name__ == "__main__":
    unittest.main()
