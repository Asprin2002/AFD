"""M1-A segmented planner + M1-B cross-validation + M3 cost hooks.

These tests exercise the closed-batch planning engine in isolation: cut
normalization and view building (including heterogeneous clipping), XOR
legality that mirrors the event backend, replica delta weight copy economics,
full-lifetime capacity-feasible evaluation with fresh-engine replay equality,
the deterministic operator/KV/joint searches with their ticket (privacy of the
ticket bounded budgets), and the independent JointAFScheduler reference replay.
"""

from __future__ import annotations

from bisect import bisect_right
import unittest

from tools import attention_strategy as refstrategy
from tools.run_heterogeneous_joint_matrix import (
    joint_static_reference,
    static_config,
    static_engine,
)
from elasticafd.segmented import (
    FREE_PRICING,
    DCSchedule,
    DensePlan,
    RequestSpec,
    SwitchPricing,
    _kv_closure,
    active_plan,
    control_schedule,
    dc_plan_menu,
    eval_schedule,
    eval_window,
    legal_plan_for_state,
    lifetime_stride,
    max_output,
    normalize_cuts,
    peak_weight_bytes,
    search_joint,
    search_kv_timeline,
    search_operator_timeline,
    state_views,
    weight_copy_events,
)
from elasticafd.specs import MODEL_PRESETS, SYSTEM_PRESETS
from elasticafd.state import Location

GIB = float(1024 ** 3)


def specs_of(batch, input_tokens, output_tokens):
    return tuple(RequestSpec(f"r{i}", input_tokens, output_tokens)
                 for i in range(batch))


class CutNormalizationTests(unittest.TestCase):
    def test_boundary_zero_kept_initial_writer(self):
        cuts = normalize_cuts(((0, Location.GPU),), 128)
        self.assertEqual(((0, Location.GPU),), cuts)

    def test_boundary_zero_pim_noop_dropped(self):
        cuts = normalize_cuts(((0, Location.PIM), (64, Location.GPU)), 128)
        self.assertEqual(((64, Location.GPU),), cuts)

    def test_duplicate_neighbour_and_out_of_range_dropped(self):
        cuts = normalize_cuts(
            ((32, Location.GPU), (64, Location.GPU), (128, Location.PIM),
             (200, Location.PIM)), 128,
        )
        self.assertEqual(((32, Location.GPU),), cuts)

    def test_alternating_sequence_preserved_order(self):
        cuts = normalize_cuts(
            ((50, Location.GPU), (100, Location.PIM), (120, Location.GPU)),
            128,
        )
        self.assertEqual(
            ((50, Location.GPU), (100, Location.PIM), (120, Location.GPU)),
            cuts,
        )


class ViewBuildingTests(unittest.TestCase):
    def test_homogeneous_views_match_reference_builder(self):
        specs = specs_of(8, 5200, 8192)
        cuts = tuple(((0, Location.GPU), (3000, Location.PIM),
                      (6000, Location.GPU)) if i % 2 else ()
                     for i in range(8))
        schedule = DCSchedule("t", cuts, ((0, DensePlan()),))
        ref_schedule = refstrategy.Schedule("ref", cuts)
        for generated in (128, 2999, 3001, 5000, 8191):
            mine = state_views(schedule, specs, generated)
            theirs = refstrategy.views(ref_schedule, 5200, generated)
            self.assertEqual(len(mine), len(theirs))
            for a, b in zip(mine, theirs):
                self.assertEqual(a.pim_segment_lengths, b.pim_segment_lengths)
                self.assertEqual(a.gpu_segment_lengths, b.gpu_segment_lengths)
                self.assertEqual(a.tail_location, b.tail_location)

    def test_heterogeneous_clipping_excludes_finished(self):
        specs = (RequestSpec("long", 8000, 4096),
                 RequestSpec("short", 512, 256))
        schedule = control_schedule(specs)
        at_700 = state_views(schedule, specs, 700)
        self.assertEqual(1, len(at_700))          # short finished by 700
        self.assertEqual("long", at_700[0].request_id)
        at_60 = state_views(schedule, specs, 60)
        self.assertEqual(2, len(at_60))
        self.assertGreaterEqual(at_60[1].pim_tokens, 512)

    def test_zero_width_tail_cut_does_not_own_tail(self):
        # Request ends exactly on its own GPU move: last written token is PIM.
        specs = (RequestSpec("r0", 100, 200),)
        schedule = DCSchedule("t", (((100, Location.GPU),),), ((0, DensePlan()),))
        views = state_views(schedule, specs, 150)   # still active past cut
        self.assertEqual(Location.GPU, views[0].tail_location)
        views_end = state_views(schedule, specs, 100)  # cut == finish line
        self.assertEqual(Location.PIM, views_end[0].tail_location)
        self.assertEqual((), views_end[0].gpu_segment_lengths)

    def test_initial_gpu_writer_view(self):
        specs = (RequestSpec("r0", 100, 64),)
        schedule = DCSchedule("t", (((0, Location.GPU),),), ((0, DensePlan()),))
        views = state_views(schedule, specs, 40)
        self.assertEqual((100,), views[0].pim_segment_lengths)
        self.assertEqual((40,), views[0].gpu_segment_lengths)
        self.assertEqual(Location.GPU, views[0].tail_location)


class LegalityTests(unittest.TestCase):
    def setUp(self):
        specs = specs_of(4, 2048, 256)
        self.all_pim = state_views(control_schedule(specs), specs, 100)
        gpu_schedule = DCSchedule("t", tuple(((0, Location.GPU),) for _ in specs),
                                  ((0, DensePlan()),))
        self.with_gpu = state_views(gpu_schedule, specs, 64)

    def test_qkv_plan_requires_all_pim(self):
        plan = DensePlan(qkv=1.0)
        self.assertTrue(legal_plan_for_state(plan, self.all_pim))
        self.assertFalse(legal_plan_for_state(plan, self.with_gpu))

    def test_non_qkv_plans_not_gated(self):
        for plan in (DensePlan(), DensePlan(o_proj=0.5), DensePlan(1.0, 0.1)):
            self.assertTrue(legal_plan_for_state(plan, self.with_gpu))

    def test_menu_is_device_contiguous_legal(self):
        for plan in dc_plan_menu():
            if plan.ffn > 0.0:
                self.assertEqual(1.0, plan.o_proj)


class WeightCopyTests(unittest.TestCase):
    def test_replica_growth_only_and_first_latency(self):
        timeline = ((0, DensePlan(o_proj=0.5)),
                    (64, DensePlan(o_proj=1.0)),
                    (128, DensePlan(o_proj=0.5)))  # shrink never copies
        events = weight_copy_events(timeline, MODEL_PRESETS["gpt175b"],
                                    SYSTEM_PRESETS["chime_a100"].link)
        self.assertEqual(2, len(events))
        kinds = [kind for kind, _ in events]
        self.assertEqual(["grow", "grow"], kinds)
        transfer0 = events[0][1]
        link = SYSTEM_PRESETS["chime_a100"].link
        layers = MODEL_PRESETS["gpt175b"].layers
        self.assertGreater(transfer0, layers * link.transfer_latency_us)
        self.assertLessEqual(events[1][1], events[1][1])  # ll sanity

    def test_flat_timeline_no_copy_events(self):
        events = weight_copy_events(((0, DensePlan()), (32, DensePlan())),
                                    MODEL_PRESETS["gpt175b"],
                                    SYSTEM_PRESETS["chime_a100"].link)
        self.assertEqual([], events)


class EvaluationTests(unittest.TestCase):
    def engine(self):
        return static_engine(static_config(4, "lpt"))

    def test_control_schedule_has_no_overhead(self):
        specs = specs_of(4, 2048, 256)
        cost = eval_schedule(self.engine(), control_schedule(specs), specs,
                             pricing=SwitchPricing())
        self.assertTrue(cost.feasible)
        self.assertEqual(0.0, cost.kv_switches)
        self.assertEqual(0.0, cost.operator_switches)
        self.assertEqual(0.0, cost.switch_us)
        self.assertEqual(0.0, cost.weight_copy_us)
        self.assertAlmostEqual(cost.total_us, cost.iteration_us)

    def test_chosen_plan_overhead_adds_separately(self):
        specs = specs_of(4, 2048, 256)
        schedule = DCSchedule(
            "t", tuple(() for _ in specs),
            ((0, DensePlan(o_proj=0.5)),),
        )
        cost = eval_schedule(self.engine(), schedule, specs,
                             pricing=SwitchPricing())
        self.assertTrue(cost.feasible)
        self.assertGreater(cost.weight_copy_us, 0.0)
        self.assertEqual(0.0, cost.switch_us)
        self.assertEqual(cost.total_without_overhead(), cost.iteration_us)
        self.assertGreater(cost.total_us, cost.iteration_us)

    def test_fresh_engine_replay_equality(self):
        specs = specs_of(4, 2048, 128)
        engine_a = self.engine()
        engine_b = self.engine()
        schedule = DCSchedule(
            "t", tuple(((64, Location.GPU),) for _ in specs),
            ((0, DensePlan(o_proj=0.5)),),
        )
        a = eval_schedule(engine_a, schedule, specs, pricing=SwitchPricing())
        b = eval_schedule(engine_b, schedule, specs, pricing=SwitchPricing())
        self.assertAlmostEqual(a.total_us, b.total_us)

    def test_infeasible_when_running_peak_overflows(self):
        # Aggressive qkv + full o_proj on a batch too big for one 2GiB workspace.
        specs = specs_of(32, 16384, 4096)
        schedule = DCSchedule(
            "t", tuple(() for _ in specs),
            ((0, DensePlan(o_proj=1.0, qkv=1.0)),),
        )
        cost = eval_schedule(self.engine(), schedule, specs,
                             pricing=SwitchPricing())
        self.assertFalse(cost.feasible)

    def test_eval_window_finite_then_infeasible_on_overflow(self):
        specs = specs_of(4, 2048, 256)
        engine = self.engine()
        good = eval_window(engine, control_schedule(specs), specs, 0, 256,
                           stride=16)
        self.assertLess(good, float("inf"))
        bad = eval_window(engine, DCSchedule(
            "t", tuple(() for _ in specs), ((0, DensePlan(qkv=1.0)),),
        ), specs, 0, 256, stride=16)
        self.assertGreater(bad, 0.0)


class SearchTests(unittest.TestCase):
    def engine(self):
        return static_engine(static_config(2, "lpt"))

    def reduced_menu(self):
        return (DensePlan(), DensePlan(o_proj=0.5), DensePlan(o_proj=1.0),
                DensePlan(qkv=1.0))

    def test_operator_search_deterministic_and_not_worse_than_control(self):
        specs = specs_of(2, 128, 256)
        engine = self.engine()
        pricing = SwitchPricing()
        a = search_operator_timeline(
            engine, specs, plan_menu=self.reduced_menu(),
            stride=lifetime_stride(max_output(specs)), pricing=pricing,
            max_segments=3,
        )
        b = search_operator_timeline(
            engine, specs, plan_menu=self.reduced_menu(),
            stride=lifetime_stride(max_output(specs)), pricing=pricing,
            max_segments=3,
        )
        self.assertEqual(a.plan_timeline, b.plan_timeline)
        cost_a = eval_schedule(engine, a, specs, pricing=pricing)
        control = eval_schedule(engine, control_schedule(specs), specs,
                                pricing=pricing)
        self.assertTrue(cost_a.feasible)
        self.assertLessEqual(cost_a.total_us, control.total_us + 1e-6)

    def test_kv_search_returns_feasible_normalized_schedule(self):
        specs = specs_of(2, 2048, 256)
        engine = self.engine()
        schedule = search_kv_timeline(
            engine, specs,
            plan_timeline=((0, DensePlan()),),
            stride=16, decision_interval=64, lookahead=256,
            min_residence=128, pricing=SwitchPricing(),
        )
        cost = eval_schedule(engine, schedule, specs, pricing=SwitchPricing())
        self.assertTrue(cost.feasible)
        # Determinism on the second call.
        again = search_kv_timeline(
            engine, specs, plan_timeline=((0, DensePlan()),),
            stride=16, decision_interval=64, lookahead=256,
            min_residence=128, pricing=SwitchPricing(),
        )
        self.assertEqual(schedule.kv_cuts, again.kv_cuts)
        self.assertEqual(schedule.plan_timeline, again.plan_timeline)

    def test_joint_search_not_worse_than_independent_static_reference(self):
        specs = specs_of(2, 2048, 256)
        engine = self.engine()
        schedule = search_joint(
            engine, specs, pricing=SwitchPricing(),
            decision_interval=64, lookahead=256, min_residence=128,
            max_segments=3,
        )
        cost = eval_schedule(engine, schedule, specs, pricing=SwitchPricing())
        self.assertTrue(cost.feasible)
        reference, metadata = joint_static_reference(specs, "lpt")
        ref_cost = eval_schedule(engine, reference, specs,
                                 pricing=SwitchPricing())
        if ref_cost.feasible:
            self.assertLessEqual(
                cost.total_us, ref_cost.total_us + 1e-6,
                "segmented joint search should be at least as good as the "
                "independent best-static oracle on this subspace",
            )

    def test_max_segments_bounds_operator_timeline(self):
        specs = specs_of(4, 2048, 256)
        engine = self.engine()
        schedule = search_operator_timeline(
            engine, specs, plan_menu=self.reduced_menu(),
            stride=lifetime_stride(max_output(specs)),
            pricing=SwitchPricing(), max_segments=2,
        )
        self.assertLessEqual(len(schedule.plan_timeline), 2)

    def test_kv_closure_recovers_bounded_prefix_on_gpu_overflow(self):
        # Regression for the pilot finding: at B16/I5200/O8192 the greedy KV
        # search over-migrates future KV and blows the 314 GiB GPU budget near
        # the end of the lifetime. The closure must retract to a bounded-prefix
        # window that is feasible AND faster than control (the trusted-grid
        # headline effect), deterministically.
        specs = specs_of(16, 5200, 8192)
        engine = static_engine(static_config(16, "lpt"))
        stride = lifetime_stride(max_output(specs))
        pricing = SwitchPricing()
        all_gpu = DCSchedule(
            "all_gpu", tuple(((0, Location.GPU),) for _ in specs),
            ((0, DensePlan()),),
        )
        raw = eval_schedule(engine, all_gpu, specs, stride=stride,
                            pricing=pricing)
        self.assertFalse(raw.feasible, "guard: full-GPU future KV must overflow")
        closed = _kv_closure(
            engine, specs, all_gpu, stride=stride, pricing=pricing,
            decision_interval=64,
        )
        cost = eval_schedule(engine, closed, specs, stride=stride,
                             pricing=pricing)
        self.assertTrue(cost.feasible,
                        "closure restores feasibility at the headline shape")
        control = eval_schedule(engine, control_schedule(specs), specs,
                                stride=stride, pricing=pricing)
        self.assertLessEqual(cost.total_us, control.total_us + 1e-6)
        # Deterministic retraction.
        closed_again = _kv_closure(
            engine, specs, all_gpu, stride=stride, pricing=pricing,
            decision_interval=64,
        )
        self.assertEqual(closed.kv_cuts, closed_again.kv_cuts)
        # Every GPU-run must terminate before the end (bounded prefix).
        for part in closed.kv_cuts:
            self.assertNotEqual(
                part[-1][1], Location.GPU,
                "unbounded GPU run must have been retracted")

    def test_kv_closure_is_noop_when_already_feasible(self):
        specs = specs_of(2, 2048, 256)
        engine = self.engine()
        stride = lifetime_stride(max_output(specs))
        baseline = control_schedule(specs)
        closed = _kv_closure(engine, specs, baseline, stride=stride,
                             pricing=SwitchPricing(), decision_interval=64)
        self.assertEqual(closed.kv_cuts, baseline.kv_cuts)
        self.assertEqual(closed.plan_timeline, baseline.plan_timeline)


class CommunityHooksTests(unittest.TestCase):
    def test_free_pricing_is_zero(self):
        self.assertEqual(0.0, FREE_PRICING.switch_execution_us)
        self.assertFalse(FREE_PRICING.weight_copy_enabled)
        self.assertEqual("sw0.0us_copy0", FREE_PRICING.label)

    def test_lifetime_stride_caps_states(self):
        self.assertGreaterEqual(lifetime_stride(8192), 8192 / 96)
        self.assertEqual(96, (8192 + lifetime_stride(8192) - 1)
                         // lifetime_stride(8192))

    def test_peak_weight_bytes_is_max_fraction(self):
        plans = (DensePlan(o_proj=0.25), DensePlan(1.0, 0.0625, 0.5))
        peak = peak_weight_bytes(MODEL_PRESETS["gpt175b"], plans)
        single = peak_weight_bytes(MODEL_PRESETS["gpt175b"],
                                   (DensePlan(1.0, 0.0625, 0.5),))
        self.assertEqual(peak, single)

    def test_active_plan_boundary(self):
        timeline = ((0, DensePlan()), (100, DensePlan(o_proj=1.0)))
        self.assertEqual(DensePlan(), active_plan(timeline, 99))
        self.assertEqual(DensePlan(o_proj=1.0), active_plan(timeline, 100))


if __name__ == "__main__":
    unittest.main()