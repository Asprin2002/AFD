from __future__ import annotations

import unittest

from elasticafd.pipeline import (
    PipelineTask, order_two_subbatch_device_segments,
    order_two_subbatch_stages, schedule_pipeline,
)


class PipelineSchedulerTests(unittest.TestCase):
    @staticmethod
    def _two_stream_tasks(layers: int, *, slow_second_attention: bool = False):
        tasks = []
        for layer in range(layers):
            for subbatch in (0, 1):
                prefix = f"l{layer}:s{subbatch}"
                own_prior = (
                    (f"l{layer-1}:s{subbatch}:ffn",) if layer else ()
                )
                tasks.extend((
                    PipelineTask(
                        f"{prefix}:qkv", 1.0, ("gpu",), own_prior,
                        "qkv", layer, subbatch, priority=4,
                    ),
                    PipelineTask(
                        f"{prefix}:attention",
                        20.0 if slow_second_attention and subbatch else 4.0,
                        (f"pim:{subbatch}" if slow_second_attention else "pim:0",),
                        (f"{prefix}:qkv",),
                        "pim_attention", layer, subbatch, priority=1,
                    ),
                    PipelineTask(
                        f"{prefix}:attention_done", 0.0, (),
                        (f"{prefix}:attention",),
                        "attention_done", layer, subbatch,
                    ),
                    PipelineTask(
                        f"{prefix}:ffn", 4.0, ("gpu",),
                        (f"{prefix}:attention_done",),
                        "proj_ff", layer, subbatch, priority=3,
                    ),
                ))
        return tasks

    def test_ordered_two_streams_overlap_without_multi_layer_overtake(self) -> None:
        events = schedule_pipeline(
            order_two_subbatch_stages(self._two_stream_tasks(3))
        ).events
        by_id = {event.task_id: event for event in events}
        # The intended A/F overlap exists in both directions at a layer edge.
        a1 = by_id["l0:s1:attention"]
        f0 = by_id["l0:s0:ffn"]
        self.assertLess(a1.start_us, f0.end_us)
        self.assertLess(f0.start_us, a1.end_us)
        a0_next = by_id["l1:s0:attention"]
        f1 = by_id["l0:s1:ffn"]
        self.assertLess(a0_next.start_us, f1.end_us)
        self.assertLess(f1.start_us, a0_next.end_us)

        for layer in range(3):
            self.assertGreaterEqual(
                by_id[f"l{layer}:s1:qkv"].start_us,
                by_id[f"l{layer}:s0:qkv"].end_us,
            )
            self.assertGreaterEqual(
                by_id[f"l{layer}:s1:attention"].start_us,
                by_id[f"l{layer}:s0:attention_done"].end_us,
            )
            self.assertGreaterEqual(
                by_id[f"l{layer}:s1:ffn"].start_us,
                by_id[f"l{layer}:s0:ffn"].end_us,
            )
            if layer:
                self.assertGreaterEqual(
                    by_id[f"l{layer}:s0:qkv"].start_us,
                    by_id[f"l{layer-1}:s1:qkv"].end_us,
                )
                self.assertGreaterEqual(
                    by_id[f"l{layer}:s0:attention"].start_us,
                    by_id[f"l{layer-1}:s1:attention_done"].end_us,
                )
                self.assertGreaterEqual(
                    by_id[f"l{layer}:s0:ffn"].start_us,
                    by_id[f"l{layer-1}:s1:ffn"].end_us,
                )
        for resource in ("gpu", "pim:0"):
            occupied = sorted(
                (event.start_us, event.end_us)
                for event in events if resource in event.resources
            )
            for previous, current in zip(occupied, occupied[1:]):
                self.assertLessEqual(previous[1], current[0])

    def test_diagonal_pair_has_shared_phase_and_waits_for_both_sides(self) -> None:
        events = schedule_pipeline(
            order_two_subbatch_stages(self._two_stream_tasks(2))
        ).events
        by_id = {event.task_id: event for event in events}

        # The requested pair: B0 Attention in layer 1 runs beside B1's
        # layer-0 F and its next-layer QKV preparation.
        third_start = by_id["phase:3:start"].end_us
        third_done = by_id["phase:3:done"].start_us
        a0_l1 = by_id["l1:s0:attention"]
        f1_l0 = by_id["l0:s1:ffn"]
        qkv1_l1 = by_id["l1:s1:qkv"]
        self.assertGreaterEqual(a0_l1.start_us, third_start)
        self.assertGreaterEqual(f1_l0.start_us, third_start)
        self.assertLess(a0_l1.start_us, f1_l0.end_us)
        self.assertLess(f1_l0.start_us, a0_l1.end_us)
        self.assertGreaterEqual(third_done, max(a0_l1.end_us, qkv1_l1.end_us))
        self.assertAlmostEqual(third_done - third_start, 5.0)

        # The next pair cannot begin until both branches above complete.
        fourth_start = by_id["phase:4:start"].start_us
        self.assertGreaterEqual(fourth_start, third_done)
        a1_l1 = by_id["l1:s1:attention"]
        f0_l1 = by_id["l1:s0:ffn"]
        self.assertGreaterEqual(a1_l1.start_us, fourth_start)
        self.assertGreaterEqual(f0_l1.start_us, fourth_start)
        self.assertLess(a1_l1.start_us, f0_l1.end_us)
        self.assertLess(f0_l1.start_us, a1_l1.end_us)

    def test_unconstrained_wavefront_can_lead_multiple_layers(self) -> None:
        tasks = self._two_stream_tasks(4, slow_second_attention=True)
        free = {event.task_id: event for event in schedule_pipeline(tasks).events}
        ordered = {
            event.task_id: event
            for event in schedule_pipeline(order_two_subbatch_stages(tasks)).events
        }
        self.assertLess(
            free["l2:s0:attention"].start_us,
            free["l0:s1:attention"].end_us,
        )
        self.assertGreaterEqual(
            ordered["l1:s0:attention"].start_us,
            ordered["l0:s1:attention_done"].end_us,
        )

    def test_device_segment_keeps_pim_oproj_with_attention(self) -> None:
        tasks = []
        for subbatch in (0, 1):
            prefix = f"l0:s{subbatch}"
            tasks.extend((
                PipelineTask(f"{prefix}:qkv", 1, ("gpu",), (),
                             "gpu_qkv", 0, subbatch),
                PipelineTask(f"{prefix}:pim_attention", 8, ("pim:0",),
                             (f"{prefix}:qkv",), "pim_attention", 0, subbatch),
                PipelineTask(f"{prefix}:pim_o_proj", 2, ("pim:0",),
                             (f"{prefix}:pim_attention",), "pim_o_proj", 0, subbatch),
                PipelineTask(f"{prefix}:ffn", 10, ("gpu",),
                             (f"{prefix}:pim_o_proj",), "gpu_ffn", 0, subbatch),
            ))
        events = schedule_pipeline(order_two_subbatch_device_segments(tasks)).events
        by_id = {event.task_id: event for event in events}
        self.assertEqual(
            by_id["l0:s0:pim_o_proj"].start_us,
            by_id["l0:s0:pim_attention"].end_us,
        )
        self.assertLess(
            by_id["l0:s0:pim_o_proj"].start_us,
            by_id["l0:s1:qkv"].end_us + 10,
        )
        self.assertGreaterEqual(
            by_id["visit:2:start"].start_us,
            by_id["visit:1:done"].end_us,
        )

    def test_ready_queue_keeps_gap_filling_schedule(self) -> None:
        schedule = schedule_pipeline(
            (
                PipelineTask("a", 4.0, ("r1",), priority=0),
                PipelineTask("b", 10.0, ("r2",), priority=0),
                PipelineTask("c", 2.0, ("r1", "r2"), ("a",)),
                PipelineTask("d", 3.0, ("r1",), ("a",), priority=0),
            )
        )
        by_id = {event.task_id: event for event in schedule.events}
        self.assertEqual((by_id["a"].start_us, by_id["a"].end_us), (0.0, 4.0))
        self.assertEqual((by_id["b"].start_us, by_id["b"].end_us), (0.0, 10.0))
        self.assertEqual((by_id["d"].start_us, by_id["d"].end_us), (4.0, 7.0))
        self.assertEqual((by_id["c"].start_us, by_id["c"].end_us), (10.0, 12.0))
        self.assertEqual(schedule.makespan_us, 12.0)

    def test_cycle_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cycle"):
            schedule_pipeline(
                (
                    PipelineTask("a", 1.0, ("r",), ("b",)),
                    PipelineTask("b", 1.0, ("r",), ("a",)),
                )
            )


if __name__ == "__main__":
    unittest.main()
