"""Decode-ready CHIME baseline, kept separate from candidate placement models.

CHIME §VI-C selects two sub-batches in N-request rounds using predicted GPU
and PIM latency. For GPT-175B MHA, N=1. The paper does not specify a complete
decode-only request ordering or whether unselected requests wait for another
iteration. This closed-batch specialization schedules every supplied request
once per iteration and pairs longest remaining contexts first. It must not be
described as the official CHIME artifact. Predicted times come from this
simulator's shared cost model, not CHIME's runtime-trained RFR predictor.
"""

from __future__ import annotations

from typing import Sequence

from .cost import ChimeCostModel
from .specs import ModelSpec, SystemSpec
from .state import RequestView


class ChimeDecodeBaseline(ChimeCostModel):
    """CHIME placement, paired device visits, independent decode grouping."""

    def __init__(self, model: ModelSpec, system: SystemSpec) -> None:
        if model.data_parallel != 1:
            raise ValueError("decode-ready CHIME grouping currently requires DP=1")
        if model.num_attention_heads != model.num_kv_heads:
            raise ValueError("N=1 grouping is specified here for MHA only")
        super().__init__(
            model, system, fidelity="tile", pipeline_model="rankset_event",
            subbatch_scheduler="fixed", subbatch_progression="device_contiguous",
            operator_mapping="chime",
        )

    def _paired_score(self, left: Sequence[RequestView],
                      right: Sequence[RequestView]) -> tuple[float, float]:
        """Paper Eq. (2)-(5) decode-only cross-device alignment objective."""
        left_cost = self.subbatch_layer_cost(left)
        right_cost = self.subbatch_layer_cost(right)
        pim_left = left_cost.pim.total_us
        pim_right = right_cost.pim.total_us
        gpu_left = left_cost.dense.total_us
        gpu_right = right_cost.dense.total_us
        return (
            max(pim_left, gpu_right) + max(pim_right, gpu_left),
            abs(pim_left - gpu_right) + abs(pim_right - gpu_left),
        )

    def partition_views(
        self, views: Sequence[RequestView]
    ) -> tuple[tuple[RequestView, ...], ...]:
        if not views:
            raise ValueError("cannot form sub-batches from no requests")
        if any(view.gpu_tokens or not view.pim_tokens for view in views):
            raise ValueError("CHIME baseline requires all historical KV on PIM")
        if len(views) == 1:
            return (tuple(views),)
        # §VI-C adds N decoding requests to *each* sub-batch before updating
        # predicted times. GPT-175B is MHA, for which the paper uses N=1.
        remaining = sorted(views, key=lambda view: (-view.pim_tokens,
                                                     view.request_id))
        bins: list[list[RequestView]] = [[], []]
        for start in range(0, len(remaining), 2):
            pair = remaining[start:start + 2]
            if len(pair) == 1:
                choices = ((0, pair[0]), (1, pair[0]))
                # One unpaired request is allowed only in the final round.
                chosen = min(choices, key=lambda item: (
                    self._paired_score(
                        [*bins[0], item[1]] if item[0] == 0 else bins[0],
                        [*bins[1], item[1]] if item[0] == 1 else bins[1],
                    ), item[0]
                ))
                bins[chosen[0]].append(chosen[1])
                break
            first, second = pair
            forward = ([*bins[0], first], [*bins[1], second])
            reverse = ([*bins[0], second], [*bins[1], first])
            if self._paired_score(*reverse) < self._paired_score(*forward):
                bins[0].append(second)
                bins[1].append(first)
            else:
                bins[0].append(first)
                bins[1].append(second)
        return tuple(tuple(part) for part in bins if part)
