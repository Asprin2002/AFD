"""Logical operator mappings shared by all simulator experiments.

The presets deliberately describe *where* an operator runs, independently of
how a particular GPU or PIM implements it.  This keeps CHIME and DOPS-inspired
experiments on one workload/capacity/event engine without claiming that their
published hardware is identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ExecutionTarget(str, Enum):
    GPU = "gpu"
    PIM = "pim"
    # Attention follows each request's KV-segment placement.  This is the
    # mechanism used by CHIME (all-PIM KV) and ElasticAFD (split KV).
    KV = "kv"


@dataclass(frozen=True)
class OperatorMapping:
    name: str
    label: str
    qkv: ExecutionTarget
    attention: ExecutionTarget
    o_proj: ExecutionTarget
    ffn: ExecutionTarget
    provenance: str

    def target(self, operator: str) -> ExecutionTarget:
        try:
            return {
                "qkv": self.qkv,
                "attention": self.attention,
                "o_proj": self.o_proj,
                "ffn": self.ffn,
            }[operator]
        except KeyError as error:
            raise ValueError(f"unknown logical operator: {operator}") from error

    @property
    def pim_dense_operators(self) -> tuple[str, ...]:
        return tuple(
            operator
            for operator in ("qkv", "o_proj", "ffn")
            if self.target(operator) is ExecutionTarget.PIM
        )


# DOPS's released implementation uses a finer graph (Q/K/V, QK, softmax, SV,
# etc.).  These presets retain its operator-boundary idea at the four-stage
# granularity supported by this simulator.  They are therefore explicitly
# called DOPS-inspired, not numerical reproductions of the paper.
OPERATOR_MAPPINGS: dict[str, OperatorMapping] = {
    "chime": OperatorMapping(
        name="chime",
        label="CHIME fixed Attention-FC boundary",
        qkv=ExecutionTarget.GPU,
        attention=ExecutionTarget.KV,
        o_proj=ExecutionTarget.GPU,
        ffn=ExecutionTarget.GPU,
        provenance="CHIME",
    ),
    "gpu_only": OperatorMapping(
        name="gpu_only",
        label="GPU-only",
        qkv=ExecutionTarget.GPU,
        attention=ExecutionTarget.GPU,
        o_proj=ExecutionTarget.GPU,
        ffn=ExecutionTarget.GPU,
        provenance="control",
    ),
    "attn_pim_control": OperatorMapping(
        name="attn_pim_control",
        label="generic-event attention-only PIM control",
        qkv=ExecutionTarget.GPU,
        attention=ExecutionTarget.PIM,
        o_proj=ExecutionTarget.GPU,
        ffn=ExecutionTarget.GPU,
        provenance="control: same placement as CHIME, generic operator event graph",
    ),
    # Factorial F-side placement family used to isolate dense-operator
    # migration from KV/Attention placement.  Attention and all KV remain on
    # PIM; QKV, O_proj and FFN are independently opened, giving all 2^3
    # combinations on one dependency/resource event graph.  PIM dense costs
    # are conditional roofline estimates, not CHIME-supported kernels.
    "f_qkv_pim": OperatorMapping(
        name="f_qkv_pim",
        label="F placement: QKV on PIM",
        qkv=ExecutionTarget.PIM,
        attention=ExecutionTarget.PIM,
        o_proj=ExecutionTarget.GPU,
        ffn=ExecutionTarget.GPU,
        provenance="conditional PIM-dense roofline factorial",
    ),
    "f_oproj_pim": OperatorMapping(
        name="f_oproj_pim",
        label="F placement: O_proj on PIM",
        qkv=ExecutionTarget.GPU,
        attention=ExecutionTarget.PIM,
        o_proj=ExecutionTarget.PIM,
        ffn=ExecutionTarget.GPU,
        provenance="conditional PIM-dense roofline factorial",
    ),
    "f_ffn_pim": OperatorMapping(
        name="f_ffn_pim",
        label="F placement: FFN on PIM",
        qkv=ExecutionTarget.GPU,
        attention=ExecutionTarget.PIM,
        o_proj=ExecutionTarget.GPU,
        ffn=ExecutionTarget.PIM,
        provenance="conditional PIM-dense roofline factorial",
    ),
    "f_qkv_oproj_pim": OperatorMapping(
        name="f_qkv_oproj_pim",
        label="F placement: QKV and O_proj on PIM",
        qkv=ExecutionTarget.PIM,
        attention=ExecutionTarget.PIM,
        o_proj=ExecutionTarget.PIM,
        ffn=ExecutionTarget.GPU,
        provenance="conditional PIM-dense roofline factorial",
    ),
    "f_qkv_ffn_pim": OperatorMapping(
        name="f_qkv_ffn_pim",
        label="F placement: QKV and FFN on PIM",
        qkv=ExecutionTarget.PIM,
        attention=ExecutionTarget.PIM,
        o_proj=ExecutionTarget.GPU,
        ffn=ExecutionTarget.PIM,
        provenance="conditional PIM-dense roofline factorial",
    ),
    "f_oproj_ffn_pim": OperatorMapping(
        name="f_oproj_ffn_pim",
        label="F placement: O_proj and FFN on PIM",
        qkv=ExecutionTarget.GPU,
        attention=ExecutionTarget.PIM,
        o_proj=ExecutionTarget.PIM,
        ffn=ExecutionTarget.PIM,
        provenance="conditional PIM-dense roofline factorial",
    ),
    "f_all_pim": OperatorMapping(
        name="f_all_pim",
        label="F placement: QKV, O_proj and FFN on PIM",
        qkv=ExecutionTarget.PIM,
        attention=ExecutionTarget.PIM,
        o_proj=ExecutionTarget.PIM,
        ffn=ExecutionTarget.PIM,
        provenance="conditional PIM-dense roofline factorial",
    ),
    "dops_attn": OperatorMapping(
        name="dops_attn",
        label="DOPS-inspired attention-only PIM",
        qkv=ExecutionTarget.GPU,
        attention=ExecutionTarget.PIM,
        o_proj=ExecutionTarget.GPU,
        ffn=ExecutionTarget.GPU,
        provenance="DOPS PD+Attn (coarse abstraction)",
    ),
    "dops_af": OperatorMapping(
        name="dops_af",
        label="DOPS-inspired attention-family PIM",
        qkv=ExecutionTarget.PIM,
        attention=ExecutionTarget.PIM,
        o_proj=ExecutionTarget.PIM,
        ffn=ExecutionTarget.GPU,
        provenance="DOPS AF (coarse abstraction)",
    ),
    "dops_all_pim": OperatorMapping(
        name="dops_all_pim",
        label="DOPS-inspired decode-on-PIM",
        qkv=ExecutionTarget.PIM,
        attention=ExecutionTarget.PIM,
        o_proj=ExecutionTarget.PIM,
        ffn=ExecutionTarget.PIM,
        provenance="DOPS PD decode phase (coarse abstraction)",
    ),
    "dops_ffn": OperatorMapping(
        name="dops_ffn",
        label="DOPS-inspired FFN-side PIM",
        qkv=ExecutionTarget.GPU,
        attention=ExecutionTarget.GPU,
        o_proj=ExecutionTarget.PIM,
        ffn=ExecutionTarget.PIM,
        provenance="DOPS PD+FFN (coarse abstraction)",
    ),
}


def get_operator_mapping(name: str) -> OperatorMapping:
    normalized = name.lower().replace("-", "_").replace("+", "_")
    aliases = {
        "gpu": "gpu_only",
        "pd": "dops_all_pim",
        "af": "dops_af",
        "pd_attn": "dops_attn",
        "pd_ffn": "dops_ffn",
    }
    normalized = aliases.get(normalized, normalized)
    try:
        return OPERATOR_MAPPINGS[normalized]
    except KeyError as error:
        choices = ", ".join(sorted(OPERATOR_MAPPINGS))
        raise ValueError(f"operator_mapping must be one of: {choices}") from error
