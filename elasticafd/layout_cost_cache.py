"""Bounded, exact cost reuse for one layout search.

Only immutable cost results live here. Physical worlds, forecast lifetimes,
capacity checks and link calendars are never cached. Context-local storage
keeps candidate deep copies cheap and concurrent searches independent.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any


class CostCache:
    def __init__(self, *, stage_limit: int = 8192, kernel_limit: int = 4096):
        self.limits = {"stage": stage_limit, "linear": kernel_limit,
                       "shards": kernel_limit, "attention": kernel_limit}
        if min(self.limits.values()) <= 0:
            raise ValueError("cost cache limits must be positive")
        self.values: dict[str, dict[tuple, Any]] = {name: {} for name in self.limits}
        self.counts: dict[str, int] = defaultdict(int)
        self.configs: dict[tuple, int] = {}
        self.config_ids: dict[int, tuple] = {}
        self.layer_layouts: dict[int, tuple[tuple, ...]] = {}

    def config_token(self, config) -> int:
        known = self.config_ids.get(id(config))
        if (known is not None and known[0] is config.model
                and known[1] is config.system and known[2] is config.simulation):
            return known[3]
        # Workload rows and hidden output lengths are not cost inputs. Keep
        # immutable settings, rather than holding every copied workload alive.
        settings = (config.model, config.system, config.simulation)
        token = self.configs.get(settings)
        if token is None:
            token = len(self.configs)
            self.configs[settings] = token
        self.config_ids[id(config)] = (*settings, token)
        return token

    def get(self, kind: str, key: tuple):
        value = self.values[kind].get(key)
        self.counts[f"{kind}_hits" if value is not None else f"{kind}_misses"] += 1
        return value

    def put(self, kind: str, key: tuple, value):
        bucket = self.values[kind]
        if len(bucket) >= self.limits[kind]:
            bucket.clear()
            self.counts[f"{kind}_resets"] += 1
        bucket[key] = value
        self.counts[f"{kind}_peak_entries"] = max(
            self.counts[f"{kind}_peak_entries"], len(bucket))


_CURRENT: ContextVar[CostCache | None] = ContextVar("layout_cost_cache", default=None)


def current_cost_cache() -> CostCache | None:
    return _CURRENT.get()


@contextmanager
def reuse_costs(statistics: dict | None = None, **limits):
    cache = CostCache(**limits)
    token = _CURRENT.set(cache)
    try:
        yield cache
    finally:
        _CURRENT.reset(token)
        if statistics is not None:
            statistics.update(cache.counts)
            statistics["configuration_count"] = len(cache.configs)
