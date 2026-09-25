"""Plain-dict spec serialization: recipes persist as config; state rides as netCDF."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["from_spec", "spec_of", "to_spec"]


def to_spec(transformer: Any) -> dict[str, Any]:
    """Serialize a transformer recipe to a plain dict (JSON-serializable)."""
    kwargs = {
        k: (_spec_value(v)) for k, v in transformer.to_spec_kwargs().items()
    }
    return {"kind": type(transformer).__name__, "kwargs": kwargs}


def _spec_value(value: Any) -> Any:
    import xarray as xr

    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, xr.DataTree):
        return value
    return value


def spec_of(transformer: Any) -> dict[str, Any]:
    return to_spec(transformer)


def from_spec(spec: Mapping[str, Any]) -> Any:
    """Rebuild a transformer from a plain-dict spec."""
    from .core import REGISTRY

    kind = spec["kind"]
    if kind == "Pipeline":
        from .pipeline import Pipeline

        return Pipeline.from_spec(spec)
    if kind not in REGISTRY:
        raise ValueError(f"Unknown transformer kind {kind!r}. Known: {sorted(REGISTRY)}")
    cls = REGISTRY[kind]
    return cls(**_from_spec_kwargs(spec["kwargs"]))


def _from_spec_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    return {k: _from_spec_value(v) for k, v in kwargs.items()}


def _from_spec_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if {"kind", "kwargs"} <= set(value):
            return from_spec(value)
        return {k: _from_spec_value(v) for k, v in value.items()}
    if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
        return tuple(value)
    return value


class SpecMixin:
    """Scalers write their constructor kwargs here; spec pairs with from_spec."""

    def to_spec_kwargs(self) -> dict[str, Any]:
        raise NotImplementedError
