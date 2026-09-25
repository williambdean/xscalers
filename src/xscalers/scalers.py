"""The v1 basics: FixedScaler (constant, no data needed), ZScore, MeanScaler.

All are affine — y = (x - center) / scale — and expose state keys `center`
and `scale` (z-score's mean/std map onto this pair, matching PR-2831).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from .core import BaseAffineTransformer, BaseTransformer, as_state, register

__all__ = ["FixedScaler", "Func", "MeanScaler", "ZScore"]

# named elementwise ops; inverse pairs keep the whole chain invertible
_FUNC_INVERSES: dict[str, str] = {
    "log": "exp",
    "log1p": "expm1",
    "sqrt": "square",
    "exp": "log",
    "expm1": "log1p",
    "square": "sqrt",
}


@register
class Func(BaseTransformer):
    """Named elementwise transform per variable — non-affine on purpose.

    The second half of the "ratio, then log" chain: after a ``Derive`` step
    produces a derived metric, apply a named unary function (e.g. ``log``) to
    selected variables. Unmapped variables pass through unchanged.

    Because it is not affine, ``Pipeline.affine()`` refuses pipelines that
    contain it (by design): if the *target* is log-transformed, real-terms
    contributions must go through ``inverse_transform`` — the batched path —
    rather than the ``β·scale`` shortcut.
    """

    def __init__(self, mapping: dict[str, str]):
        unknown = {op for op in mapping.values() if op not in _FUNC_INVERSES}
        if unknown:
            raise ValueError(
                f"Unknown functions {sorted(unknown)}; known: {sorted(_FUNC_INVERSES)}"
            )
        self.mapping = dict(mapping)

    def to_spec_kwargs(self) -> dict[str, Any]:
        return {"mapping": dict(self.mapping)}

    def compute(self, X: Any) -> xr.Dataset:
        name = getattr(X, "name", None)
        scaled_from = str(name) if name is not None else ""
        # the mapping lives on the *recipe*; each per-variable state carries
        # only the scalar op decision for that variable (netCDF-safe)
        return as_state(
            "function",
            None,
            scaled_from,
            attrs={
                # None → "" for netCDF-safety (scalar string attrs only)
                "op": self.mapping.get(scaled_from, ""),
            },
        )

    def _transform(self, X: Any, state: Any) -> Any:
        op = state.attrs.get("op", "")
        if not op:
            return X
        return getattr(np, op)(X)

    def _inverse_transform(self, X: Any, state: Any) -> Any:
        op = state.attrs.get("op", "")
        if not op:
            return X
        return getattr(np, _FUNC_INVERSES[op])(X)


@register
class FixedScaler(BaseAffineTransformer):
    """Constant affine pair — scale by a fixed amount (e.g. scale by 100).

    No data-derived stats: ``compute`` returns constants, so the same object
    works as pymc-marketing's ``FixedScaling``.
    """

    def __init__(self, scale: float = 100.0, center: float = 0.0):
        super().__init__(dims=None)
        self.scale = scale
        self.center = center

    def to_spec_kwargs(self) -> dict[str, Any]:
        return {"scale": self.scale, "center": self.center}

    def compute(self, X: Any) -> xr.Dataset:
        scaled_from = getattr(X, "name", None)
        return as_state(
            "fixed", None, scaled_from, center=self.center, scale=self.scale
        )


class _DataDerivedScaler(BaseAffineTransformer):
    """Shared plumbing: compute affine stats from data over the reduce dims."""

    kind: str

    def to_spec_kwargs(self) -> dict[str, Any]:
        return {"dims": self.dims}

    def _stats(self, da: Any) -> tuple[Any, Any]:
        raise NotImplementedError

    def compute(self, X: Any) -> xr.Dataset:
        scaled_from = getattr(X, "name", None)
        center, scale = self._stats(X)
        return as_state(self.kind, self.dims, scaled_from, center=center, scale=scale)


@register
class ZScore(_DataDerivedScaler):
    """Standardize along the reduce dims: center = mean, scale = std (ddof=0).

    Matches sklearn's ``StandardScaler`` when the kept dim is the feature axis.
    NaNs are skipped in the statistics.
    """

    kind = "zscore"

    def _stats(self, da: Any):
        dims = self._reduce_dims(da)
        center = da.mean(dim=dims, skipna=True)
        scale = da.std(dim=dims, ddof=0, skipna=True)
        return center, scale


@register
class MeanScaler(_DataDerivedScaler):
    """Divide by the mean along the reduce dims: center = 0, scale = mean."""

    kind = "mean"

    def _stats(self, da: Any):
        dims = self._reduce_dims(da)
        center = xr.DataArray(0.0)
        scale = da.mean(dim=dims, skipna=True)
        return center, scale
