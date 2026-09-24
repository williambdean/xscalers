"""The v1 basics: FixedScaler (constant, no data needed), ZScore, MeanScaler.

All are affine — y = (x - center) / scale — and expose state keys `center`
and `scale` (z-score's mean/std map onto this pair, matching PR-2831).
"""

from __future__ import annotations

from typing import Any

import xarray as xr

from .core import BaseAffineTransformer, as_state, register

__all__ = ["FixedScaler", "MeanScaler", "ZScore"]


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
