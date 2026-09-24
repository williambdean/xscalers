"""Lift fitted xarray states into pytensor.xtensor graphs — and transform symbolically.

The flagship flow: ``fit`` is eager xarray (stats are data); ``transform`` /
``inverse_transform`` are elementwise arithmetic that work identically on
DataArrays and XTensorVariables. ``lift_state`` bridges the two — via
``ptx.as_xtensor``, which takes DataArrays natively (dims preserved, coords
stripped) and is idempotent for entries that are already symbolic, which is
what makes prior-over-scale ("learnable") states supported for free.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import xarray as xr

from .core import is_symbolic

__all__ = ["lift_state", "lift_state_if_symbolic"]


def lift_state(state: Mapping) -> dict:
    """Coerce every state entry into an XTensorVariable/constant."""
    out: dict[str, Any] = {}
    for key, value in state.items():
        if is_symbolic(value):
            out[key] = value
        elif isinstance(value, xr.DataArray):
            out[key] = _as_symbol(value)
        elif isinstance(value, xr.Dataset):
            for entry, da in value.data_vars.items():
                out[str(entry)] = _as_symbol(da)
        else:
            import pytensor.xtensor as ptx

            out[key] = ptx.as_xtensor(np.asarray(value.item() if hasattr(value, "item") else value))
    return out


def _as_symbol(value: xr.DataArray) -> Any:
    import pytensor.xtensor as ptx

    if value.ndim == 0:
        return ptx.as_xtensor(np.asarray(value.item()))
    return ptx.as_xtensor(value)


def lift_state_if_symbolic(X: Any, state: Mapping) -> Mapping:
    """Lift only when the input is symbolic; xarray passes through untouched."""
    if not is_symbolic(X):
        return state
    return lift_state(state)
