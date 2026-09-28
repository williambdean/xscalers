"""VariableTransformer — the ColumnTransformer analog, one unit per variable.

Mirroring ``sklearn.compose``: a map from variable (or DataTree node) to
transformer, applied per unit. Mixed-grain inputs need no special casing —
each variable's fitted state keeps that variable's own remaining dims.
Unmapped variables pass through or drop, mirroring
``ColumnTransformer(remainder="passthrough")`` / ``remainder="drop"``.

Fitted state for this composite is ``dict[str, xr.Dataset]`` (one Dataset per
variable), which ``state_tree`` nests into the canonical DataTree artifact.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import xarray as xr

from .container import flatten, reassemble
from .core import _stamp_state, register

__all__ = ["VariableTransformer"]


@register
class VariableTransformer:
    """Map variables/data-tree nodes onto their own transformers.

    ``remainder`` is either ``"passthrough"`` (unknown keys are left untouched,
    the sklearn ``remainder`` semantics) or another transformer applied to
    every variable without an explicit entry. ``default`` is a deprecated
    alias kept so older specs and call sites load unchanged.
    """

    def __init__(self, mapping: Mapping[str, Any] | None = None,
                 remainder: Any = "passthrough", default: Any = None):
        self.mapping = dict(mapping or {})
        self.remainder = default if default is not None else remainder

    # -- key matching ------------------------------------------------------

    def _scaler_for(self, key: str) -> Any:
        if key in self.mapping:
            return self.mapping[key]
        for mapped, scaler in self.mapping.items():
            if key.endswith(f"/{mapped}"):
                return scaler
        return self.remainder

    @staticmethod
    def _apply(scaler: Any, op: str, da: Any, state: Any) -> Any:
        method = getattr(scaler, op)
        if op == "transform" or op == "inverse_transform":
            return method(da, state)
        return method(da)

    # -- transformer API ---------------------------------------------------

    def fit(self, X: Any) -> dict[str, Any]:
        items = flatten(X)
        self._warn_on_missing_keys(items)
        states: dict[str, Any] = {}
        for key, da in items.items():
            scaler = self._scaler_for(key)
            if scaler == "passthrough" or scaler == "drop":
                continue
            # stamped with the *scaler's* recipe: the leaf states are the
            # provenance units, not the dispatch wrapper around them
            states[key] = _stamp_state(scaler.compute(da), scaler)
        return states

    def _warn_on_missing_keys(self, items: Mapping[str, Any]) -> None:
        """Explicit mapping entries naming absent variables are config bugs.

        Coordinates (e.g. `date`) are never dispatch keys — a recipe naming
        one would otherwise pass through silently under passthrough defaults.
        Suffix matches (split pieces like 'media/digital' ↔ 'digital') are
        legitimate, so they don't warn.
        """
        item_keys = list(items)

        def _matched(key: str) -> bool:
            return any(k == key or k.endswith(f"/{key}") for k in item_keys)

        missing = [k for k in self.mapping if not _matched(k)]
        if not missing:
            return
        for key in missing:
            import difflib

            suggestion = difflib.get_close_matches(key, item_keys, n=1)
            hint = f"; did you mean {suggestion[0]!r}?" if suggestion else ""
            import warnings

            warnings.warn(
                f"VariableTransformer mapping key {key!r} is missing from the input's "
                f"data variables{hint} (coordinates or DataTree children "
                "cannot be scaled directly); the variable will pass through unscaled.",
                UserWarning,
                stacklevel=3,
            )

    def transform(self, X: Any, state: Mapping[str, Any] | xr.Dataset) -> Any:
        out: dict[str, Any] = {}
        for key, da in flatten(X).items():
            scaler = self._scaler_for(key)
            if scaler == "drop":
                continue
            st = state.get(key)
            if st is None:
                out[key] = da
            else:
                out[key] = scaler.transform(da, st)
        return reassemble(X, out)

    def inverse_transform(self, X: Any, state: Mapping[str, Any] | xr.Dataset) -> Any:
        out: dict[str, Any] = {}
        for key, da in flatten(X).items():
            scaler = self._scaler_for(key)
            st = state.get(key)
            if st is None or scaler == "drop":
                out[key] = da
            else:
                out[key] = scaler.inverse_transform(da, st)
        return reassemble(X, out)

    def affine_view(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Per-variable affine views from this step's states."""
        views: dict[str, Any] = {}
        for key, ds in state.items():
            views[key] = self._scaler_for(key).affine_view(ds)
        return views

    def to_spec_kwargs(self) -> dict[str, Any]:
        from .core import to_spec

        # "passthrough" (and any other bare string) is a documented value, not
        # a transformer -- leave it alone instead of trying to spec it
        mapping = {
            key: to_spec(scaler) if not isinstance(scaler, str) else scaler
            for key, scaler in self.mapping.items()
        }
        remainder = self.remainder
        if not isinstance(remainder, str):
            remainder = to_spec(remainder)
        return {"mapping": mapping, "remainder": remainder}
