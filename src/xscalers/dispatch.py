"""PerVariable dispatch: a map from variable (or DataTree node) to transformer.

Mixed-grain inputs need no special casing — each variable's fitted state keeps
that variable's own remaining dims. Unmapped variables pass through, mirroring
sklearn's ``ColumnTransformer(remainder="passthrough")`` semantics on ``data_vars``.

Fitted state for this composite is ``dict[str, xr.Dataset]`` (one Dataset per
variable), which ``state_tree`` nests into the canonical DataTree artifact.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .container import flatten, reassemble
from .core import register

__all__ = ["PerVariable"]


@register
class PerVariable:
    """Map variables/data-tree nodes onto their own transformers.

    ``default`` is either ``"passthrough"`` (unknown keys are left untouched,
    the sklearn ``remainder`` semantics) or another transformer applied to
    every variable without an explicit entry.
    """

    def __init__(self, mapping: Mapping[str, Any] | None = None, default: Any = "passthrough"):
        self.mapping = dict(mapping or {})
        self.default = default

    # -- key matching ------------------------------------------------------

    def _scaler_for(self, key: str) -> Any:
        if key in self.mapping:
            return self.mapping[key]
        for mapped, scaler in self.mapping.items():
            if key.endswith(f"/{mapped}"):
                return scaler
        return self.default

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
            states[key] = scaler.compute(da)
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
                f"PerVariable mapping key {key!r} is missing from the input's "
                f"data variables{hint} (coordinates or DataTree children "
                "cannot be scaled directly); the variable will pass through unscaled.",
                UserWarning,
                stacklevel=3,
            )

    def transform(self, X: Any, state: Mapping[str, Any]) -> Any:
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

    def inverse_transform(self, X: Any, state: Mapping[str, Any]) -> Any:
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

        mapping = {key: to_spec(scaler) for key, scaler in self.mapping.items()}
        default = self.default
        if not isinstance(default, str):
            default = to_spec(default)
        return {"mapping": mapping, "default": default}
