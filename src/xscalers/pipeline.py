"""sklearn-shaped pipelines: named steps, chained states, reversed inverse, affine view.

``Pipeline.fit`` collects each step's state as it transforms its way down;
``inverse_transform`` runs the chain backwards against the same states.
``affine`` composes each step's affine pair per variable — the contract
pymc-marketing consumes — and *refuses* pipelines that include non-affine
steps rather than silently returning a wrong pair. The whole fitted state is
one netCDF artifact via ``state_tree``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import xarray as xr

from .container import state_tree
from .core import from_spec, to_spec

__all__ = ["Pipeline"]


class Pipeline:
    def __init__(self, steps: list[tuple[str, Any]]):
        self.steps = [tuple(step) for step in steps]

    def named_steps(self) -> dict[str, Any]:
        return dict(self.steps)

    # -- sklearn-shaped API ------------------------------------------------

    def fit(self, X: Any) -> dict[str, Any]:
        states: dict[str, Any] = {}
        current = X
        for name, step in self.steps:
            states[name] = step.fit(current)
            current = step.transform(current, states[name])
        return states

    def transform(self, X: Any, state: Mapping[str, Any]) -> Any:
        current = X
        for name, step in self.steps:
            current = step.transform(current, state[name])
        return current

    def inverse_transform(self, X: Any, state: Mapping[str, Any]) -> Any:
        current = X
        for name, step in reversed(self.steps):
            current = step.inverse_transform(current, state[name])
        return current

    def fit_transform(self, X: Any) -> Any:
        state = self.fit(X)
        return self.transform(X, state)

    # -- affine composition -------------------------------------------------

    def affine(self, state: Mapping[str, Any]) -> dict[str, xr.Dataset]:
        """Composed (center, scale) per variable across all steps.

        Raises TypeError if any step is not affine — a shape-changing step
        would silently produce a wrong mapping otherwise.
        """
        composed: dict[str, xr.Dataset] = {}
        for (name, step), step_state in zip(self.steps, state.values(), strict=True):
            view = getattr(step, "affine_view", None)
            if view is None:
                raise TypeError(f"Step {name!r} ({type(step).__name__}) is not affine")
            for key, pair in view(step_state).items():
                center, scale = pair["center"], pair["scale"]
                if key in composed:
                    prev = composed[key]
                    # y2 = (y1 - c2) / s2 where y1 = (x - c1) / s1
                    composed[key] = xr.Dataset(
                        {"center": prev["center"] + prev["scale"] * center, "scale": prev["scale"] * scale}
                    )
                else:
                    composed[key] = xr.Dataset({"center": center, "scale": scale})
        return composed

    # -- state artifact ------------------------------------------------------

    def to_state_tree(self, state: Mapping[str, Any]) -> xr.DataTree:
        return state_tree(state)

    # -- spec serialization ---------------------------------------------------

    def to_spec(self) -> dict[str, Any]:
        steps = [[name, to_spec(step)] for name, step in self.steps]
        return {"kind": "Pipeline", "kwargs": {"steps": steps}}

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> Pipeline:
        steps = [
            (name, from_spec(step))
            for name, step in spec["kwargs"]["steps"]
        ]
        return cls(steps=steps)
