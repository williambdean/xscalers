"""sklearn-shaped pipelines: named steps, chained states, reversed inverse, affine view.

``Pipeline.fit`` collects each step's state as it transforms its way down;
``inverse_transform`` runs the chain backwards against the same states.
``affine`` composes each step's affine pair per variable — the contract
pymc-marketing consumes — and *refuses* pipelines that include non-affine
steps rather than silently returning a wrong pair. The whole fitted state is
one netCDF artifact via ``state_tree``.

Ergonomics: ``pipe["name"]`` yields a *bound* step (transformer + its fitted
state slice) that can transform new data on its own; ``"passthrough"`` is a
valid step value; ``make_pipeline`` derives names from transformer classes.
The whole fitted state is one netCDF artifact via ``state_tree``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import xarray as xr

from .container import state_tree
from .core import from_spec, to_spec

__all__ = ["BoundStep", "Pipeline", "make_pipeline"]


class BoundStep:
    """A fitted step: transformer bound to its slice of the pipeline state.

    Returned by ``Pipeline.__getitem__`` — the "access transformations after
    the fact" surface. ``step.transform(X)`` uses the fitted state directly,
    so a single fitted step can be applied to fresh data or reused elsewhere.
    """

    __slots__ = ("name", "state", "transformer")

    def __init__(self, name: str, transformer: Any, state: Any):
        self.name = name
        self.transformer = transformer
        self.state = state

    def transform(self, X: Any) -> Any:
        return self.transformer.transform(X, self.state)

    def inverse_transform(self, X: Any) -> Any:
        return self.transformer.inverse_transform(X, self.state)


class _Passthrough:
    """The ``"passthrough"`` step value: fitted state is empty, transform is identity."""

    def fit(self, X: Any) -> xr.Dataset:
        return xr.Dataset()

    def transform(self, X: Any, state: Any) -> Any:
        return X

    def inverse_transform(self, X: Any, state: Any) -> Any:
        return X

    def affine_view(self, state: Any) -> dict[str, Any]:
        return {}


class Pipeline:
    def __init__(self, steps: list[tuple[str, Any]]):
        self.steps = [
            (name, _Passthrough() if step == "passthrough" else step)
            for name, step in steps
        ]
        self.states_: dict[str, Any] | None = None

    def named_steps(self) -> dict[str, Any]:
        return dict(self.steps)

    # -- sklearn-shaped API ------------------------------------------------

    def fit(self, X: Any) -> dict[str, Any]:
        states: dict[str, Any] = {}
        current = X
        for name, step in self.steps:
            states[name] = step.fit(current)
            current = step.transform(current, states[name])
        self.states_ = states
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

    # -- post-fit step access ------------------------------------------------

    def __getitem__(self, key: str) -> BoundStep:
        if self.states_ is None:
            raise RuntimeError("Pipeline is not fitted; call fit() before step access")
        steps = self.named_steps()
        if key not in steps:
            raise KeyError(f"No step named {key!r}. Steps: {sorted(steps)}")
        return BoundStep(key, steps[key], self.states_[key])

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
            (name, from_spec(step) if step != "passthrough" else step)
            for name, step in spec["kwargs"]["steps"]
        ]
        return cls(steps=steps)


def make_pipeline(*transformers: Any) -> Pipeline:
    """Sklearn's ``make_pipeline``: auto-name steps from transformer classes."""
    steps: list[tuple[str, Any]] = []
    counts: dict[str, int] = {}
    for step in transformers:
        name = type(step).__name__.lower()
        counts[name] = counts.get(name, 0) + 1
        steps.append((name if counts[name] == 1 else f"{name}-{counts[name] - 1}", step))
    return Pipeline(steps=steps)
