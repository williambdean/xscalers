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

import numpy as np
import xarray as xr

from .container import flatten as _flatten
from .container import reassemble as _reassemble
from .container import state_tree
from .core import from_spec, to_spec

__all__ = ["BoundStep", "Pipeline", "Union", "make_pipeline", "make_union"]


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

        This is a *diagnostic* view of what the pipeline decided (and the
        internals behind :meth:`unscale`); model code should call
        ``pipe.unscale(x, var=...)`` rather than applying the pair by hand.
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

    def unscale(self, x: Any, *, var: str, state: Mapping[str, Any] | None = None) -> Any:
        """Convert a scaled-space contribution back into real units — one verb.

        For affine pipelines this applies the composed ``(center, scale)``
        pair per variable (symbolic ``pytensor.xtensor`` contributions com-
        pose identically). For pipelines that contain non-affine steps (e.g.
        a log transform on this variable) the fallback walks *this varia-
        ble's own* steps backwards — un-log, un-scale — stopping at the step
        that *created* the variable (e.g. ``Derive``), so the result is the
        pre-pipeline value of that variable in real units, not a knock-on
        restoration of unrelated inputs. `var` names a **data variable** of
        the contributions (never a coordinate). The state defaults to the
        fitted ``states_``; pass ``state=`` explicitly for pipelines rebuilt
        from specs.
        """
        chosen = state if state is not None else self.states_
        if chosen is None:
            raise RuntimeError(
                "Pipeline has no fitted state (was this pipeline rebuilt with "
                "from_spec?); pass state=<state> explicitly or bind one"
            )
        if any(getattr(step, "affine_view", None) is None for _, step in self.steps):
            # non-affine pipeline: partial inverse — from the last step back,
            # owned variables are inverted; creation steps STOP the walk
            current_value: Any = (
                xr.Dataset({var: x}) if isinstance(x, xr.DataArray) else x
            )
            if var not in current_value.data_vars:
                raise KeyError(
                    f"No data variable named {var!r}; variables: "
                    f"{sorted(map(str, current_value.data_vars))}"
                )
            for name, step in reversed(self.steps):
                step_state = chosen[name]
                if _step_creates_var(step, step_state, var):
                    break  # stop *before* the step that created the variable
                if _step_owns(step, step_state, var):
                    current_value = step.inverse_transform(current_value, step_state)
            if isinstance(x, xr.DataArray):
                return current_value[var]
            return current_value
        try:
            pair = self.affine(chosen)[var]
        except KeyError as err:
            composed = self.affine(chosen)
            raise KeyError(
                f"No data variable named {var!r} in the composed affine view; "
                f"available: {sorted(str(key) for key in composed)}"
            ) from err
        from .binding import lift_state_if_symbolic

        pair_lifted = lift_state_if_symbolic(x, pair)
        return x * pair_lifted["scale"] + pair_lifted["center"]

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


class Union:
    """Many pipelines over the same dataset, merged back into one container.

    sklearn's ``FeatureUnion``/``make_union`` composition, xarray-native:
    every branch fits and transforms the input independently; the outputs are
    merged by variable name under a collision policy.

    - ``collisions="require_equal"`` (default): variables present in several
      branches must transform identically — shared passthroughs merge
      silently, genuinely conflicting recipes raise naming branch+variable.
    - ``collisions="error"``: branches must produce disjoint variable sets.
    - ``collisions="keep_first"``: first branch wins; discouraged (silent).

    The state nests one level per branch (``{"branch": branch_state}``), so
    ``state_tree`` makes the whole fitted union a single artifact.
    ``inverse_transform`` runs each branch's chain backwards, then applies the
    same collision policy — derived variables get dropped per-branch, so
    pipelines containing unions still round-trip.

    Unions are **non-affine by design**: ``Pipeline.affine()`` refuses
    pipelines containing one (the covariates-only pattern; real-terms
    conversion for target-like paths goes through ``inverse_transform``).
    """

    def __init__(
        self,
        branches: list[tuple[str, Any]],
        collisions: str = "require_equal",
    ):
        if collisions not in {"require_equal", "error", "keep_first"}:
            raise ValueError(
                f"collisions must be require_equal | error | keep_first; got {collisions!r}"
            )
        seen: dict[str, int] = {}
        resolved: list[tuple[str, Any]] = []
        for name, transformer in branches:
            count = seen.get(name, 0) + 1
            seen[name] = count
            resolved.append((name if count == 1 else f"{name}-{count - 1}", transformer))
        self.branches = resolved
        self.collisions = collisions
        self.states_: dict[str, Any] | None = None

    # -- sklearn-composite API -----------------------------------------------

    def fit(self, X: Any) -> dict[str, Any]:
        states: dict[str, Any] = {}
        for name, branch in self.branches:
            states[name] = branch.fit(X)
        self.states_ = states
        return states

    def transform(self, X: Any, state: Mapping[str, Any]) -> Any:
        merged: dict[str, Any] = {}
        for name, branch in self.branches:
            out = branch.transform(X, state[name])
            self._merge(merged, _flatten(out), name)
        return _reassemble(X, merged)

    def inverse_transform(self, X: Any, state: Mapping[str, Any]) -> Any:
        # per-branch inverse; a variable dropped by ANY branch is dropped from
        # the result (each branch removes its own derived vars), and variables
        # surviving in every branch must restore identically
        dropped: set[str] = set()
        outputs: list[Mapping[str, Any]] = []
        for name, branch in reversed(self.branches):
            flat = _flatten(branch.inverse_transform(X, state[name]))
            dropped |= {key for key in _flatten(X) if key not in flat}
            outputs.append(flat)
        out = {key: value for key, value in outputs[0].items() if key not in dropped}
        return _reassemble(X, out)

    def fit_transform(self, X: Any) -> Any:
        state = self.fit(X)
        return self.transform(X, state)

    def _merge(
        self,
        merged: dict[str, Any],
        branch_out: Mapping[str, Any],
        branch: str,
    ) -> None:
        for key, value in branch_out.items():
            if key not in merged:
                merged[key] = value
                continue
            existing = merged[key]
            if self.collisions == "keep_first":
                continue
            if isinstance(existing, xr.DataArray) and isinstance(value, xr.DataArray):
                same_dims = existing.dims == value.dims
                same_values = same_dims and bool(np.allclose(existing.values, value.values, equal_nan=True))
                if same_values:
                    continue
            if self.collisions == "require_equal":
                raise ValueError(
                    f"Union branches {branch!r} and an earlier branch produce conflicting "
                    f"values for variable {key!r} (collisions={self.collisions!r}); "
                    "either scale a variable in one branch only, or pass collisions=\'error\'/'keep_first'"
                )
            raise ValueError(
                f"Union: variable {key!r} collides between branches under "
                f"collisions={self.collisions!r}"
            )

    def __getitem__(self, key: str) -> BoundStep:
        if self.states_ is None:
            raise RuntimeError("Union is not fitted; call fit() before step access")
        steps = dict(self.branches)
        if key not in steps:
            raise KeyError(f"No branch named {key!r}. Branches: {sorted(steps)}")
        return BoundStep(key, steps[key], self.states_[key])

    # -- spec serialization ----------------------------------------------------

    def to_spec(self) -> dict[str, Any]:
        return {
            "kind": "Union",
            "kwargs": {
                "branches": [[name, to_spec(branch)] for name, branch in self.branches],
                "collisions": self.collisions,
            },
        }

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> Union:
        branches = [
            (name, from_spec(spec_branch) if spec_branch != "passthrough" else spec_branch)
            for name, spec_branch in spec["kwargs"]["branches"]
        ]
        return cls(branches=branches, collisions=spec["kwargs"]["collisions"])


def make_union(*branches: Any, collisions: str = "require_equal") -> Union:
    """sklearn's ``make_union``: auto-name branches from transformer classes.

    Names derive from transformer or Pipeline class names (``zscore``,
    ``zscore-1``, ``pipeline``); pass ``("name", transformer)`` tuples to pin
    names explicitly. Same suffixing convention as ``make_pipeline``.
    """
    resolved: list[tuple[str, Any]] = []
    counts: dict[str, int] = {}
    for branch in branches:
        if isinstance(branch, tuple):
            resolved.append((branch[0], branch[1]))
            continue
        base = type(branch).__name__.lower()
        count = counts.get(base, 0) + 1
        counts[base] = count
        resolved.append((base if count == 1 else f"{base}-{count - 1}", branch))
    return Union(branches=resolved, collisions=collisions)



def _step_owns(step: Any, step_state: Any, var: str) -> bool:
    """Does this step transform ``var`` (i.e. would its inverse participate)?"""
    # VariableTransformer/dispatch: the per-variable mapping owns vars whose key exists
    if isinstance(step_state, Mapping) and var in step_state:
        return True
    # dataset-level per-variable states nest one further level
    if isinstance(step_state, Mapping):
        for value in step_state.values():
            if isinstance(value, Mapping) and var in value:
                return True
    # scalar-step states: the affine view owns variables listed in its state
    view = getattr(step, "affine_view", None)
    if view is not None:
        return var in getattr(view(step_state), "data_vars", {})
    return False


def _step_creates_var(step: Any, step_state: Any, var: str) -> bool:
    """True when the step *produced* the variable (a Derive-style check)."""
    attrs = getattr(step_state, "attrs", None)
    return isinstance(attrs, Mapping) and attrs.get("derived") == var
