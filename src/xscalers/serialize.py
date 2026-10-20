"""Plain-dict spec serialization: recipes persist as config; state rides as netCDF."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["from_spec", "provenance", "spec_of", "to_spec"]


def to_spec(transformer: Any) -> dict[str, Any]:
    """Serialize a transformer recipe to a plain dict (JSON-serializable)."""
    # containers (Pipeline, Union) override ``to_spec`` with a nested spec of
    # their children and have no to_spec_kwargs -- from_spec understands those
    # kinds, so ask them. Leaves follow the to_spec_kwargs protocol below.
    override = type(transformer).__dict__.get("to_spec")
    if override is not None and not hasattr(type(transformer), "to_spec_kwargs"):
        return override(transformer)
    kwargs = {k: (_spec_value(v)) for k, v in transformer.to_spec_kwargs().items()}
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
    if kind == "Union":
        from .pipeline import Union

        return Union.from_spec(spec)
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


def _is_spec(value):
    return isinstance(value, Mapping) and "kind" in value


def _recipe_line(spec: Mapping[str, Any]) -> str:
    from .core import _recipe_from_spec

    return _recipe_from_spec(spec)


def provenance(spec: Mapping[str, Any]) -> Any:
    """Machine-readable lineage: a (step, variable) table from a recipe spec.

    Pure configuration: no fitted state, no objects. Rows exist only for
    variables a step touches; absence IS the answer for "untouched by".
    Columns: ``recipe`` (one-liner), ``kind``, ``op`` (named elementwise
    functions), ``affine`` (contributes an exact affine answer; data
    movement counts as identity), ``role`` (transformed / created / split /
    dropped). Unknown kinds land affine=False: refusal is the default.
    """
    import xarray as xr

    if spec.get("kind") != "Pipeline":
        raise TypeError(f"provenance expects a Pipeline spec, got {spec.get('kind')!r}")

    rows: dict[str, list] = {"recipe": [], "kind": [], "op": [], "affine": [], "role": []}
    steps: list[str] = []
    row_keys: list[tuple[str, str]] = []
    variables: set[str] = set()

    def row(step, variable, *, kind, recipe, op, affine, role):
        steps.append(step)
        row_keys.append((step, variable))
        variables.add(variable)
        rows["recipe"].append(recipe)
        rows["kind"].append(kind)
        rows["op"].append(op)
        rows["affine"].append(affine)
        rows["role"].append(role)

    from .core import REGISTRY

    for step_name, step_spec in spec["kwargs"]["steps"]:
        kind = step_spec["kind"]
        kw = step_spec["kwargs"]
        step_cls = REGISTRY.get(kind)
        described = (
            step_cls.describe(kw)
            if step_cls is not None
            else [
                {
                    "variable": None,
                    "role": "transformed",
                    "kind": kind,
                    "recipe": _recipe_line(step_spec),
                    "op": "",
                    "affine": False,
                }
            ]
        )
        for r in described:
            row(
                step_name,
                str(r.get("variable") or f"<{kind}>"),
                kind=r["kind"],
                recipe=r.get("recipe") or _recipe_line(step_spec),
                op=str(r.get("op", "")),
                affine=bool(r.get("affine", False)),
                role=r["role"],
            )

    import numpy as np

    step_names = list(dict.fromkeys(steps))
    var_names = sorted(variables)
    grid = xr.Dataset(coords={"step": step_names, "variable": var_names})
    si = {s: i for i, s in enumerate(step_names)}
    vi = {v: i for i, v in enumerate(var_names)}
    for col, default, dtype in (
        ("recipe", "", "U256"),
        ("kind", "", "U64"),
        ("op", "", "U32"),
        ("affine", False, bool),
        ("role", "", "U32"),
    ):
        arr = np.full((len(step_names), len(var_names)), default, dtype=dtype)
        for (step, variable), value in zip(row_keys, rows[col], strict=True):
            arr[si[step], vi[variable]] = value
        grid[col] = (("step", "variable"), arr)
    return grid


class SpecMixin:
    """Scalers write their constructor kwargs here; spec pairs with from_spec."""

    def to_spec_kwargs(self) -> dict[str, Any]:
        raise NotImplementedError
