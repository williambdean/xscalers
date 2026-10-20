"""The transformer contract: explicit, ecosystem-native state; affine views; spec registry.

State conventions
-----------------
- A fitted state is an ``xr.Dataset`` whose data_vars are the learned values
  (0-d scalars or DataArrays over the kept dims) and whose ``attrs`` carry the
  metadata: ``kind``, ``dims``, ``scaled_from``.
- Values live as data variables, never as attrs (attrs must stay JSON/netCDF
  scalar-safe for round-trips).
- The contract types as ``Mapping[str, DataArray]`` so a plain dict of symbolic
  (``pytensor.xtensor``) entries substitutes transparently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
import xarray as xr

__all__ = [
    "REGISTRY",
    "BaseAffineTransformer",
    "BaseTransformer",
    "NonInvertibleError",
    "affine_pair",
    "as_state",
    "from_spec",
    "is_symbolic",
    "recipe_repr",
    "register",
    "spec_sha",
    "to_spec",
]


def recipe_repr(transformer: Any) -> str:
    """Human-readable one-liner of a transformer's recipe.

    Constructor style, nested specs recurse (e.g.
    ``VariableTransformer(mapping={'tv': ZScore(dims=['date'])})``).
    Best-effort: non-serializable transformers raise through.
    """
    from .serialize import to_spec

    return _recipe_from_spec(to_spec(transformer))


def _recipe_from_spec(spec: Any) -> str:
    if isinstance(spec, str):
        return spec
    if not isinstance(spec, Mapping) or "kind" not in spec:
        return repr(spec)
    kind = spec["kind"]
    kwargs = spec.get("kwargs", {}) or {}
    parts = [
        f"{key}={_recipe_from_spec(value)}" if _is_spec(value) else f"{key}={value!r}"
        for key, value in kwargs.items()
        if value is not None
    ]
    return f"{kind}({', '.join(parts)})"


def _is_spec(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and "kind" in value
        and "kwargs" in value
        and isinstance(value["kwargs"], Mapping)
    ) or (
        isinstance(value, (list, tuple))
        and bool(value)
        and all(isinstance(item, Mapping) and "kind" in item for item in value)
    )


def spec_sha(transformer: Any) -> str:
    """Deterministic 12-hex fingerprint of a transformer's recipe.

    Source of the hash: ``to_spec`` (JSON, sort_keys). Non-serializable
    transformers raise whatever ``to_spec_kwargs`` raises — stamping is
    best-effort and callers should tolerate that.
    """
    import hashlib
    import json

    from .serialize import to_spec

    blob = json.dumps(to_spec(transformer), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _stamp_state(state: Any, transformer: Any) -> Any:
    """Best-effort provenance: stamp ``spec_sha`` onto state Dataset leaves.

    Mutates and returns ``state`` (states are freshly built in ``compute``
    and dispatch). A fingerprint only — the recipe itself lives in the
    recipe artifact, never in the state (recipe/state never mix).
    """
    try:
        sha = spec_sha(transformer)
        recipe = recipe_repr(transformer)
    except (NotImplementedError, AttributeError, TypeError):
        return state  # non-serializable transformer: no stamp, no harm

    def walk(node: Mapping[str, Any]) -> None:
        for value in node.values():
            if isinstance(value, xr.Dataset):
                value.attrs["spec_sha"] = sha
                value.attrs["recipe"] = recipe
            elif isinstance(value, Mapping):
                walk(value)

    if isinstance(state, xr.Dataset):
        state.attrs["spec_sha"] = sha
        state.attrs["recipe"] = recipe
    elif isinstance(state, Mapping):
        walk(state)
    return state


class NonInvertibleError(RuntimeError):
    """Raised by steps that change shape and cannot honestly be reversed."""


REGISTRY: dict[str, type[Any]] = {}


def register(cls: Any) -> Any:
    """Register a transformer class under its own name for ``from_spec``."""
    REGISTRY[cls.__name__] = cls
    return cls


def is_symbolic(x: Any) -> bool:
    """True for pytensor xtensor variables/constants.

    Lazy import: the xarray path never needs pytensor installed.
    """
    try:
        import pytensor.xtensor.type as ptx_type
    except ImportError:
        return False
    return isinstance(x, ptx_type.XTensorVariable)


def expand(value: Any) -> Any:
    """Normalize a state entry value: arrays/lists → DataArray, scalars stay."""
    if isinstance(value, (xr.DataArray, xr.Dataset)):
        return value
    if isinstance(value, np.ndarray):
        return xr.DataArray(value)
    return value


def as_state(
    kind: str,
    dims: tuple[str, ...] | None,
    scaled_from: str | None = None,
    attrs: Mapping[str, Any] | None = None,
    **entries: Any,
) -> xr.Dataset:
    """Build a fitted state: values as data vars, metadata in attrs."""
    state = xr.Dataset({k: expand(v) for k, v in entries.items()})
    state.attrs.update(
        {
            "kind": kind,
            # None-dims (reduce over everything) persists as "__all__" for
            # JSON/netCDF-safety; from state construction it's re-readable
            "dims": list(dims) if dims is not None else "__all__",
            # None-scaled_from (e.g. shape steps like Derive) → "" for netCDF
            "scaled_from": scaled_from or "",
        }
    )
    if attrs:
        state.attrs.update(attrs)
    return state


class BaseTransformer(ABC):
    """The xscalers contract, sklearn vocabulary over ecosystem-native objects."""

    @classmethod
    def describe(cls, spec: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Provenance rows this kind contributes: the protocol ``provenance``
        walks. Spec kwargs in, plain dicts out. Default: one transformed row
        with no known variable; step-shaped transformers (splitters, derivers,
        droppers, function maps) override to declare their variables, roles,
        and ops, so customs are first-class without library edits."""
        from .serialize import _recipe_line

        return [
            {
                "variable": None,
                "role": "transformed",
                "kind": cls.__name__,
                "recipe": _recipe_line({"kind": cls.__name__, "kwargs": dict(spec)}),
                "op": "",
                "affine": hasattr(cls, "affine_view"),
            }
        ]

    dims: tuple[str, ...] | None

    def __init__(self, dims: tuple[str, ...] | str | None = None):
        self.dims = (dims,) if isinstance(dims, str) else dims

    @abstractmethod
    def compute(self, X: Any) -> xr.Dataset:
        """Data-derived fitted state."""

    def _transform(self, X: Any, state: Mapping[str, xr.DataArray] | xr.Dataset) -> Any:
        raise NotImplementedError

    def _inverse_transform(self, X: Any, state: Mapping[str, xr.DataArray] | xr.Dataset) -> Any:
        raise NotImplementedError

    def fit(self, X: Any) -> Any:
        """Fit: DataArrays (and symbolic inputs) directly; containers via dispatch.

        Stamps ``spec_sha`` provenance onto the returned state leaves
        (best-effort; see :func:`_stamp_state`).
        """
        from .compose import VariableTransformer

        if self.__scalarish(X):
            return _stamp_state(self.compute(X), self)
        return _stamp_state(VariableTransformer(remainder=self).fit(X), self)

    def transform(self, X: Any, state: Mapping[str, xr.DataArray] | xr.Dataset) -> Any:
        from .compose import VariableTransformer

        if self.__scalarish(X):
            return self._transform(X, state)
        return VariableTransformer(remainder=self).transform(X, state)

    def inverse_transform(self, X: Any, state: Mapping[str, xr.DataArray] | xr.Dataset) -> Any:
        from .compose import VariableTransformer

        if self.__scalarish(X):
            return self._inverse_transform(X, state)
        return VariableTransformer(remainder=self).inverse_transform(X, state)

    @staticmethod
    def __scalarish(X: Any) -> bool:
        """True for non-Dataset inputs, which compute/transform handle directly."""
        return not isinstance(X, xr.Dataset)

    def to_spec(self) -> dict[str, Any]:
        from .serialize import spec_of

        return spec_of(self)

    def _reduce_dims(self, X: Any) -> tuple[str, ...]:
        """Dims of X to reduce over: ``None`` means all of them."""
        data_dims: tuple[str, ...] | None = (
            tuple(str(d) for d in X.dims) if isinstance(X, xr.DataArray) else None
        )
        if self.dims is None:
            return _all_dims(X, data_dims)
        if data_dims is None:
            return tuple(self.dims)
        return tuple(d for d in self.dims if d in data_dims)


def _all_dims(X: Any, data_dims: tuple[str, ...] | None) -> tuple[str, ...]:
    if data_dims is None:
        raise TypeError(f"Cannot infer reduce dims from {type(X).__name__}")
    return data_dims


class BaseAffineTransformer(BaseTransformer):
    """Scalers whose transform is the affine pair: y = (x - center) / scale.

    States carry *statistics* — the entries people think in (``mean``,
    ``std``); ``PAIR`` maps the affine *roles* onto those entries. A role
    with no entry (divisive scalers have no center) defaults to ``0.0``.
    The composed affine *view* (:meth:`affine_view`) always carries both
    roles, so downstream contracts (``affine()``, ``unscale()``) stay total.
    """

    PAIR: ClassVar[dict[str, str]] = {"center": "center", "scale": "scale"}

    @classmethod
    def _entry(cls, state: Mapping[str, xr.DataArray] | xr.Dataset, role: str) -> Any:
        entry = cls.PAIR.get(role)
        if entry is None or entry not in state:
            if role == "scale":
                raise KeyError(f"no {entry!r} entry for the scale role in {sorted(map(str, state))}")
            return xr.DataArray(0.0)  # center: additive identity
        return state[entry]

    def _transform(self, X: Any, state: Mapping[str, xr.DataArray] | xr.Dataset) -> Any:
        from .binding import lift_state_if_symbolic

        state = lift_state_if_symbolic(X, state)
        return (X - self._entry(state, "center")) / self._entry(state, "scale")

    def _inverse_transform(self, X: Any, state: Mapping[str, xr.DataArray] | xr.Dataset) -> Any:
        from .binding import lift_state_if_symbolic

        state = lift_state_if_symbolic(X, state)
        return X * self._entry(state, "scale") + self._entry(state, "center")

    def affine_view(self, state: Mapping[str, xr.DataArray]) -> dict[str, Any]:
        """The composed affine roles for this scaler's state — always total."""
        return {"center": self._entry(state, "center"), "scale": self._entry(state, "scale")}


def affine_pair(
    state: Mapping[str, xr.DataArray], transformer: BaseAffineTransformer | None = None
) -> dict[str, Any]:
    """The (center, scale) roles of a fitted affine state.

    States carry *statistics*; the role mapping lives on the scaler class
    (``PAIR``), so pass the transformer for states whose entries are not
    literally ``center``/``scale``. A missing center role defaults to 0.0
    (divisive scalers).
    """
    if transformer is None:
        return {
            "center": state["center"] if "center" in state else xr.DataArray(0.0),
            "scale": state["scale"],
        }
    return {
        "center": transformer._entry(state, "center"),
        "scale": transformer._entry(state, "scale"),
    }


def to_spec(transformer: Any) -> dict[str, Any]:
    """Serialize a transformer's recipe to a plain dict."""
    from .serialize import to_spec as _to_spec

    return _to_spec(transformer)


def from_spec(spec: Mapping[str, Any]) -> Any:
    """Rebuild a transformer from a plain-dict spec."""
    from .serialize import from_spec as _from_spec

    return _from_spec(spec)


def lift_state_if_symbolic(X: Any, state: Mapping[str, xr.DataArray]) -> Mapping[str, Any]:
    """Coerce state entries when X is symbolic (no-op for xarray inputs)."""
    if not is_symbolic(X):
        return state
    from .binding import lift_state

    return lift_state(state)
