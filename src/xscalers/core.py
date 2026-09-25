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
from typing import Any

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
    "register",
    "to_spec",
]


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

    dims: tuple[str, ...] | None

    def __init__(self, dims: tuple[str, ...] | str | None = None):
        self.dims = (dims,) if isinstance(dims, str) else dims

    @abstractmethod
    def compute(self, X: Any) -> xr.Dataset:
        """Data-derived fitted state."""

    def _transform(
        self, X: Any, state: Mapping[str, xr.DataArray] | xr.Dataset
    ) -> Any:
        raise NotImplementedError

    def _inverse_transform(
        self, X: Any, state: Mapping[str, xr.DataArray] | xr.Dataset
    ) -> Any:
        raise NotImplementedError

    def fit(self, X: Any) -> Any:
        """Fit: DataArrays (and symbolic inputs) directly; containers via dispatch."""
        from .compose import PerVariable

        if self.__scalarish(X):
            return self.compute(X)
        return PerVariable(default=self).fit(X)

    def transform(
        self, X: Any, state: Mapping[str, xr.DataArray] | xr.Dataset
    ) -> Any:
        from .compose import PerVariable

        if self.__scalarish(X):
            return self._transform(X, state)
        return PerVariable(default=self).transform(X, state)

    def inverse_transform(
        self, X: Any, state: Mapping[str, xr.DataArray] | xr.Dataset
    ) -> Any:
        from .compose import PerVariable

        if self.__scalarish(X):
            return self._inverse_transform(X, state)
        return PerVariable(default=self).inverse_transform(X, state)

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
    """Scalers whose transform is the affine pair: y = (x - center) / scale."""

    def _transform(
        self, X: Any, state: Mapping[str, xr.DataArray] | xr.Dataset
    ) -> Any:
        from .binding import lift_state_if_symbolic

        state = lift_state_if_symbolic(X, state)
        return (X - state["center"]) / state["scale"]

    def _inverse_transform(
        self, X: Any, state: Mapping[str, xr.DataArray] | xr.Dataset
    ) -> Any:
        from .binding import lift_state_if_symbolic

        state = lift_state_if_symbolic(X, state)
        return X * state["scale"] + state["center"]

    def affine_view(self, state: Mapping[str, xr.DataArray]) -> Mapping[str, Any]:
        return state


def affine_pair(state: Mapping[str, xr.DataArray]) -> dict[str, Any]:
    """The (center, scale) pair of a fitted affine state."""
    return {"center": state["center"], "scale": state["scale"]}


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
