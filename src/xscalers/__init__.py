"""sklearn-shaped, dim-aware scaling transformers for xarray and pytensor.xtensor.

- ``Fit`` is eager xarray: the fitted state is an ``xr.Dataset`` (or a dict of
  them for per-variable fits) that you can inspect root, edit root, and write
  to netCDF root.
- ``transform`` / ``inverse_transform`` are elementwise arithmetic by
  dimension name — identical semantics on ``DataArrays`` and symbolic
  ``pytensor.xtensor`` variables; state entries lift via ``ptx.as_xtensor``.
- All v1 scalers are affine: bound only to the composed ``(center, scale)``
  pair via ``pipe.affine(state)``.
"""

from .binding import lift_state, lift_state_if_symbolic
from .container import flatten, reassemble, state_tree
from .core import (
    REGISTRY,
    BaseAffineTransformer,
    BaseTransformer,
    NonInvertibleError,
    affine_pair,
    as_state,
    register,
)
from .datasets import load_channel_tree, load_mixed_grain, load_national
from .dispatch import PerVariable
from .pipeline import BoundStep, Pipeline, make_pipeline
from .scalers import FixedScaler, MeanScaler, ZScore
from .serialize import from_spec, to_spec
from .shape import Selector, Splitter

__all__ = [
    "REGISTRY",
    "BaseAffineTransformer",
    "BaseTransformer",
    "BoundStep",
    "FixedScaler",
    "MeanScaler",
    "NonInvertibleError",
    "PerVariable",
    "Pipeline",
    "Selector",
    "Splitter",
    "ZScore",
    "affine_pair",
    "as_state",
    "flatten",
    "from_spec",
    "lift_state",
    "lift_state_if_symbolic",
    "load_channel_tree",
    "load_mixed_grain",
    "load_national",
    "make_pipeline",
    "reassemble",
    "register",
    "state_tree",
    "to_spec",
]
