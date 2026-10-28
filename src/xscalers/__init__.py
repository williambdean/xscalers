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
from .compose import VariableTransformer
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
from .datasets import load_media_channels, load_mixed_grain, load_national, load_panel
from .persistence import append_state
from .pipeline import BoundStep, Pipeline, Union, make_pipeline, make_union
from .preprocessing import (
    Derive,
    Dropper,
    FixedScaler,
    Func,
    FunctionTransformer,
    MeanScaler,
    Renamer,
    RollingMean,
    Selector,
    Splitter,
    VariableDropper,
    ZScore,
)
from .serialize import from_spec, provenance, to_spec

__all__ = [
    "REGISTRY",
    "BaseAffineTransformer",
    "BaseTransformer",
    "BoundStep",
    "Derive",
    "Dropper",
    "FixedScaler",
    "Func",
    "FunctionTransformer",
    "MeanScaler",
    "NonInvertibleError",
    "Pipeline",
    "Renamer",
    "RollingMean",
    "Selector",
    "Splitter",
    "Union",
    "VariableDropper",
    "VariableTransformer",
    "ZScore",
    "affine_pair",
    "append_state",
    "as_state",
    "flatten",
    "from_spec",
    "lift_state",
    "lift_state_if_symbolic",
    "load_media_channels",
    "load_mixed_grain",
    "load_national",
    "load_panel",
    "make_pipeline",
    "make_union",
    "provenance",
    "reassemble",
    "register",
    "state_tree",
    "to_spec",
]
