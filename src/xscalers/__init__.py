"""sklearn-shaped, dim-aware scaling transformers for xarray and pytensor.xtensor."""

from .core import (
    BaseAffineTransformer,
    BaseTransformer,
    NonInvertibleError,
    affine_pair,
    from_spec,
    register,
)

__all__ = [
    "BaseAffineTransformer",
    "BaseTransformer",
    "NonInvertibleError",
    "affine_pair",
    "from_spec",
    "register",
]
