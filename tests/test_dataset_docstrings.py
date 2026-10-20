"""Docstring shape claims are data-verified for the example datasets.

The loaders' docstrings (and the module docstring's bullets) state variable
shapes as ``var (dims)``. This test executes each loader and asserts:

1. every stated shape matches the produced variable's dims (no overclaims),
2. every data variable's dims are stated in its loader's docstring
   (no underclaims), so the docstrings stay a complete, checked spec.
"""

import inspect
import re

import pytest

from xscalers.datasets import (
    load_media_channels,
    load_mixed_grain,
    load_national,
    load_panel,
)

_LOADERS = [load_media_channels, load_mixed_grain, load_national, load_panel]

_SHAPE_CLAIM = re.compile(r"(\w+)\s*\(([\w\s,]+)\)")


def _dims_string(var) -> str:
    dims = tuple(var.dims)
    inner = ", ".join(dims)
    return f"({inner},)" if len(dims) == 1 else f"({inner})"


def _stated_claims(text: str, ds) -> list[tuple[str, tuple[str, ...]]]:
    """``var (dims)`` claims whose words are real dims of this dataset."""
    claims = []
    for var, dim_text in _SHAPE_CLAIM.findall(text):
        dims = tuple(d.strip() for d in dim_text.split(","))
        if var in ds.data_vars and all(d in ds.dims for d in dims):
            claims.append((var, dims))
    return claims


@pytest.mark.parametrize("loader", _LOADERS, ids=[f.__name__ for f in _LOADERS])
def test_shape_claims_are_true(loader) -> None:
    ds = loader()
    module_doc = inspect.getdoc(inspect.getmodule(loader)) or ""
    claims = _stated_claims(f"{module_doc}\n{inspect.getdoc(loader)}", ds)
    assert claims, "docstring states no checkable shape claims"
    for var, dims in claims:
        assert tuple(ds[var].dims) == dims, f"{loader.__name__} docstring: {var} shape"


@pytest.mark.parametrize("loader", _LOADERS, ids=[f.__name__ for f in _LOADERS])
def test_every_variable_shape_is_stated(loader) -> None:
    ds = loader()
    doc = inspect.getdoc(loader) or ""
    unstated = [
        name for name, var in ds.data_vars.items() if _dims_string(var) not in doc
    ]
    assert not unstated, f"{loader.__name__} docstring omits shapes of {unstated}"
