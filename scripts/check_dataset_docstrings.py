#!/usr/bin/env python
"""Verify the example datasets' docstring shape claims against real data.

The loaders' docstrings (and the module docstring's bullets) state variable
shapes as ``var (dims)``. This script executes each loader and asserts:

1. every stated shape matches the produced variable's dims (no overclaims),
2. every data variable's dims are stated in its loader's docstring
   (no underclaims).

Exit 0 on success, 1 with the offending claims otherwise. Wired into the
pre-commit hooks; also runnable directly:
``uv run python scripts/check_dataset_docstrings.py``.
"""

from __future__ import annotations

import inspect
import re
import sys

from xscalers import datasets

_LOADERS = [
    datasets.load_media_channels,
    datasets.load_mixed_grain,
    datasets.load_national,
    datasets.load_panel,
]

# ``var (dims)`` claims: word + comma-separated dim names in parentheses
_SHAPE_CLAIM = re.compile(r"(\w+)\s*\(([\w\s,]+)\)")


def _dims_string(var) -> str:
    """The docstring convention: ``(date,)`` for 1-tuples, else ``(a, b)``."""
    dims = tuple(var.dims)
    inner = ", ".join(dims)
    return f"({inner},)" if len(dims) == 1 else f"({inner})"


def _stated_claims(text: str, ds) -> list[tuple[str, tuple[str, ...]]]:
    """Shape claims whose every word is a real dim of this dataset.

    Prose parentheses like "(see :func:`load_mixed_grain`)" name no real
    dims, so they are skipped without special-casing.
    """
    claims = []
    for var, dim_text in _SHAPE_CLAIM.findall(text):
        dims = tuple(d.strip() for d in dim_text.split(","))
        if var in ds.data_vars and all(d in ds.dims for d in dims):
            claims.append((var, dims))
    return claims


def check() -> list[str]:
    """Run both directions of the check; return human-readable failures."""
    module_doc = inspect.getdoc(datasets) or ""
    failures: list[str] = []
    for loader in _LOADERS:
        ds = loader()
        doc = inspect.getdoc(loader) or ""
        claims = _stated_claims(f"{module_doc}\n{doc}", ds)
        if not claims:
            failures.append(f"{loader.__name__}: docstring states no shape claims")
        for var, dims in claims:
            if tuple(ds[var].dims) != dims:
                failures.append(
                    f"{loader.__name__}: docstring says {var} is {dims}, data is {tuple(ds[var].dims)}"
                )
        unstated = [name for name, var in ds.data_vars.items() if _dims_string(var) not in doc]
        if unstated:
            failures.append(f"{loader.__name__}: docstring omits shapes of {unstated}")
    return failures


if __name__ == "__main__":
    problems = check()
    for problem in problems:
        print(problem)
    sys.exit(1 if problems else 0)
