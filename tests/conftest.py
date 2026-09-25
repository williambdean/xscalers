"""Fixtures capturing the MMM-shaped scenarios the library must serve."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def national_ds(rng):
    """National model: target + media + control at daily grain."""
    return xr.Dataset(
        {
            "target": (("date",), rng.normal(100, 5, 30)),
            "media": (("date", "channel"), rng.normal(50, 10, (30, 3))),
        },
        coords={
            "date": np.arange(30),
            "channel": ["search", "social", "fb"],
        },
    )


@pytest.fixture
def mixed_grain_ds():
    """Variables at different grains: national media, geo media, product
    controls (+ a controls variable whose `control` coordinate holds
    level/reference, the derived-metrics shape)."""
    from xscalers.datasets import load_mixed_grain

    return load_mixed_grain()


@pytest.fixture
def split_tree(rng):
    """DataTree: one media variable whose channels need different treatment."""
    return xr.DataTree.from_dict(
        {
            "/media": xr.Dataset(
                {
                    "media": (("date", "channel"), rng.normal(10, 2, (20, 5))),
                },
                coords={
                    "date": np.arange(20),
                    "channel": ["search", "social", "fb", "tv", "stream"],
                },
            ),
            "/controls": xr.Dataset(
                {"controls": (("date",), rng.normal(0, 1, 20))},
                coords={"date": np.arange(20)},
            ),
        }
    )
