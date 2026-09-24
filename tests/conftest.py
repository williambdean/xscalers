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
def mixed_grain_ds(rng):
    """Variables at different grains: national media, geo media, product controls."""
    dates = np.arange(30)
    return xr.Dataset(
        {
            "media_national": (("date", "media"), rng.normal(20, 2, (30, 2))),
            "media_digital": (("date", "geo", "media"), rng.normal(80, 4, (30, 2, 2))),
            "media_tv": (("date", "media"), rng.normal(1_000, 100, (30, 2))),
            "controls_nat": (("date",), rng.normal(0, 1, 30)),
            "controls_prod": (("date", "product"), rng.normal(0, 1, (30, 4))),
        },
        coords={
            "date": dates,
            "media": ["tv", "ooh"],
            "geo": ["us", "eu"],
            "product": ["p1", "p2", "p3", "p4"],
        },
    )


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
