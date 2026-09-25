"""Reproducible example datasets, in the spirit of ``sklearn.datasets``.

Every function is deterministic (fixed seed), so README examples and tests
produce identical data anywhere. Sizes are deliberately MMM-shaped:

- :func:`load_national` — a national model: daily target, media spend by
  channel, one competitor control.
- :func:`load_mixed_grain` — variables at different grains (national media,
  geo-grain digital media, product-grain controls), the case that motivated
  per-variable dispatch and per-variable dims.
- :func:`load_media_channels` — a single ``media (date, channel)`` variable plus
  controls in one Dataset, the raw-input shape for split-by-channel treatment
  (DataTree inputs remain supported by the library, but real MMM inputs are
  usually just datasets).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

__all__ = ["load_media_channels", "load_mixed_grain", "load_national"]

_CHANNELS = ["search", "social", "fb", "tv", "stream"]


def _dates(n: int) -> pd.DatetimeIndex:
    """Daily grid so examples read like a real MMM."""
    return pd.date_range("2024-01-01", periods=n, freq="D")


def load_national() -> xr.Dataset:
    """National model: daily ``target``, ``media (date, channel)``, controls."""
    rng = np.random.default_rng(12)
    trend = np.linspace(90, 110, 104)
    return xr.Dataset(
        {
            "target": (("date",), trend + rng.normal(0, 4, 104)),
            "media": (
                ("date", "channel"),
                np.abs(rng.gamma(2.0, 25.0, size=(104, 3))),
            ),
            "controls": (("date",), rng.normal(10, 1, 104)),
        },
        coords={"date": _dates(104), "channel": _CHANNELS[:3]},
    )


def load_mixed_grain() -> xr.Dataset:
    """Variables at different grains for per-variable dispatch examples.

    Includes a ``controls (date, product, control)`` variable whose
    ``control`` coordinate holds ``[price, comp_price]`` — the shape for
    derived-metric examples (e.g. the price/competitor-price ratio).
    """
    rng = np.random.default_rng(31)
    return xr.Dataset(
        {
            "media_national": (
                ("date", "media"),
                np.abs(rng.gamma(2.0, 20.0, size=(104, 2))),
            ),
            "media_digital": (
                ("date", "geo", "media"),
                np.abs(rng.gamma(2.0, 5.0, size=(104, 2, 2))),
            ),
            "media_tv": (("date", "media"), np.abs(rng.gamma(2.0, 900.0, size=(104, 2)))),
            "controls_nat": (("date",), rng.normal(0, 1, 104)),
            "controls_prod": (("date", "product"), rng.normal(0, 1, (104, 4))),
            "controls": (
                ("date", "product", "control"),
                np.abs(rng.gamma(2.0, 3.0, size=(104, 4, 3))),
            ),
        },
        coords={
            "date": _dates(104),
            "media": ["tv", "ooh"],
            "geo": ["us", "eu"],
            "product": ["p1", "p2", "p3", "p4"],
            "control": ["price", "comp_price", "inflation"],
        },
    )


def load_media_channels() -> xr.Dataset:
    """A realistic MMM input: one ``media (date, channel)`` variable needing
    split treatment, alongside controls — as a single Dataset of variables
    (the common shape; DataTree inputs remain supported by the library, but
    real raw inputs are usually just datasets)."""
    rng = np.random.default_rng(7)
    n = 60
    return xr.Dataset(
        {
            "media": (("date", "channel"), np.abs(rng.gamma(2.0, 40.0, (n, 5)))),
            "controls": (("date",), rng.normal(5, 1, n)),
        },
        coords={"date": _dates(n), "channel": _CHANNELS},
    )
