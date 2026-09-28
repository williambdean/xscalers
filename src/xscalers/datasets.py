"""Reproducible example datasets, in the spirit of ``sklearn.datasets``.

Every function is deterministic (fixed seed), so README examples and tests
produce identical data anywhere. Sizes are deliberately panel-shaped:

- :func:`load_national` — one series per channel plus a target and a control,
  all at ``(date,)`` grain.
- :func:`load_mixed_grain` — variables at different grains (national media,
  geo-grain digital media, product-grain controls), the case that motivated
  per-variable dispatch and per-variable dims.
- :func:`load_media_channels` — a single ``media (date, channel)`` variable plus
  controls in one Dataset, the raw-input shape for split-by-channel treatment
  (DataTree inputs remain supported by the library, but real inputs are
  usually just datasets).
- :func:`load_panel` — a ``(date, product)`` panel with inputs at two grains
  (shared + per-product) and inputs that differ in character, the case that
  motivates per-variable dispatch at more than one grain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

__all__ = [
    "load_media_channels",
    "load_mixed_grain",
    "load_national",
    "load_panel",
]

_CHANNELS = ["search", "social", "fb", "tv", "stream"]


def _dates(n: int) -> pd.DatetimeIndex:
    """Daily grid so examples read like a real calendar."""
    return pd.date_range("2024-01-01", periods=n, freq="D")


def load_national() -> xr.Dataset:
    """``target`` and ``controls`` at ``(date,)``, ``media`` at
    ``(date, channel)``."""
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

    Two shapes matter for derived metrics: ``controls_prod (date, product)``
    is positive, so it can serve as the denominator of a cross-grain ratio
    (a national media series over a product-grain control, broadcasting to
    product grain), and ``controls (date, product, control)`` carries an
    extra coordinate dimension whose labels are generic
    ``[level, reference, seasonality]`` — the shape for selecting within
    one variable and reducing a group of coordinate values.
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
            "controls_prod": (("date", "product"), np.abs(rng.gamma(2.0, 5.0, (104, 4)))),
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
            "control": ["level", "reference", "seasonality"],
        },
    )


def load_media_channels() -> xr.Dataset:
    """One ``media (date, channel)`` variable needing split treatment,
    alongside controls — as a single Dataset of variables (the common shape;
    DataTree inputs remain supported by the library, but real raw inputs are
    usually just datasets)."""
    rng = np.random.default_rng(7)
    n = 60
    return xr.Dataset(
        {
            "media": (("date", "channel"), np.abs(rng.gamma(2.0, 40.0, (n, 5)))),
            "controls": (("date",), rng.normal(5, 1, n)),
        },
        coords={"date": _dates(n), "channel": _CHANNELS},
    )


def load_panel() -> xr.Dataset:
    """A ``(date, product)`` panel whose inputs sit at *two grains*, each
    variable on its own dim name.

    - ``media_national (date, channel)`` — one series per channel, shared by
      every product. Lifting it to product grain is an upstream xarray op (see
      :func:`load_mixed_grain`); the point is that a *scaled* shared series
      and a *scaled* per-product series keep separate fitted states and
      separate affine pairs.
    - ``media_digital (date, product, digital)`` — per-product, splittable on
      a *different* dim name, because a Dataset holds one coordinate per dim.
    - ``level`` / ``peer_level`` — magnitudes; the pair a cross-variable index
      is derived from.
    - ``flag`` / ``peer_flag`` — 0/1 indicators, and ``share`` — a 0..1
      proportion. Deliberately different in character from ``level``, which is
      what per-variable recipes are for.
    - ``target (date, product)`` — the quantity to model, at product grain.
    """
    rng = np.random.default_rng(97)
    n = 90
    n_products = 4
    level = 10 + rng.gamma(4.0, 3.0, (n, n_products))
    peer_level = 11 + rng.gamma(4.0, 3.0, (n, n_products))
    flag = (rng.random((n, n_products)) < 0.25).astype(float)
    peer_flag = (rng.random((n, n_products)) < 0.2).astype(float)
    share = rng.uniform(0.4, 1.0, (n, n_products))
    return xr.Dataset(
        {
            "media_national": (("date", "channel"), np.abs(rng.gamma(2.0, 400.0, (n, 2)))),
            "media_digital": (
                ("date", "product", "digital"),
                np.abs(rng.gamma(2.0, 30.0, (n, n_products, 2))),
            ),
            "level": (("date", "product"), level),
            "flag": (("date", "product"), flag),
            "share": (("date", "product"), share),
            "peer_level": (("date", "product"), peer_level),
            "peer_flag": (("date", "product"), peer_flag),
            "target": (
                ("date", "product"),
                50
                + 6 * flag
                + 3 * share
                + 20 * flag * peer_flag
                + rng.normal(0, 3, (n, n_products)),
            ),
        },
        coords={
            "date": _dates(n),
            "channel": ["tv", "ooh"],
            "digital": ["search", "social"],
            "product": [f"p{i}" for i in range(1, n_products + 1)],
        },
    )
