"""Flagship: fit on xarray → scale/unscale pytensor.xtensor variables symbolically."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("pytensor")
import pytensor
import pytensor.xtensor as ptx

from xscalers import scalers


class TestLiftState:
    def test_dataarray_state_entry_lifts_with_dims(self, national_ds):
        from xscalers.binding import lift_state

        state = scalers.MeanScaler(dims=("date",)).fit(national_ds["media"])
        lifted = lift_state(state)
        assert lifted["scale"].dims == ("channel",)
        assert type(lifted["center"]).__name__ == "XTensorConstant"

    def test_lift_is_idempotent_for_symbolic_entries(self, national_ds):
        from xscalers.binding import lift_state

        state = scalers.MeanScaler(dims=("date",)).fit(national_ds["media"])
        symbol = ptx.xtensor("scale", dims=("channel",))
        lifted = lift_state({"center": state["center"], "scale": symbol})
        assert lifted["scale"] is symbol

    def test_dataarray_lift_strips_coords(self, national_ds):
        from xscalers.binding import lift_state

        state = scalers.MeanScaler(dims=("date",)).fit(national_ds["media"])
        lifted = lift_state(state)
        assert not hasattr(lifted["scale"], "coords")


class TestSymbolicTransformFromFittedState:
    def test_fit_mean_scaler_unscales_symbolic_contributions(self, national_ds):
        """The MMM pattern: fit on xarray, unscale in-graph contributions."""
        scaler = scalers.MeanScaler(dims=("date",))
        state = scaler.fit(national_ds["media"])

        contrib = ptx.xtensor("channel_contrib", dims=("date", "channel"))
        restored = scaler.inverse_transform(contrib, state)

        f = pytensor.function([contrib], restored.values)
        value = np.arange(6.0).reshape(2, 3)
        expected = value * national_ds["media"].mean("date").values
        assert np.allclose(f(value), expected)

    def test_symbolic_zscore_forward_matches_xarray(self, national_ds):
        scaler = scalers.ZScore(dims=("date",))
        da = national_ds["media"]
        state = scaler.fit(da)

        sym = ptx.xtensor("x", dims=("date", "channel"))
        f = pytensor.function([sym], scaler.transform(sym, state).values)
        result = f(da.values)

        expected = (da - da.mean("date")) / da.std("date", ddof=0)
        assert np.allclose(result, expected.values)

    def test_post_sampling_style_xarray_unscale_with_extra_dims(self, national_ds):
        """After sampling, contributions carry (chain, draw, ...) — state broadcasts over them."""
        scaler = scalers.MeanScaler(dims=("date",))
        da = national_ds["media"]
        state = scaler.fit(da)

        contrib = xr.DataArray(
            np.ones((2, 3, 4, 5)),  # chain, draw, date, channel
            dims=("chain", "draw", "date", "channel"),
        )
        out = scaler.inverse_transform(contrib, state)
        assert out.dims == ("chain", "draw", "date", "channel")
        assert np.allclose(out.values, contrib.values * da.mean("date").values)

    def test_symbolic_state_with_geo_kept_dims(self, mixed_grain_ds):
        scaler = scalers.ZScore(dims=("date",))
        da = mixed_grain_ds["media_digital"]  # (date, geo, media)
        state = scaler.fit(da)

        sym = ptx.xtensor("x", dims=("date", "geo", "media"))
        f = pytensor.function(
            [sym], scaler.inverse_transform(scaler.transform(sym, state), state).values
        )
        value = np.arange(24.0).reshape(4, 2, 3)
        assert np.allclose(f(value), value)
