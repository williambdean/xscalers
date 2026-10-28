"""RollingMean: a trailing, no-NaN rolling mean usable on out-of-sample data."""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from xscalers.core import NonInvertibleError, from_spec, to_spec
from xscalers.preprocessing import RollingMean


class TestFit:
    def test_fit_records_window_in_state(self, national_ds):
        media = national_ds["media"]
        state = RollingMean(window=4, dims=("date",)).fit(media)
        assert state.attrs["kind"] == "rolling_mean"
        assert state.attrs["window"] == 4
        assert state.attrs["scaled_from"] == "media"


class TestTransform:
    def test_trailing_mean_shrinks_at_the_start_instead_of_nan(self):
        x = xr.DataArray(np.arange(1.0, 6.0), dims="date")
        smoother = RollingMean(window=3)
        state = smoother.fit(x)
        smoothed = smoother.transform(x, state)
        assert not smoothed.isnull().any()
        assert np.allclose(smoothed.values, [1.0, 1.5, 2.0, 3.0, 4.0])

    def test_dataset_dispatch_rolls_each_variable_over_date(self, national_ds):
        smoother = RollingMean(window=3, dims=("date",))
        states = smoother.fit(national_ds)
        out = smoother.transform(national_ds, states)
        expected = national_ds["media"].rolling({"date": 3}, min_periods=1).mean()
        assert np.allclose(out["media"].values, expected.values)
        # untouched grain: target rolled over date with no special casing
        expected_target = national_ds["target"].rolling({"date": 3}, min_periods=1).mean()
        assert np.allclose(out["target"].values, expected_target.values)

    def test_out_of_sample_slice_smoothes_from_state_alone(self, national_ds):
        media = national_ds["media"]
        smoother = RollingMean(window=4, dims=("date",))
        state = smoother.fit(media)
        tail = media.isel(date=slice(-12, None))
        out = smoother.transform(tail, state)
        assert not out.isnull().any()
        # lead-in burns in from the slice itself (documented: concat an
        # in-sample tail for boundary continuity), never NaN
        assert np.allclose(out.isel(date=0).values, tail.isel(date=0).values)


class TestInverse:
    def test_inverse_is_honestly_non_invertible(self, national_ds):
        media = national_ds["media"]
        smoother = RollingMean(window=3, dims=("date",))
        state = smoother.fit(media)
        with pytest.raises(NonInvertibleError, match="RollingMean"):
            smoother.inverse_transform(smoother.transform(media, state), state)


class TestPersistence:
    def test_spec_round_trips_and_refits_identically(self, national_ds):
        original = RollingMean(window=6, dims=("date",))
        spec = to_spec(original)
        assert spec == {
            "kind": "RollingMean",
            "kwargs": {"window": 6, "dims": ["date"]},
        }
        rebuilt = from_spec(json.loads(json.dumps(spec)))
        before = original.fit(national_ds["media"])
        after = rebuilt.fit(national_ds["media"])
        assert before.attrs["window"] == after.attrs["window"] == 6

    def test_state_netcdf_roundtrip_keeps_the_window(self, national_ds, tmp_path):
        media = national_ds["media"]
        smoother = RollingMean(window=3, dims=("date",))
        state = smoother.fit(media)
        path = tmp_path / "state.nc"
        state.to_netcdf(path)
        reopened = xr.open_dataset(path)
        out = smoother.transform(media, reopened)
        expected = media.rolling({"date": 3}, min_periods=1).mean()
        assert np.allclose(out.values, expected.values)

    def test_exported_from_package_root(self):
        import xscalers

        assert xscalers.RollingMean(window=2).window == 2
