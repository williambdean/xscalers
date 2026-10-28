"""RollingMean: a trailing, no-NaN rolling mean usable on out-of-sample data."""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from xscalers.core import NonInvertibleError, from_spec, to_spec
from xscalers.preprocessing import EwmSmooth, Loess, RollingMean


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


class TestEwmFit:
    def test_fit_records_alpha_in_state(self, national_ds):
        target = national_ds["target"]
        state = EwmSmooth(alpha=0.3, dims=("date",)).fit(target)
        assert state.attrs["kind"] == "ewm"
        assert state.attrs["alpha"] == pytest.approx(0.3)
        assert state.attrs["scaled_from"] == "target"

    def test_alpha_in_open_interval(self):
        with pytest.raises(ValueError, match="alpha"):
            EwmSmooth(alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            EwmSmooth(alpha=1.5)


class TestEwmTransform:
    def test_recursion_seeds_at_the_first_value(self):
        x = xr.DataArray(np.arange(1.0, 6.0), dims="date")
        smoother = EwmSmooth(alpha=0.5)
        state = smoother.fit(x)
        smoothed = smoother.transform(x, state)
        assert not smoothed.isnull().any()
        assert np.allclose(
            smoothed.values,
            [1.0, 1.5, 2.25, 3.125, 4.0625],  # s_t = 0.5*x_t + 0.5*s_{t-1}
        )

    def test_dataset_dispatch_smoothes_each_variable(self, national_ds):
        smoother = EwmSmooth(alpha=0.5, dims=("date",))
        states = smoother.fit(national_ds)
        out = smoother.transform(national_ds, states)
        for var in ("target", "media"):
            assert out[var].dims == national_ds[var].dims
            assert not out[var].isnull().any()

    def test_oos_slice_smoothes_from_state_alone(self, national_ds):
        target = national_ds["target"]
        smoother = EwmSmooth(alpha=0.5, dims=("date",))
        state = smoother.fit(target)
        tail = target.isel(date=slice(-12, None))
        out = smoother.transform(tail, state)
        assert not out.isnull().any()
        assert np.allclose(out.isel(date=0).values, tail.isel(date=0).values)

    def test_one_named_dim_only(self, national_ds):
        with pytest.raises(ValueError, match="one dim"):
            EwmSmooth().fit(national_ds["media"])  # (date, channel): ambiguous

    def test_symbolic_input_has_no_path(self, national_ds):
        pytest.importorskip("pytensor")
        import pytensor.xtensor as ptx

        symbol = ptx.xtensor("x", dims=("date",))
        smoother = EwmSmooth(alpha=0.5)
        with pytest.raises(NotImplementedError, match="xarray-only"):
            smoother.fit(symbol)
        state = smoother.fit(national_ds["target"])
        with pytest.raises(NotImplementedError, match="xarray-only"):
            smoother.transform(symbol, state)

    def test_inverse_is_honestly_non_invertible(self, national_ds):
        target = national_ds["target"]
        smoother = EwmSmooth(alpha=0.5, dims=("date",))
        state = smoother.fit(target)
        with pytest.raises(NonInvertibleError, match="EwmSmooth"):
            smoother.inverse_transform(smoother.transform(target, state), state)

    def test_spec_round_trips_and_refits_identically(self, national_ds):
        original = EwmSmooth(alpha=0.25, dims=("date",))
        spec = to_spec(original)
        assert spec == {
            "kind": "EwmSmooth",
            "kwargs": {"alpha": 0.25, "dims": ["date"]},
        }
        rebuilt = from_spec(json.loads(json.dumps(spec)))
        before = original.fit(national_ds["target"])
        after = rebuilt.fit(national_ds["target"])
        assert before.attrs["alpha"] == after.attrs["alpha"] == pytest.approx(0.25)

    def test_state_netcdf_roundtrip_keeps_alpha(self, national_ds, tmp_path):
        target = national_ds["target"]
        smoother = EwmSmooth(alpha=0.25, dims=("date",))
        path = tmp_path / "state.nc"
        smoother.fit(target).to_netcdf(path)
        out = smoother.transform(target, xr.open_dataset(path))
        expected = smoother.transform(target, smoother.fit(target))
        assert np.allclose(out.values, expected.values)

    def test_exported_from_package_root(self):
        import xscalers

        assert xscalers.EwmSmooth(alpha=0.9).alpha == pytest.approx(0.9)


class TestLoessFit:
    def test_fit_records_window_and_degree_in_state(self, national_ds):
        target = national_ds["target"]
        state = Loess(window=5, degree=1, dims=("date",)).fit(target)
        assert state.attrs["kind"] == "loess"
        assert state.attrs["window"] == 5
        assert state.attrs["degree"] == 1
        assert state.attrs["scaled_from"] == "target"

    def test_window_degrees_vocabulary(self):
        with pytest.raises(ValueError, match="window"):
            Loess(window=0)
        with pytest.raises(ValueError, match="degree"):
            Loess(degree=2)


class TestLoessTransform:
    def test_degree1_reproduces_an_exact_linear_series(self):
        x = xr.DataArray(np.linspace(0.0, 10.0, 25), dims="date")
        for window in (1, 3, 5):
            smoother = Loess(window=window, degree=1)
            out = smoother.transform(x, smoother.fit(x))
            assert np.allclose(out.values, x.values)

    def test_degree0_weighted_trailing_mean_shrinks_lead_in(self):
        # tricube over lags from the window end, renormalized per position
        def tricube_lags(window):
            distances = np.arange(window - 1, -1, -1)  # newest has distance 0
            u = distances / (window - 1) if window > 1 else distances * 0.0
            return (1 - u**3) ** 3  # weight 1.0 at distance 0, 0.0 at window-1

        x = xr.DataArray([1.0, 3.0, 2.0, 4.0], dims="date")
        window = 3
        smoother = Loess(window=window, degree=0)
        out = smoother.transform(x, smoother.fit(x))
        assert not out.isnull().any()
        w = tricube_lags(window)  # distance from the window end
        expected = []
        vals = x.values
        for t in range(4):
            pts = vals[max(t - window + 1, 0) : t + 1]  # trailing block
            wk = w[-len(pts) :]  # smallest distances sit at the tail
            expected.append(np.sum(wk * pts) / np.sum(wk))
        assert np.allclose(out.values, expected)

    def test_smooths_the_high_end_of_a_noisy_series(self):
        rng = np.random.default_rng(7)
        x = xr.DataArray(np.sin(np.arange(50) / 3) + rng.normal(0, 0.5, 50), dims="date")
        smoother = Loess(window=7, degree=1)
        out = smoother.transform(x, smoother.fit(x))
        assert not out.isnull().any()
        assert np.std(np.diff(out.values)) < np.std(np.diff(x.values))

    def test_oos_slice_smooths_from_state_alone(self, national_ds):
        target = national_ds["target"]
        smoother = Loess(window=5, degree=1, dims=("date",))
        state = smoother.fit(target)
        tail = target.isel(date=slice(-12, None))
        out = smoother.transform(tail, state)
        assert not out.isnull().any()
        assert np.allclose(out.isel(date=0).values, tail.isel(date=0).values)

    def test_one_named_dim_only(self, national_ds):
        with pytest.raises(ValueError, match="one dim"):
            Loess().fit(national_ds["media"])  # (date, channel): ambiguous

    def test_symbolic_input_has_no_path(self, national_ds):
        pytest.importorskip("pytensor")
        import pytensor.xtensor as ptx

        symbol = ptx.xtensor("x", dims=("date",))
        smoother = Loess(window=3)
        with pytest.raises(NotImplementedError, match="xarray-only"):
            smoother.fit(symbol)
        state = smoother.fit(national_ds["target"])
        with pytest.raises(NotImplementedError, match="xarray-only"):
            smoother.transform(symbol, state)

    def test_inverse_is_honestly_non_invertible(self, national_ds):
        target = national_ds["target"]
        smoother = Loess(window=5, degree=1, dims=("date",))
        state = smoother.fit(target)
        with pytest.raises(NonInvertibleError, match="Loess"):
            smoother.inverse_transform(smoother.transform(target, state), state)

    def test_spec_round_trips_and_refits_identically(self, national_ds):
        original = Loess(window=6, degree=1, dims=("date",))
        spec = to_spec(original)
        assert spec == {
            "kind": "Loess",
            "kwargs": {"window": 6, "degree": 1, "dims": ["date"]},
        }
        rebuilt = from_spec(json.loads(json.dumps(spec)))
        before = original.fit(national_ds["target"])
        after = rebuilt.fit(national_ds["target"])
        assert before.attrs["window"] == after.attrs["window"] == 6
        assert before.attrs["degree"] == after.attrs["degree"] == 1

    def test_state_netcdf_roundtrip_keeps_window_and_degree(self, national_ds, tmp_path):
        target = national_ds["target"]
        smoother = Loess(window=5, degree=1, dims=("date",))
        path = tmp_path / "state.nc"
        smoother.fit(target).to_netcdf(path)
        out = smoother.transform(target, xr.open_dataset(path))
        expected = smoother.transform(target, smoother.fit(target))
        assert np.allclose(out.values, expected.values)

    def test_exported_from_package_root(self):
        import xscalers

        assert xscalers.Loess(window=6, degree=0).degree == 0
