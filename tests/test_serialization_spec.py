"""Spec serialization: plain-dict recipes round-trip; state artifacts stay netCDF."""

from __future__ import annotations

import numpy as np
import xarray as xr

from xscalers import preprocessing
from xscalers.compose import PerVariable
from xscalers.core import from_spec, to_spec
from xscalers.pipeline import Pipeline


class TestScalerSpecs:
    def test_fixed_spec_is_jsonable(self):
        spec = to_spec(preprocessing.FixedScaler(scale=100.0, center=2.0))
        assert spec == {
            "kind": "FixedScaler",
            "kwargs": {"scale": 100.0, "center": 2.0},
        }

    def test_from_spec_refits_identically(self, national_ds):
        original = preprocessing.ZScore(dims=("date",))
        rebuilt = from_spec(to_spec(original))
        before = original.fit(national_ds["target"])
        after = rebuilt.fit(national_ds["target"])
        assert np.allclose(before["scale"].values, after["scale"].values)
        assert before.attrs == after.attrs or before.attrs["kind"] == after.attrs["kind"]


class TestSpecRegistry:
    def test_all_scalers_registered(self):
        from xscalers.core import REGISTRY

        for cls in (preprocessing.FixedScaler, preprocessing.ZScore, preprocessing.MeanScaler, preprocessing.Func):
            assert cls.__name__ in REGISTRY


class TestPipelineSpec:
    def test_pipeline_spec_rebuilds_and_refits_identically(self, national_ds):
        pipe = Pipeline(
            [
                ("scale", PerVariable({"target": preprocessing.ZScore(dims=("date",))})),
            ]
        )
        spec = pipe.to_spec()
        rebuilt = from_spec(spec)
        fitted = rebuilt.fit(national_ds)
        original_state = pipe.fit(national_ds)
        assert np.allclose(
            fitted["scale"]["target"]["scale"].values,
            original_state["scale"]["target"]["scale"].values,
        )


class TestStateNetcdf:
    def test_state_netcdf_roundtrip_preserves_attrs(self, national_ds, tmp_path):
        state = preprocessing.ZScore(dims=("date",)).fit(national_ds["target"])
        path = tmp_path / "state.nc"
        state.to_netcdf(path)
        loaded = xr.open_dataset(path)
        out = preprocessing.ZScore().transform(national_ds["target"], loaded)
        expected = (national_ds["target"] - national_ds["target"].mean("date")) / (
            national_ds["target"].std("date", ddof=0)
        )
        assert np.allclose(out.values, expected.values)
        assert loaded.attrs["kind"] == "zscore"
