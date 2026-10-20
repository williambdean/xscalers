"""Spec serialization: plain-dict recipes round-trip; state artifacts stay netCDF."""

from __future__ import annotations

import numpy as np
import xarray as xr

from xscalers import preprocessing
from xscalers.compose import VariableTransformer
from xscalers.core import from_spec, to_spec
from xscalers.pipeline import Pipeline, Union


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
        assert np.allclose(before["mean"].values, after["mean"].values)
        assert before.attrs == after.attrs or before.attrs["kind"] == after.attrs["kind"]


class TestSpecRegistry:
    def test_all_scalers_registered(self):
        from xscalers.core import REGISTRY

        for cls in (preprocessing.FixedScaler, preprocessing.ZScore, preprocessing.MeanScaler, preprocessing.Func):
            assert cls.__name__ in REGISTRY

    def test_data_movement_steps_registered(self):
        """Derive/Splitter/Selector/Dropper are serializable too — a spec is
        the recipe for the whole chain, not just its affine steps."""
        from xscalers.core import REGISTRY

        for cls in (
            preprocessing.Derive,
            preprocessing.Splitter,
            preprocessing.Selector,
            preprocessing.Dropper,
        ):
            assert cls.__name__ in REGISTRY


class TestPipelineSpec:
    def test_the_free_to_spec_matches_the_pipeline_method(self):
        pipe = Pipeline(
            [("scale", VariableTransformer({"target": preprocessing.ZScore(dims=("date",))}))]
        )
        assert to_spec(pipe) == pipe.to_spec()
        assert to_spec(pipe)["kind"] == "Pipeline"
        assert from_spec(to_spec(pipe)).to_spec() == pipe.to_spec()

    def test_union_spec_round_trips(self):
        pipe = Pipeline(
            [("scale", VariableTransformer({"target": preprocessing.ZScore(dims=("date",))}))]
        )
        union = Union([("a", pipe), ("b", pipe)])
        assert to_spec(union)["kind"] == "Union"
        assert from_spec(to_spec(union)).to_spec() == union.to_spec()

    def test_variable_transformer_spec_keeps_passthrough_strings(self):
        vt = VariableTransformer(
            {"price": preprocessing.MeanScaler(dims=("date",)), "promo": "passthrough"}
        )
        rebuilt = from_spec(to_spec(vt))
        assert rebuilt.mapping["promo"] == "passthrough"
        assert to_spec(rebuilt) == to_spec(vt)

    def test_pipeline_spec_rebuilds_and_refits_identically(self, national_ds):
        pipe = Pipeline(
            [
                ("scale", VariableTransformer({"target": preprocessing.ZScore(dims=("date",))})),
            ]
        )
        spec = pipe.to_spec()
        rebuilt = from_spec(spec)
        fitted = rebuilt.fit(national_ds)
        original_state = pipe.fit(national_ds)
        assert np.allclose(
            fitted["scale"]["target"]["std"].values,
            original_state["scale"]["target"]["std"].values,
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


class TestProvenanceStamps:
    """README-advertised inspection: every fitted state leaf answers "what has
    been done to this variable" via attrs — ``recipe`` (human one-liner) and
    ``spec_sha`` (fingerprint). These tests lock that contract down."""

    def test_pipeline_leaves_carry_recipe_stamps(self, national_ds):
        from xscalers.core import spec_sha

        pipe = Pipeline([
            ("scale", VariableTransformer({
                "target": preprocessing.ZScore(dims=("date",)),
                "media": preprocessing.MeanScaler(dims=("date",)),
            })),
        ])
        leaves = pipe.fit(national_ds)["scale"]
        assert leaves["target"].attrs["recipe"] == "ZScore(dims=['date'])"
        assert leaves["target"].attrs["spec_sha"] == spec_sha(
            preprocessing.ZScore(dims=("date",))
        )
        assert leaves["media"].attrs["recipe"] == "MeanScaler(dims=['date'])"

    def test_non_affine_op_stamp_names_the_function(self, national_ds):
        pipe = Pipeline([
            ("choose", VariableTransformer({
                "target": preprocessing.FunctionTransformer({"target": "log"}),
            })),
        ])
        leaf = pipe.fit(national_ds)["choose"]["target"]
        assert leaf.attrs["op"] == "log"
        assert leaf.attrs["recipe"] == "FunctionTransformer(mapping={'target': 'log'})"

    def test_stamps_survive_datatree_netcdf_roundtrip(self, national_ds, tmp_path):
        from xscalers.container import state_tree

        pipe = Pipeline([
            ("choose", VariableTransformer({
                "target": preprocessing.FunctionTransformer({"target": "log"}),
            })),
            ("scale", VariableTransformer({
                "target": preprocessing.ZScore(dims=("date",)),
                "media": preprocessing.MeanScaler(dims=("date",)),
            })),
        ])
        path = tmp_path / "state.nc"
        state_tree(pipe.fit(national_ds)).to_netcdf(path)
        loaded = xr.open_datatree(path)
        assert loaded["choose"]["target"].attrs["recipe"] == (
            "FunctionTransformer(mapping={'target': 'log'})"
        )
        assert loaded["scale"]["media"].attrs["spec_sha"]
        # replay with the reloaded state reproduces scaled values; the log op
        # lives in the earlier step's state, so both steps apply
        logged = preprocessing.FunctionTransformer({"target": "log"}).transform(
            national_ds["target"], loaded["choose"]["target"]
        )
        out = preprocessing.ZScore().transform(logged, loaded["scale"]["target"])
        lg = np.log(national_ds["target"])
        expected = (lg - lg.mean("date")) / lg.std("date", ddof=0)
        assert np.allclose(out.values, expected.values)
