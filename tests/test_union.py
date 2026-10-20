"""Union: many pipelines over one dataset, merged back into a single container."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xscalers import Func, MeanScaler, ZScore
from xscalers.compose import VariableTransformer
from xscalers.container import state_from_tree, state_tree
from xscalers.core import NonInvertibleError
from xscalers.pipeline import Pipeline, make_pipeline, make_union
from xscalers.preprocessing import Derive, Dropper

DIGITAL = ["search", "social", "fb"]
LINEAR = ["tv", "stream"]


def _branch_digital():
    return make_pipeline(
        Derive(
            name="level_vs_reference",
            left=("controls", {"control": "level"}),
            right=("controls", {"control": "reference"}),
            op="ratio",
        ),
        Func({"level_vs_reference": "log"}),
        VariableTransformer({"level_vs_reference": ZScore(dims=("date",))}),
    )


def _branch_seasonality():
    return make_pipeline(
        Derive(
            name="level_vs_seasonality",
            left=("controls", {"control": "level"}),
            right=("controls", {"control": "seasonality"}),
            op="ratio",
        ),
        Func({"level_vs_seasonality": "log"}),
        VariableTransformer({"level_vs_seasonality": ZScore(dims=("date",))}),
    )


def _linear_reference():
    """The one-pipeline form of the same two-derivation scenario."""
    return Pipeline(
        [
            (
                "rel_comp",
                Derive(
                    name="level_vs_reference",
                    left=("controls", {"control": "level"}),
                    right=("controls", {"control": "reference"}),
                    op="ratio",
                ),
            ),
            (
                "rel_infl",
                Derive(
                    name="level_vs_seasonality",
                    left=("controls", {"control": "level"}),
                    right=("controls", {"control": "seasonality"}),
                    op="ratio",
                ),
            ),
            ("log", Func({"level_vs_reference": "log", "level_vs_seasonality": "log"})),
            (
                "scale",
                VariableTransformer(
                    {
                        "level_vs_reference": ZScore(dims=("date",)),
                        "level_vs_seasonality": ZScore(dims=("date",)),
                    }
                ),
            ),
        ]
    )


class TestMerge:
    def test_union_parity_with_linear_pipeline(self, mixed_grain_ds):
        """The union form and the linear chain produce identical scaled variables."""
        union = make_union(_branch_digital(), _branch_seasonality())
        state = union.fit(mixed_grain_ds)
        out_union = union.transform(mixed_grain_ds, state)

        pipe = _linear_reference()
        linear_state = pipe.fit(mixed_grain_ds)
        out_linear = pipe.transform(mixed_grain_ds, linear_state)

        for var in ("level_vs_reference", "level_vs_seasonality"):
            assert np.allclose(out_union[var].values, out_linear[var].values)
        # passthrough inputs survive
        assert np.allclose(out_union["media_national"].values, mixed_grain_ds["media_national"].values)

    def test_derived_vars_present_and_inputs_unscaled(self, mixed_grain_ds):
        union = make_union(_branch_digital(), _branch_seasonality())
        state = union.fit(mixed_grain_ds)
        out = union.transform(mixed_grain_ds, state)
        assert "level_vs_reference" in out.data_vars
        assert "level_vs_seasonality" in out.data_vars
        # media_tv scaled? both branches pass it through untouched (no scaling branch owns it)
        assert np.allclose(out["media_tv"].values, mixed_grain_ds["media_tv"].values)


class TestCollisions:
    def test_conflicting_scales_raise(self, national_ds):
        union = make_union(
            ("a", VariableTransformer({"target": ZScore(dims=("date",))})),
            ("b", VariableTransformer({"target": MeanScaler(dims=("date",))})),
        )
        state = union.fit(national_ds)
        with pytest.raises(ValueError, match="target"):
            union.transform(national_ds, state)

    def test_identical_branches_merge_silently(self, national_ds):
        union = make_union(
            ("a", VariableTransformer({"target": ZScore(dims=("date",))})),
            ("b", VariableTransformer({"target": ZScore(dims=("date",))})),
        )
        state = union.fit(national_ds)
        out = union.transform(national_ds, state)
        expected = (national_ds["target"] - national_ds["target"].mean("date")) / (
            national_ds["target"].std("date", ddof=0)
        )
        assert np.allclose(out["target"].values, expected.values)


class TestInverse:
    def test_union_inverse_recovers_inputs(self, mixed_grain_ds):
        union = make_union(_branch_digital(), _branch_seasonality())
        state = union.fit(mixed_grain_ds)
        back = union.inverse_transform(union.transform(mixed_grain_ds, state), state)
        for var in mixed_grain_ds.data_vars:
            assert np.allclose(back[var].values, mixed_grain_ds[var].values)
        for name in ("level_vs_reference", "level_vs_seasonality"):
            assert name not in back.data_vars

    def test_dropping_branch_refuses_inverse(self, mixed_grain_ds):
        """A branch that drops a variable must refuse the whole union
        inverse — never a silently half-restored result."""
        union = make_union(
            _branch_digital(),
            ("prune", Dropper(vars=["media_national"])),
        )
        state = union.fit(mixed_grain_ds)
        with pytest.raises(NonInvertibleError):
            union.inverse_transform(union.transform(mixed_grain_ds, state), state)


class TestAffine:
    def test_affine_refusal_names_the_union(self, mixed_grain_ds):
        union = make_union(_branch_digital(), _branch_seasonality())
        pipe = Pipeline(
            [("union", union), ("scale", VariableTransformer({"controls_nat": ZScore(dims=("date",))}))]
        )
        state = pipe.fit(mixed_grain_ds)
        with pytest.raises(TypeError, match="Union"):
            pipe.affine(state)


class TestStatePersistence:
    def test_state_tree_netcdf_roundtrip(self, mixed_grain_ds, tmp_path):
        union = make_union(_branch_digital(), _branch_seasonality())
        state = union.fit(mixed_grain_ds)
        path = tmp_path / "union_state.nc"
        state_tree(state).to_netcdf(path)

        loaded = xr.open_datatree(path)
        reloaded = state_from_tree(loaded)  # nested dicts like fit returned
        out = union.transform(mixed_grain_ds, state)
        out_reloaded = union.transform(mixed_grain_ds, reloaded)
        for var in ("level_vs_reference", "level_vs_seasonality"):
            assert np.allclose(out[var].values, out_reloaded[var].values)


class TestMakeUnionNaming:
    def test_auto_names_from_transformer_classes(self):
        u = make_union(ZScore(dims=("date",)), ZScore(dims=("date",)))
        assert [name for name, _ in u.branches] == ["zscore", "zscore-1"]

    def test_pipeline_branch_names_kept(self):
        branch = make_pipeline(ZScore(dims=("date",)))
        u = make_union(("explicit", branch))
        assert [name for name, _ in u.branches] == ["explicit"]
