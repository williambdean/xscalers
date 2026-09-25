"""Union: many pipelines over one dataset, merged back into a single container."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xscalers import Func, MeanScaler, ZScore
from xscalers.container import state_from_tree, state_tree
from xscalers.dispatch import PerVariable
from xscalers.pipeline import Pipeline, make_pipeline, make_union
from xscalers.shape import Derive

DIGITAL = ["search", "social", "fb"]
LINEAR = ["tv", "stream"]


def _branch_digital():
    return make_pipeline(
        Derive(
            name="price_rel_comp",
            numerator=("controls", {"control": "price"}),
            denominator=("controls", {"control": "comp_price"}),
            op="ratio",
        ),
        Func({"price_rel_comp": "log"}),
        PerVariable({"price_rel_comp": ZScore(dims=("date",))}),
    )


def _branch_inflation():
    return make_pipeline(
        Derive(
            name="price_rel_infl",
            numerator=("controls", {"control": "price"}),
            denominator=("controls", {"control": "inflation"}),
            op="ratio",
        ),
        Func({"price_rel_infl": "log"}),
        PerVariable({"price_rel_infl": ZScore(dims=("date",))}),
    )


def _linear_reference():
    """The one-pipeline form of the same two-derivation scenario."""
    return Pipeline(
        [
            (
                "rel_comp",
                Derive(
                    name="price_rel_comp",
                    numerator=("controls", {"control": "price"}),
                    denominator=("controls", {"control": "comp_price"}),
                    op="ratio",
                ),
            ),
            (
                "rel_infl",
                Derive(
                    name="price_rel_infl",
                    numerator=("controls", {"control": "price"}),
                    denominator=("controls", {"control": "inflation"}),
                    op="ratio",
                ),
            ),
            ("log", Func({"price_rel_comp": "log", "price_rel_infl": "log"})),
            (
                "scale",
                PerVariable(
                    {
                        "price_rel_comp": ZScore(dims=("date",)),
                        "price_rel_infl": ZScore(dims=("date",)),
                    }
                ),
            ),
        ]
    )


class TestMerge:
    def test_union_parity_with_linear_pipeline(self, mixed_grain_ds):
        """The union form and the linear chain produce identical scaled variables."""
        union = make_union(_branch_digital(), _branch_inflation())
        state = union.fit(mixed_grain_ds)
        out_union = union.transform(mixed_grain_ds, state)

        pipe = _linear_reference()
        linear_state = pipe.fit(mixed_grain_ds)
        out_linear = pipe.transform(mixed_grain_ds, linear_state)

        for var in ("price_rel_comp", "price_rel_infl"):
            assert np.allclose(out_union[var].values, out_linear[var].values)
        # passthrough inputs survive
        assert np.allclose(out_union["media_national"].values, mixed_grain_ds["media_national"].values)

    def test_derived_vars_present_and_inputs_unscaled(self, mixed_grain_ds):
        union = make_union(_branch_digital(), _branch_inflation())
        state = union.fit(mixed_grain_ds)
        out = union.transform(mixed_grain_ds, state)
        assert "price_rel_comp" in out.data_vars
        assert "price_rel_infl" in out.data_vars
        # media_tv scaled? both branches pass it through untouched (no scaling branch owns it)
        assert np.allclose(out["media_tv"].values, mixed_grain_ds["media_tv"].values)


class TestCollisions:
    def test_conflicting_scales_raise(self, national_ds):
        union = make_union(
            ("a", PerVariable({"target": ZScore(dims=("date",))})),
            ("b", PerVariable({"target": MeanScaler(dims=("date",))})),
        )
        state = union.fit(national_ds)
        with pytest.raises(ValueError, match="target"):
            union.transform(national_ds, state)

    def test_identical_branches_merge_silently(self, national_ds):
        union = make_union(
            ("a", PerVariable({"target": ZScore(dims=("date",))})),
            ("b", PerVariable({"target": ZScore(dims=("date",))})),
        )
        state = union.fit(national_ds)
        out = union.transform(national_ds, state)
        expected = (national_ds["target"] - national_ds["target"].mean("date")) / (
            national_ds["target"].std("date", ddof=0)
        )
        assert np.allclose(out["target"].values, expected.values)


class TestInverse:
    def test_union_inverse_recovers_inputs(self, mixed_grain_ds):
        union = make_union(_branch_digital(), _branch_inflation())
        state = union.fit(mixed_grain_ds)
        back = union.inverse_transform(union.transform(mixed_grain_ds, state), state)
        for var in mixed_grain_ds.data_vars:
            assert np.allclose(back[var].values, mixed_grain_ds[var].values)
        for name in ("price_rel_comp", "price_rel_infl"):
            assert name not in back.data_vars


class TestAffine:
    def test_affine_refusal_names_the_union(self, mixed_grain_ds):
        union = make_union(_branch_digital(), _branch_inflation())
        pipe = Pipeline([("union", union), ("scale", PerVariable({"controls_nat": ZScore(dims=("date",))}))])
        state = pipe.fit(mixed_grain_ds)
        with pytest.raises(TypeError, match="Union"):
            pipe.affine(state)


class TestStatePersistence:
    def test_state_tree_netcdf_roundtrip(self, mixed_grain_ds, tmp_path):
        union = make_union(_branch_digital(), _branch_inflation())
        state = union.fit(mixed_grain_ds)
        path = tmp_path / "union_state.nc"
        state_tree(state).to_netcdf(path)

        loaded = xr.open_datatree(path)
        reloaded = state_from_tree(loaded)  # nested dicts like fit returned
        out = union.transform(mixed_grain_ds, state)
        out_reloaded = union.transform(mixed_grain_ds, reloaded)
        for var in ("price_rel_comp", "price_rel_infl"):
            assert np.allclose(out[var].values, out_reloaded[var].values)


class TestMakeUnionNaming:
    def test_auto_names_from_transformer_classes(self):
        u = make_union(ZScore(dims=("date",)), ZScore(dims=("date",)))
        assert [name for name, _ in u.branches] == ["zscore", "zscore-1"]

    def test_pipeline_branch_names_kept(self):
        branch = make_pipeline(ZScore(dims=("date",)))
        u = make_union(("explicit", branch))
        assert [name for name, _ in u.branches] == ["explicit"]
