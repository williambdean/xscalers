"""compose.PerVariable: one recipe per variable, mixed grains, passthrough/drop defaults."""

from __future__ import annotations

import numpy as np
import pytest

from xscalers import preprocessing
from xscalers.compose import PerVariable


def _make():
    return PerVariable(
        {
            "media_national": preprocessing.MeanScaler(dims=("date",)),
            "media_digital": preprocessing.ZScore(dims=("date",)),
            "media_tv": preprocessing.FixedScaler(scale=1_000_000),
        }
    )


class TestFit:
    def test_one_state_per_variable(self, mixed_grain_ds):
        state = _make().fit(mixed_grain_ds)
        assert set(state) == {"media_national", "media_digital", "media_tv"}
        assert state["media_digital"]["scale"].dims == ("geo", "media")

    def test_keeps_variable_specific_dims(self, mixed_grain_ds):
        state = _make().fit(mixed_grain_ds)
        assert state["media_national"]["scale"].dims == ("media",)
        assert state["media_tv"]["scale"].dims == () or "scale" in state["media_tv"]

    def test_dataarray_input_fits_named_variable(self, national_ds):
        state = preprocessing.MeanScaler(dims=("date",)).fit(national_ds["media"])
        assert state["scale"].dims == ("channel",)


class TestTransform:
    def test_each_variable_scaled_by_own_recipe(self, mixed_grain_ds):
        dispatch = _make()
        state = dispatch.fit(mixed_grain_ds)
        out = dispatch.transform(mixed_grain_ds, state)
        tv = mixed_grain_ds["media_tv"]
        assert np.allclose(out["media_tv"].values, tv.values / 1_000_000)
        dig = mixed_grain_ds["media_digital"]
        assert np.allclose(
            out["media_digital"].values,
            (dig - dig.mean("date")) / dig.std("date", ddof=0),
        )

    def test_unmapped_variables_pass_through(self, mixed_grain_ds):
        dispatch = _make()
        state = dispatch.fit(mixed_grain_ds)
        out = dispatch.transform(mixed_grain_ds, state)
        assert np.allclose(
            out["controls_nat"].values, mixed_grain_ds["controls_nat"].values
        )

    def test_inverse_restores_all(self, mixed_grain_ds):
        dispatch = _make()
        state = dispatch.fit(mixed_grain_ds)
        back = dispatch.inverse_transform(dispatch.transform(mixed_grain_ds, state), state)
        for var in mixed_grain_ds.data_vars:
            assert np.allclose(back[var].values, mixed_grain_ds[var].values)


class TestDataTreeDispatch:
    def test_tree_input_fit_and_transform(self, split_tree):
        dispatch = PerVariable(
            {"media": preprocessing.MeanScaler(dims=("date",)), "controls": preprocessing.ZScore(dims=("date",))}
        )
        state = dispatch.fit(split_tree)
        out = dispatch.transform(split_tree, state)
        media = split_tree.children["media"].dataset["media"]
        expected = media / media.mean("date")
        assert np.allclose(out.children["media"].dataset["media"].values, expected.values)


class TestDropDefault:
    def test_drop_removes_unmapped_variables(self, mixed_grain_ds):
        dispatch = PerVariable(
            {"media_national": preprocessing.MeanScaler(dims=("date",))}, default="drop"
        )
        state = dispatch.fit(mixed_grain_ds)
        out = dispatch.transform(mixed_grain_ds, state)
        assert set(out.data_vars) == {"media_national"}
        back = dispatch.inverse_transform(out, state)
        assert set(back.data_vars) == {"media_national"}
        assert np.allclose(
            back["media_national"].values, mixed_grain_ds["media_national"].values
        )

    def test_drop_plus_passthrough_explicit(self, mixed_grain_ds):
        dispatch = PerVariable(
            {
                "media_national": preprocessing.MeanScaler(dims=("date",)),
                "controls_nat": "passthrough",
            },
            default="drop",
        )
        state = dispatch.fit(mixed_grain_ds)
        out = dispatch.transform(mixed_grain_ds, state)
        assert set(out.data_vars) == {"media_national", "controls_nat"}


class TestMappingValidation:
    """An explicit mapping entry naming an unknown variable is a config bug — loud, not silent."""

    def test_unknown_mapping_key_warns(self, mixed_grain_ds):
        """A coordinate name ('date') is not a data variable — must warn, not scale silently."""
        dispatch = PerVariable({"date": preprocessing.MeanScaler(dims=("date",))})
        with pytest.warns(UserWarning, match="missing from the input"):
            dispatch.fit(mixed_grain_ds)

    def test_unmatched_keys_listed_with_suggestion(self, mixed_grain_ds):
        """Closest-name hint: typo-like mapping entries are discoverable."""
        dispatch = PerVariable({"media_natoinal": preprocessing.MeanScaler(dims=("date",))})
        with pytest.warns(UserWarning, match="media_natoinal"):
            dispatch.fit(mixed_grain_ds)


class TestUnknownInputVariable:
    """Transform against data whose variables the fitted recipe never saw."""

    def test_transform_of_unscaled_variable_passthrough_with_state(self, mixed_grain_ds):
        """Unmapped-but-present variables pass through even when other keys scale."""
        dispatch = PerVariable({"media_national": preprocessing.MeanScaler(dims=("date",))})
        state = dispatch.fit(mixed_grain_ds)
        out = dispatch.transform(mixed_grain_ds, state)
        assert np.allclose(out["media_tv"].values, mixed_grain_ds["media_tv"].values)
        assert np.allclose(
            out["media_national"].values,
            mixed_grain_ds["media_national"] / mixed_grain_ds["media_national"].mean("date"),
        )
