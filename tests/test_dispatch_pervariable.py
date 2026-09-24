"""PerVariable dispatch: different scalers per variable, mixed grains, passthrough."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xscalers import scalers
from xscalers.dispatch import PerVariable


def _make():
    return PerVariable(
        {
            "media_national": scalers.MeanScaler(dims=("date",)),
            "media_digital": scalers.ZScore(dims=("date",)),
            "media_tv": scalers.FixedScaler(scale=1_000_000),
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
        state = scalers.MeanScaler(dims=("date",)).fit(national_ds["media"])
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
            {"media": scalers.MeanScaler(dims=("date",)), "controls": scalers.ZScore(dims=("date",))}
        )
        state = dispatch.fit(split_tree)
        out = dispatch.transform(split_tree, state)
        media = split_tree.children["media"].dataset["media"]
        expected = media / media.mean("date")
        assert np.allclose(out.children["media"].dataset["media"].values, expected.values)
