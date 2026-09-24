"""DataTree container: normalize Dataset/dict/DataTree; canonical state tree; netCDF round-trip."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xscalers import scalers
from xscalers.container import flatten, state_tree, unflatten


class TestFlatten:
    def test_dataset_flattens_to_var_keys(self, mixed_grain_ds):
        items = flatten(mixed_grain_ds)
        assert set(items) == set(mixed_grain_ds.data_vars)

    def test_dataarray_flattens_to_its_name(self, national_ds):
        items = flatten(national_ds["target"])
        assert set(items) == {"target"}

    def test_datatree_flattens_node_paths(self, split_tree):
        items = flatten(split_tree)
        assert set(items) == {"media/media", "controls/controls"}


class TestUnflatten:
    def test_dataset_roundtrip(self, mixed_grain_ds):
        rebuilt = unflatten("Dataset", flatten(mixed_grain_ds))
        for var in mixed_grain_ds.data_vars:
            assert rebuilt[var].ndim == mixed_grain_ds[var].ndim

    def test_datatree_roundtrip(self, split_tree):
        rebuilt = unflatten("DataTree", flatten(split_tree))
        assert list(rebuilt.children) == ["controls", "media"]
        assert np.allclose(
            rebuilt.children["media"].dataset["media"].values,
            split_tree.children["media"].dataset["media"].values,
        )


class TestStateTree:
    def test_state_tree_builds_nesting_from_compound_keys(self, split_tree):
        from xscalers.dispatch import PerVariable

        dispatch = PerVariable(
            {"media": scalers.FixedScaler(scale=100.0), "controls": scalers.ZScore(dims=("date",))}
        )
        state = dispatch.fit(split_tree)
        tree = state_tree(state)
        assert "scale" in tree.children["media"].dataset

    def test_state_tree_netcdf_roundtrip(self, split_tree, tmp_path):
        from xscalers.dispatch import PerVariable

        dispatch = PerVariable(
            {"media": scalers.FixedScaler(scale=100.0), "controls": scalers.ZScore(dims=("date",))}
        )
        state = dispatch.fit(split_tree)
        path = tmp_path / "state.nc"
        state_tree(state).to_netcdf(path)
        loaded = xr.open_datatree(path)
        assert "media" in loaded.children
        assert float(loaded.children["media"].dataset["scale"].item()) == 100.0
