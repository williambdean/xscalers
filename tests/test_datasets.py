"""Reproducible example datasets — sklearn.datasets-style helpers for the README."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xscalers.datasets import load_channel_tree, load_mixed_grain, load_national


class TestLoadNational:
    def test_uncertain_shape_and_names(self):
        ds = load_national()
        assert "target" in ds and "media" in ds and "controls" in ds
        assert ds["media"].dims == ("date", "channel")

    def test_deterministic(self):
        assert load_national().identical(load_national())

    def test_dates_are_real_datetime(self):
        ds = load_national()
        assert np.issubdtype(ds.date.dtype, np.datetime64)


class TestLoadMixedGrain:
    def test_heterogeneous_grains(self):
        ds = load_mixed_grain()
        assert ds["media_national"].dims == ("date", "media")
        assert ds["media_digital"].dims == ("date", "geo", "media")
        assert ds["controls_prod"].dims == ("date", "product")
        assert ds.identical(load_mixed_grain())


class TestLoadChannelTree:
    def test_is_a_datatree_with_media_and_controls(self):
        tree = load_channel_tree()
        assert isinstance(tree, xr.DataTree)
        assert list(tree.children) == ["media", "controls"]

    def test_deterministic(self):
        assert load_channel_tree().equals(load_channel_tree())


class TestReadmeExecutes:
    """The README's python blocks run end-to-end with just `pip install`."""

    @pytest.fixture
    def readme_blocks(self):
        import pathlib
        import re

        readme = (pathlib.Path(__file__).parent.parent / "README.md").read_text()
        return re.findall(r"```python\n(.*?)```", readme, re.DOTALL)

    def test_blocks_exist(self, readme_blocks):
        assert len(readme_blocks) >= 5

    def test_blocks_run(self, readme_blocks, tmp_path, monkeypatch):
        # README writes fitted_state.nc to its cwd — keep that in tmp
        monkeypatch.chdir(tmp_path)
        ns: dict = {"__name__": "readme"}
        for block in readme_blocks:
            exec(compile(block, "<readme>", "exec"), ns)
