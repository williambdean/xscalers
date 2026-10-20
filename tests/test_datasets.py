"""Reproducible example datasets — sklearn.datasets-style helpers for the README."""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest
import xarray as xr

from xscalers.compose import VariableTransformer
from xscalers.datasets import (
    load_media_channels,
    load_mixed_grain,
    load_national,
    load_panel,
)
from xscalers.pipeline import Pipeline
from xscalers.preprocessing import MeanScaler, Splitter


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


class TestLoadMediaChannels:
    def test_is_a_single_dataset_with_variables(self):
        ds = load_media_channels()
        assert isinstance(ds, xr.Dataset)
        assert ds["media"].dims == ("date", "channel")
        assert ds["controls"].dims == ("date",)

    def test_deterministic(self):
        assert load_media_channels().identical(load_media_channels())


class TestLoadPanel:
    def test_inputs_at_two_grains(self):
        ds = load_panel()
        assert ds["media_national"].dims == ("date", "channel")
        assert ds["media_digital"].dims == ("date", "product", "digital")
        assert ds["target"].dims == ("date", "product")

    def test_the_rest_of_the_inputs_are_at_product_grain(self):
        ds = load_panel()
        for name in ("level", "flag", "share", "peer_level", "peer_flag"):
            assert ds[name].dims == ("date", "product")

    def test_flags_are_binary_and_share_is_a_proportion(self):
        ds = load_panel()
        for name in ("flag", "peer_flag"):
            assert set(np.unique(ds[name].values)) <= {0.0, 1.0}
        assert ds["share"].min() >= 0 and ds["share"].max() <= 1

    def test_the_two_levels_are_positive_and_ratioable(self):
        ds = load_panel()
        assert ds["level"].min() > 0 and ds["peer_level"].min() > 0
        assert (ds["peer_level"] / ds["level"]).dims == ("date", "product")

    def test_deterministic(self):
        assert load_panel().identical(load_panel())


class TestMixedGrainBroadcastOrder:
    """A national series reaches a product-grain model by broadcasting
    upstream. Scaling at native grain and scaling after the broadcast agree
    numerically -- only the fitted state's shape differs -- so prefer the
    native-grain order and get a state without a redundant product axis."""

    def _split(self, pieces):
        return Pipeline(
            [
                (
                    "split",
                    Splitter(
                        "media_national",
                        {p: {"channel": c} for p, c in pieces.items()},
                    ),
                )
            ]
        ).fit_transform

    def test_scaling_at_native_grain_matches_broadcast_then_scale(self):
        raw = load_panel()
        pieces = {"tv": "tv", "ooh": "ooh"}  # piece name -> channel label
        recipes = {f"media_national_{p_}": MeanScaler(dims=("date",)) for p_ in pieces}

        # native grain: split (date, channel) -> scale -> broadcast to product
        native = self._split(pieces)(raw)
        scaler = VariableTransformer(recipes)
        native = scaler.transform(native, scaler.fit(native))
        native = native.copy()
        for piece in pieces:  # only the national pieces need the product axis
            name = f"media_national_{piece}"
            native[name] = native[name].expand_dims({"product": raw["product"]})

        # broadcast first, then split and scale the (date, product) pieces
        broadcast = raw.copy()
        broadcast["media_national"] = (
            raw["media_national"]
            .expand_dims({"product": raw["product"]})
            .transpose("date", "product", "channel")
        )
        after_bare = self._split(pieces)(broadcast)
        after_scaler = VariableTransformer(recipes)
        after = after_scaler.transform(after_bare, after_scaler.fit(after_bare))

        for piece in pieces:
            name = f"media_national_{piece}"
            assert np.allclose(
                native[name].transpose("date", "product", f"channel_{piece}").values,
                after[name].values,
            ), name

    def test_native_grain_state_has_no_redundant_product_axis(self):
        raw = load_panel()
        native = self._split({"tv": "tv"})(raw)
        state = VariableTransformer({"media_national_tv": MeanScaler(dims=("date",))}).fit(native)
        assert "product" not in state["media_national_tv"]["mean"].dims

        broadcast = raw.copy()
        broadcast["media_national"] = raw["media_national"].expand_dims({"product": raw["product"]})
        after = VariableTransformer({"media_national_tv": MeanScaler(dims=("date",))}).fit(
            self._split({"tv": "tv"})(broadcast)
        )
        assert "product" in after["media_national_tv"]["mean"].dims


class TestReadmeExecutes:
    """The README's python blocks run end-to-end with just `pip install`."""

    @pytest.fixture
    def readme_blocks(self):
        import pathlib

        readme = (pathlib.Path(__file__).parent.parent / "README.md").read_text()
        return re.findall(r"```python\n(.*?)```", readme, re.DOTALL)

    def test_blocks_exist(self, readme_blocks):
        assert len(readme_blocks) >= 5

    def test_blocks_run(self, readme_blocks, tmp_path, monkeypatch):
        import importlib.util

        # README writes fitted_state.nc to its cwd — keep that in tmp
        monkeypatch.chdir(tmp_path)
        ns: dict = {"__name__": "readme"}
        for block in readme_blocks:
            first_line = block.strip().splitlines()[0]
            if first_line.startswith("# requires-"):
                module = first_line.removeprefix("# requires-").strip()
                assert module.isidentifier(), f"Bad requires marker: {first_line}"
                if importlib.util.find_spec(module) is None:
                    continue  # documented optional dependency; not installed here
            exec(compile(block, "<readme>", "exec"), ns)


class TestReadmeTypecheck:
    """Self-contained README python blocks type-check under mypy.

    Illustrates the README stays honest: annotations resolve, dataset state
    types are real, and the public surface is importable as typed. (# FIT)
    """

    @pytest.fixture
    def typed_groups(self) -> list[str]:
        readme = (pathlib.Path(__file__).parent.parent / "README.md").read_text()
        blocks = re.findall(r"```python\n(.*?)```", readme, re.DOTALL)
        grouped: list[str] = []
        buffer = ""
        for block in blocks:
            if block.strip().startswith("# requires-"):
                continue
            if "# continues above" in block and buffer:
                buffer += "\n\n" + block
                continue
            if buffer:
                grouped.append(buffer)
            buffer = block
        if buffer:
            grouped.append(buffer)
        return grouped

    def test_selfcontained_blocks_typecheck(self, typed_groups, tmp_path):
        import subprocess

        for i, block in enumerate(typed_groups):
            path = tmp_path / f"readme_block_{i}.py"
            path.write_text(block)
            result = subprocess.run(
                ["uv", "run", "mypy", str(path)],
                capture_output=True,
                text=True,
                check=False,  # exit code carries the diagnostics; asserts read stdout
            )
            errors = [line for line in result.stdout.splitlines() if "error" in line]
            assert not errors, f"README block group {i} does not type-check:\n" + "\n".join(errors)
