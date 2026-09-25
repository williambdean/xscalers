"""Reproducible example datasets — sklearn.datasets-style helpers for the README."""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest
import xarray as xr

from xscalers.datasets import load_media_channels, load_mixed_grain, load_national


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
            errors = [l for l in result.stdout.splitlines() if "error" in l]
            assert not errors, f"README block group {i} does not type-check:\n" + "\n".join(errors)
