"""State persistence beyond netCDF: appending fitted state into an existing zarr store.

Native xarray IO covers the general case (``to_netcdf``/``open_dataset``);
``append_state`` exists only because ``DataTree.to_zarr(group=...)`` is not
implemented upstream — it writes per-node Datasets under one group, and the
reloaded state transforms identically to the in-memory one because dim-name
broadcasting is backend-agnostic.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from xscalers import (
    FixedScaler,
    MeanScaler,
    Pipeline,
    Splitter,
    VariableTransformer,
    append_state,
    load_media_channels,
    load_mixed_grain,
    load_national,
)
from xscalers.container import state_tree


class TestNetcdfNativeIO:
    def test_state_to_netcdf_open_dataset_roundtrip(self, tmp_path):
        """The general xarray case: state is a Dataset → plain to_netcdf/open_dataset."""
        ds = load_national()
        scaler = MeanScaler(dims=("date",))
        state = scaler.fit(ds["media"])
        path = tmp_path / "media_state.nc"
        state.to_netcdf(path)

        reloaded = xr.open_dataset(path)  # xr-native: no helper needed
        out = scaler.transform(ds["media"], reloaded)
        back = scaler.inverse_transform(out, reloaded)
        assert np.allclose(back.values, ds["media"].values)
        assert reloaded.attrs["kind"] == "mean"


class TestZarrAppend:
    def test_pipeline_state_appends_into_existing_store(self, tmp_path):
        """Fit on xarray, append into the store's 'scalers' group, read back natively."""
        ds = load_national()
        store = tmp_path / "mmm.zarr"
        ds.to_zarr(store)  # the "larger MMM input" store existing first

        pipe = Pipeline([("scale", VariableTransformer({"target": MeanScaler(dims=("date",))}))])
        state = pipe.fit(ds)
        append_state(pipe.to_state_tree(state), store, group="scalers")

        tree = xr.open_datatree(store, group="scalers")
        reloaded_state = tree.children["scale"].children["target"].dataset
        out = VariableTransformer({"target": MeanScaler(dims=("date",))}).transform(
            ds, {"target": reloaded_state}
        )
        assert np.allclose(out["target"].values, (ds["target"] / ds["target"].mean()).values)

    def test_single_scaler_state_appends_directly(self, tmp_path):
        """A bare Dataset state appends as one group; native open, native reload."""
        ds = load_national()
        store = tmp_path / "roundtrip.zarr"
        ds.to_zarr(store)

        scaler = MeanScaler(dims=("date",))
        state = scaler.fit(ds["media"])
        append_state(state, store, group="scalers/mean")

        reloaded = xr.open_zarr(store, group="scalers/mean")
        out = scaler.transform(ds["media"], reloaded)
        expected = ds["media"] / ds["media"].mean("date")
        assert np.allclose(out.values, expected.values)

    def test_refit_overrides_in_a_mode(self, tmp_path):
        """mode='a' with an existing group re-writes it (override, not duplicate)."""
        ds = load_national()
        store = tmp_path / "refit.zarr"
        ds.to_zarr(store)

        pipe = Pipeline([("scale", VariableTransformer({"target": MeanScaler(dims=("date",))}))])
        state = pipe.fit(ds)
        append_state(pipe.to_state_tree(state), store, group="scalers")
        append_state(pipe.to_state_tree(state), store, group="scalers")  # re-fit override

        tree = xr.open_datatree(store, group="scalers")
        assert set(tree.children) == {"scale"}

    def test_mixed_grain_per_variable_state(self, tmp_path):
        mixed = load_mixed_grain()
        store = tmp_path / "mixed.zarr"
        mixed.to_zarr(store)

        vt = VariableTransformer(
            {
                "media_national": MeanScaler(dims=("date",)),
                "media_digital": MeanScaler(dims=("date",)),
            },
            default="drop",
        )
        state = vt.fit(mixed)
        # VariableTransformer states are per-variable dicts; state_tree nests them
        # under their step name for persistence
        append_state(state_tree({"scale": state}), store, group="scalers")

        tree = xr.open_datatree(store, group="scalers")
        assert "media_national" in tree.children["scale"].children
        # flat state dict reconstructed from the reloaded tree — same transform
        reloaded = {
            name: node.dataset
            for name, node in tree.children["scale"].children.items()
        }
        out = vt.transform(mixed, reloaded)
        nat = mixed["media_national"]
        assert np.allclose(out["media_national"].values, (nat / nat.mean("date")).values)


class TestSplitDataset:
    """Dataset inputs split into sibling variables (``{source}_{piece}``)."""

    def test_split_dataset_pieces_and_inverse(self):
        ds = load_media_channels()
        splitter = Splitter(
            "media",
            {
                "digital": {"channel": ["search", "social", "fb"]},
                "linear": {"channel": ["tv", "stream"]},
            },
        )
        state = splitter.fit(ds)
        split = splitter.transform(ds, state)

        # pieces fully cover the source coordinate — the source var is replaced,
        # and each piece keeps its coordinate under the piece-specific dim name
        assert set(split.data_vars) == {"media_digital", "media_linear", "controls"}
        assert set(split["media_digital"].coords["channel_digital"].values) == {
            "search", "social", "fb",
        }

        back = splitter.inverse_transform(split, state)
        assert list(back["media"].channel.values) == list(ds.channel.values)
        assert np.allclose(back["media"].values, ds["media"].values)

    def test_split_pipeline_on_dataset_roundtrip(self):
        """The realistic shape: Dataset input, per-piece scaling, full inverse."""
        ds = load_media_channels()
        pipe = Pipeline(
            [
                (
                    "split",
                    Splitter(
                        "media",
                        {
                            "digital": {"channel": ["search", "social", "fb"]},
                            "linear": {"channel": ["tv", "stream"]},
                        },
                    ),
                ),
                (
                    "scale",
                    VariableTransformer(
                        {
                            "media_digital": MeanScaler(dims=("date",)),
                            "media_linear": FixedScaler(scale=1_000),
                        }
                    ),
                ),
            ]
        )
        state = pipe.fit(ds)
        scaled = pipe.transform(ds, state)
        assert set(scaled.data_vars) == {"media_digital", "media_linear", "controls"}

        # each piece scaled by its own recipe
        media = ds["media"]
        digital_mean = media.sel(channel=["search", "social", "fb"]).mean("date")
        assert np.allclose(
            scaled["media_digital"].values,
            (media.sel(channel=["search", "social", "fb"]).values / digital_mean.values),
        )

        back = pipe.inverse_transform(scaled, state)
        assert np.allclose(back["media"].values, ds["media"].values)
