"""preprocessing data-movement steps: Splitter (invertible), Selector and
Dropper (honestly non-invertible), Derive (stateless, cross-grain
capable)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from xscalers.compose import VariableTransformer
from xscalers.container import state_tree
from xscalers.core import NonInvertibleError, from_spec, to_spec
from xscalers.pipeline import Pipeline
from xscalers.preprocessing import (
    Derive,
    Dropper,
    FixedScaler,
    Func,
    MeanScaler,
    Selector,
    Splitter,
    ZScore,
)

DIGITAL = ["search", "social", "fb"]
TV = ["tv", "stream"]


def _splitter():
    return Splitter(
        source="media",
        pieces={"digital": {"channel": DIGITAL}, "tv": {"channel": TV}},
    )


def _pipe():
    return Pipeline(
        [
            ("split", _splitter()),
            (
                "scale",
                VariableTransformer(
                    {
                        "digital": MeanScaler(dims=("date",)),
                        "tv": FixedScaler(scale=1_000),
                    }
                ),
            ),
        ]
    )


class TestSplitter:
    def test_fit_records_source_coord_order(self, split_tree):
        state = _splitter().fit(split_tree)
        assert state.attrs["source"] == "media"
        assert state.attrs["split_dim"] == "channel"
        assert set(state.attrs["pieces"]) == {"digital", "tv"}
        # full original coordinate saved as a data var (netCDF-safe), not attrs
        assert list(state["labels_channel"].values) == ["search", "social", "fb", "tv", "stream"]

    def test_transform_replaces_source_with_piece_nodes(self, split_tree):
        tree = _splitter().transform(split_tree, _splitter().fit(split_tree))
        assert set(tree.children["media"].children) == {"digital", "tv"}
        piece = tree.children["media"].children["digital"].dataset["digital"]
        assert list(piece.channel.values) == DIGITAL

    def test_inverse_recombines_and_restores_coord_order(self, split_tree):
        splitter = _splitter()
        state = splitter.fit(split_tree)
        pieces = splitter.transform(split_tree, state)
        back = splitter.inverse_transform(pieces, state)
        media = back.children["media"].dataset["media"]
        assert list(media.channel.values) == ["search", "social", "fb", "tv", "stream"]
        assert np.allclose(
            media.values,
            split_tree.children["media"].dataset["media"].values,
        )

    def test_split_then_pipe_inverse_returns_original(self, split_tree):
        pipe = _pipe()
        state = pipe.fit(split_tree)
        out = pipe.transform(split_tree, state)
        back = pipe.inverse_transform(out, state)
        media = back.children["media"].dataset["media"]
        assert np.allclose(media.values, split_tree.children["media"].dataset["media"].values)

    def test_pipeline_affine_composes_through_split(self, split_tree):
        """The split is a value-preserving selection (affine identity), so
        the composed (center, scale) composes through it to each piece."""
        pipe = _pipe()
        state = pipe.fit(split_tree)
        composed = pipe.affine(state)
        # FixedScaler piece: the constant pair carries through the split
        piece = composed["media/tv"]
        assert np.allclose(float(piece["scale"]), 1_000.0)
        # MeanScaler piece: per-channel means are the scale
        piece = composed["media/digital"]
        assert np.allclose(
            piece["scale"].values, state["scale"]["media/digital"]["mean"].values
        )
        assert sorted(composed) == ["media/digital", "media/tv"]  # no junk keys

    def test_a_bare_label_string_means_one_label(self, split_tree):
        """The way Selector/Derive accept ``{"media": "tv"}``: a bare string
        must not be iterated per character."""
        splitter = Splitter("media", {"tv": {"channel": "tv"}})
        tree = splitter.transform(split_tree, splitter.fit(split_tree))
        assert splitter.pieces == {"tv": {"channel": ["tv"]}}
        assert list(tree.children["media"].children["tv"].dataset["tv"].channel.values) == ["tv"]

    def test_a_bare_label_string_survives_the_spec_round_trip(self):
        rebuilt = from_spec(to_spec(Splitter("media", {"tv": {"channel": "tv"}})))
        assert rebuilt.pieces == {"tv": {"channel": ["tv"]}}


class TestSelector:
    def test_select_within_a_coordinate(self, split_tree):
        selector = Selector({"media": {"channel": DIGITAL}})
        tree = selector.transform(split_tree, selector.fit(split_tree))
        media = tree.children["media"].dataset["media"]
        assert list(media.channel.values) == DIGITAL

    def test_inverse_raises(self, split_tree):
        selector = Selector({"media": {"channel": DIGITAL}})
        state = selector.fit(split_tree)
        with pytest.raises(NonInvertibleError):
            selector.inverse_transform(split_tree, state)


class TestDropper:
    """Dropper removes whole variables and/or dataset-wide coordinate
    values — sklego's ColumnDropper adapted to xarray; honestly
    non-invertible like Selector."""

    def test_drops_named_variables_and_leaves_the_rest_identical(self, mixed_grain_ds):
        ds = mixed_grain_ds
        dropper = Dropper(vars=["media_national"])
        out = dropper.transform(ds, dropper.fit(ds))
        assert "media_national" not in out.data_vars
        for name in set(ds.data_vars) - {"media_national"}:
            xr.testing.assert_identical(out[name], ds[name])

    def test_drops_coordinate_values_dataset_wide(self, mixed_grain_ds):
        ds = mixed_grain_ds
        dropper = Dropper(labels={"control": ["reference"]})
        out = dropper.transform(ds, dropper.fit(ds))
        assert "reference" not in out.coords["control"].values
        # the variable living on `control` shrinks with the coordinate
        assert out.sizes["control"] == ds.sizes["control"] - 1
        np.testing.assert_allclose(
            out["controls"].sel(product="p1", control="level").values,
            ds["controls"].sel(product="p1", control="level").values,
        )

    def test_accepts_a_bare_variable_name(self, national_ds):
        dropper = Dropper(vars="media")
        out = dropper.transform(national_ds, dropper.fit(national_ds))
        assert set(out.data_vars) == {"target"}

    def test_drops_variables_and_labels_together(self, mixed_grain_ds):
        ds = mixed_grain_ds
        dropper = Dropper(vars=["controls_prod"], labels={"control": ["reference"]})
        out = dropper.transform(ds, dropper.fit(ds))
        assert "controls_prod" not in out.data_vars
        assert "reference" not in out.coords["control"].values

    def test_inverse_raises(self, national_ds):
        dropper = Dropper(vars="media")
        state = dropper.fit(national_ds)
        with pytest.raises(NonInvertibleError):
            dropper.inverse_transform(national_ds, state)

    def test_unknown_variable_raises_at_fit(self, national_ds):
        with pytest.raises(KeyError):
            Dropper(vars=["nope"]).fit(national_ds)

    def test_unknown_label_raises_at_fit(self, national_ds):
        with pytest.raises(KeyError):
            Dropper(labels={"channel": ["tiktok"]}).fit(national_ds)

    def test_dropping_every_variable_raises_at_fit(self, national_ds):
        with pytest.raises(ValueError, match="empty"):
            Dropper(vars=["target", "media"]).fit(national_ds)

    def test_neither_vars_nor_labels_raises_at_construction(self):
        with pytest.raises(ValueError):
            Dropper()

    def test_non_dataset_input_raises(self, national_ds):
        da = national_ds["media"]
        with pytest.raises(TypeError):
            Dropper(vars="target").fit(da)

    def test_state_records_what_was_dropped(self, mixed_grain_ds):
        state = Dropper(vars=["media_national"], labels={"control": ["reference"]}).fit(mixed_grain_ds)
        assert state.attrs["kind"] == "drop"
        # JSON strings, so a pipeline state still writes to netcdf
        assert json.loads(state.attrs["dropped_vars"]) == ["media_national"]
        assert json.loads(state.attrs["dropped_labels"]) == {"control": ["reference"]}

    def test_pipeline_state_survives_netcdf(self, mixed_grain_ds, tmp_path):
        """A Dropper step's state must stay netCDF-writable — the recipe is
        in the spec, the attrs are JSON strings."""
        pipe = Pipeline([("prune", Dropper(vars=["media_national"], labels={"control": ["reference"]}))])
        tree = state_tree(pipe.fit(mixed_grain_ds))
        path = tmp_path / "dropper_state.nc"
        tree.to_netcdf(path)
        reopened = xr.open_datatree(path)
        assert json.loads(reopened["prune"].dataset.attrs["dropped_vars"]) == ["media_national"]
        assert json.loads(reopened["prune"].dataset.attrs["dropped_labels"]) == {"control": ["reference"]}

    def test_spec_round_trip(self, mixed_grain_ds):
        spec = to_spec(Dropper(vars=["media_national"], labels={"control": ["reference"]}))
        rebuilt = from_spec(spec)
        assert isinstance(rebuilt, Dropper)
        ds = mixed_grain_ds
        reference = Dropper(vars=["media_national"], labels={"control": ["reference"]})
        xr.testing.assert_identical(
            rebuilt.transform(ds, rebuilt.fit(ds)),
            reference.transform(ds, reference.fit(ds)),
        )

    def test_pipeline_derives_then_prunes_raw_inputs(self, mixed_grain_ds):
        """The motivating workflow: derive a ratio, then stop carrying the
        raw variables downstream."""
        deriver = Derive(
            name="tv_per_control",
            left=("media_national", {"media": "tv"}),
            right=("controls_prod", {}),
            op="ratio",
        )
        pipe = Pipeline(
            [
                ("derive", deriver),
                ("prune", Dropper(vars=["media_national", "controls_prod"])),
                ("scale", VariableTransformer({"tv_per_control": ZScore(dims=("date", "control"))})),
            ]
        )
        ds = mixed_grain_ds
        out = pipe.fit_transform(ds)
        assert set(out.data_vars) == {
            "media_digital",
            "media_tv",
            "controls_nat",
            "controls",
            "tv_per_control",
        }
        with pytest.raises(NonInvertibleError):
            pipe.inverse_transform(out, pipe.states_)
        np.testing.assert_allclose(
            pipe.unscale(out, var="tv_per_control")["tv_per_control"].values,
            deriver.transform(ds, deriver.fit(ds))["tv_per_control"].values,
        )

    def test_unscale_keeps_working_for_surviving_variables(self, national_ds):
        pipe = Pipeline(
            [
                ("drop", Dropper(vars="media")),
                ("scale", VariableTransformer({"target": ZScore(dims=("date",))})),
            ]
        )
        out = pipe.fit_transform(national_ds)
        np.testing.assert_allclose(
            pipe.unscale(out, var="target")["target"].values, national_ds["target"].values
        )


class TestDerive:
    """Derived metrics from coordinate values within one variable."""

    def _deriver(self):
        return Derive(
            name="level_ratio",
            left=("controls", {"control": "level"}),
            right=("controls", {"control": "reference"}),
            op="ratio",
        )

    def test_ratio_value_parity(self, mixed_grain_ds):
        ds = mixed_grain_ds
        state = self._deriver().fit(ds)
        derived = self._deriver().transform(ds, state)
        expected = ds["controls"].sel(control="level") / ds["controls"].sel(control="reference")
        assert "level_ratio" in derived.data_vars
        assert derived["level_ratio"].dims == ("date", "product")
        assert np.allclose(derived["level_ratio"].values, expected.values)

    def test_reduce_over_a_piece(self, mixed_grain_ds):
        """List selectors + reduce: sum over the piece (share-of-voice shape)."""
        ds = mixed_grain_ds
        near = Derive(
            name="level_share",
            left=("controls", {"control": ["level", "reference"]}),
            right=("controls", {"control": "reference"}),
            op="ratio",
            reduce="sum",
        )
        state = near.fit(ds)
        out = near.transform(ds, state)
        expected = ds["controls"].sel(control=["level", "reference"]).sum("control") / ds["controls"].sel(
            control="reference"
        )
        assert out["level_share"].dims == ("date", "product")
        assert np.allclose(out["level_share"].values, expected.values)

    def test_scalar_coordinate_selection(self, mixed_grain_ds):
        ds = mixed_grain_ds
        derive = Derive(
            name="level_only",
            left=("controls", {"control": "level"}),
            right=("controls", {"control": "reference"}),
            op="ratio",
        )
        state = derive.fit(ds)
        out = derive.transform(ds, state)
        assert out["level_only"].dims == ("date", "product")
        manual = ds["controls"].sel(control="level") / ds["controls"].sel(control="reference")
        assert np.allclose(out["level_only"].values, manual.values)

    def test_state_records_recipe_metadata_only(self, mixed_grain_ds):
        state = self._deriver().fit(mixed_grain_ds)
        assert state.attrs["kind"] == "derive"
        assert state.attrs["derived"] == "level_ratio"
        assert state.attrs["reduce"] == ""  # None → "" for netCDF-safety
        assert len(state.data_vars) == 0

    def test_inverse_drops_the_derived_variable_only(self, mixed_grain_ds):
        ds = mixed_grain_ds
        derive = self._deriver()
        state = derive.fit(ds)
        out = derive.transform(ds, state)
        back = derive.inverse_transform(out, state)
        assert "level_ratio" not in back.data_vars
        for var in ds.data_vars:
            assert np.allclose(back[var].values, ds[var].values)

    def test_spec_roundtrip(self, mixed_grain_ds):
        original = self._deriver()
        rebuilt = from_spec(original.to_spec())
        state = original.fit(mixed_grain_ds)
        back = rebuilt.transform(mixed_grain_ds, state)
        expected = mixed_grain_ds["controls"].sel(control="level") / mixed_grain_ds["controls"].sel(
            control="reference"
        )
        assert np.allclose(back["level_ratio"].values, expected.values)

    def test_pipeline_derive_then_scale_roundtrip(self, mixed_grain_ds):
        pipe = Pipeline(
            [
                ("derive", self._deriver()),
                ("scale", VariableTransformer({"level_ratio": ZScore(dims=("date",))})),
            ]
        )
        state = pipe.fit(mixed_grain_ds)
        scaled = pipe.transform(mixed_grain_ds, state)
        manual = mixed_grain_ds["controls"].sel(control="level") / mixed_grain_ds["controls"].sel(
            control="reference"
        )
        manual_scaled = (manual - manual.mean("date")) / manual.std("date", ddof=0)
        assert np.allclose(scaled["level_ratio"].values, manual_scaled.values)
        back = pipe.inverse_transform(scaled, state)
        for var in mixed_grain_ds.data_vars:
            assert np.allclose(back[var].values, mixed_grain_ds[var].values)
        assert "level_ratio" not in back.data_vars


class TestDeriveThenPrune:
    """The fit-pipeline handoff: derive a ratio across variables, then drop the
    input the ratio supersedes so the downstream frame has one column per
    concept. The prune is honest about being one-way, which is why scenarios
    keep a second pipeline that stops before the Dropper."""

    def _handoff(self):
        return Pipeline(
            [
                (
                    "ratio",
                    Derive(
                        name="level_index",
                        left=("peer_level", {}),
                        right=("level", {}),
                        op="ratio",
                    ),
                ),
                ("prune", Dropper(vars=["peer_level"])),
            ]
        )

    def _multidim(self):
        from xscalers.datasets import load_panel

        return load_panel()

    def test_model_input_carries_the_ratio_not_its_superseded_input(self):
        pipe = self._handoff()
        out = pipe.fit_transform(self._multidim())
        assert "level_index" in out.data_vars
        assert "peer_level" not in out.data_vars
        assert "level" in out.data_vars  # the denominator stays
        assert out["level_index"].dims == ("date", "product")

    def test_the_handoff_is_one_way(self):
        pipe = self._handoff()
        ds = self._multidim()
        state = pipe.fit(ds)
        with pytest.raises(NonInvertibleError):
            pipe.inverse_transform(pipe.transform(ds, state), state)

    def test_a_scenario_pipeline_stops_before_the_prune(self):
        """Same recipe, Dropper omitted: invertible, so it can drive what-ifs."""
        scenario = Pipeline([step for step in self._handoff().steps if step[0] != "prune"])
        ds = self._multidim()
        state = scenario.fit(ds)
        out = scenario.transform(ds, state)
        back = scenario.inverse_transform(out, state)
        assert "peer_level" in back.data_vars
        for name in ("level", "peer_level"):
            assert np.allclose(back[name].values, ds[name].values, equal_nan=True)


class TestSplitDerivePruneScaleHandoff:
    """The whole workflow in one pipeline: split a shared series into pieces,
    derive an index from two variables, drop the input it supersedes, and
    scale what remains. What comes out is the model-ready variable set, and
    the state tree has one leaf per piece and per surviving variable."""

    def _pipeline(self, *, prune=True):
        steps = [
            (
                "split",
                Splitter(
                    "media_national",
                    {"tv": {"channel": "tv"}, "ooh": {"channel": "ooh"}},
                ),
            ),
            (
                "index",
                Derive(
                    name="level_index",
                    left=("peer_level", {}),
                    right=("level", {}),
                    op="ratio",
                ),
            ),
            (
                "scale",
                VariableTransformer(
                    {
                        "media_national_tv": MeanScaler(dims=("date",)),
                        "media_national_ooh": FixedScaler(scale=100.0),
                        "level_index": MeanScaler(dims=("date",)),
                    }
                ),
            ),
        ]
        if prune:
            steps.insert(2, ("prune", Dropper(vars=["peer_level"])))
        return Pipeline(steps)

    def _raw(self):
        from xscalers.datasets import load_panel

        return load_panel()

    def test_output_is_the_model_ready_variable_set(self):
        pipe = self._pipeline()
        out = pipe.fit_transform(self._raw())
        assert set(out.data_vars) == {
            "media_national_tv",
            "media_national_ooh",
            "media_digital",
            "level",
            "flag",
            "share",
            "peer_flag",
            "level_index",
            "target",
        }
        assert "media_national" not in out.data_vars
        assert "peer_level" not in out.data_vars

    def test_pieces_are_scaled_by_their_own_recipes(self):
        raw = self._raw()
        out = self._pipeline().fit_transform(raw)
        # the source is (date, channel), so each piece keeps just the date
        assert out["media_national_tv"].dims == ("date", "channel_tv")
        tv = raw["media_national"].sel(channel="tv")
        assert np.allclose(
            out["media_national_tv"].squeeze("channel_tv").values,
            (tv / tv.mean("date")).values,
        )
        ooh = raw["media_national"].sel(channel="ooh")
        assert np.allclose(out["media_national_ooh"].squeeze("channel_ooh").values, (ooh / 100.0).values)

    def test_state_tree_has_one_leaf_per_piece_and_surviving_variable(self):
        from xscalers.container import state_tree

        pipe = self._pipeline()
        pipe.fit(self._raw())
        tree = state_tree(pipe.states_)
        assert set(tree.groups) >= {
            "/split",
            "/index",
            "/prune",
            "/scale",
            "/scale/media_national_tv",
            "/scale/media_national_ooh",
            "/scale/level_index",
        }
        # one leaf per piece, and the derived variable, each keeping its own
        # grain: a length-1 piece coord for the channel split, product grain
        # for the derived index
        assert tree["scale/media_national_tv"]["mean"].dims == ("channel_tv",)
        assert tree["scale/level_index"]["mean"].dims == ("product",)

    def test_the_pruned_pipeline_is_one_way(self):
        pipe = self._pipeline()
        raw = self._raw()
        state = pipe.fit(raw)
        with pytest.raises(NonInvertibleError):
            pipe.inverse_transform(pipe.transform(raw, state), state)

    def test_the_scenario_pipeline_inverts_back_to_the_derived_inputs(self):
        """No prune: the same recipes, and the round trip is honest."""
        pipe = self._pipeline(prune=False)
        raw = self._raw()
        out = pipe.fit_transform(raw)
        back = pipe.inverse_transform(out, pipe.states_)
        assert "peer_level" in back.data_vars
        # the inverse drops what Derive created and leaves its inputs alone
        assert "level_index" not in back.data_vars
        assert np.allclose(back["level"].values, raw["level"].values)
        assert np.allclose(back["peer_level"].values, raw["peer_level"].values)
        # Splitter is invertible, so the pieces recombine into the source,
        # with the original channel coord and order
        assert "media_national" in back.data_vars
        assert back["media_national"].dims == ("date", "channel")
        assert list(back["media_national"].coords["channel"].values) == [
            "tv",
            "ooh",
        ]
        assert np.allclose(back["media_national"].values, raw["media_national"].values)


class TestDeriveDogfood:
    """Two log-ratio derivations from the (date, product, control) controls:
    log(level / reference) and log(level / seasonality), then z-scored."""

    def _pipe(self):
        return Pipeline(
            [
                (
                    "derive_level_reference",
                    Derive(
                        name="level_vs_reference",
                        left=("controls", {"control": "level"}),
                        right=("controls", {"control": "reference"}),
                        op="ratio",
                    ),
                ),
                (
                    "derive_level_seasonality",
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

    def test_two_log_ratios_in_one_pipeline(self, mixed_grain_ds):
        pipe = self._pipe()
        state = pipe.fit(mixed_grain_ds)
        scaled = pipe.transform(mixed_grain_ds, state)

        controls = mixed_grain_ds["controls"]
        for name, denom in (("level_vs_reference", "reference"), ("level_vs_seasonality", "seasonality")):
            expected = np.log(controls.sel(control="level") / controls.sel(control=denom))
            expected_z = (expected - expected.mean("date")) / expected.std("date", ddof=0)
            assert scaled[name].dims == ("date", "product")
            assert np.allclose(scaled[name].values, expected_z.values)
            # z-score sanity: mean ~0, std ~1 over both kept dims
            assert np.allclose(scaled[name].mean(("date", "product")).values, 0.0, atol=1e-8)
            assert np.allclose(scaled[name].std(("date", "product"), ddof=0).values, 1.0, rtol=1e-6)

    def test_reverse_chain_recovers_inputs(self, mixed_grain_ds):
        pipe = self._pipe()
        state = pipe.fit(mixed_grain_ds)
        back = pipe.inverse_transform(pipe.transform(mixed_grain_ds, state), state)
        for var in mixed_grain_ds.data_vars:
            assert np.allclose(back[var].values, mixed_grain_ds[var].values)
        for name in ("level_vs_reference", "level_vs_seasonality"):
            assert name not in back.data_vars

    def test_affine_refuses_the_log_chain(self, mixed_grain_ds):
        pipe = self._pipe()
        state = pipe.fit(mixed_grain_ds)
        with pytest.raises(TypeError):
            pipe.affine(state)
