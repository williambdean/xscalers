"""preprocessing data-movement steps: Splitter (invertible), Selector (honestly
non-invertible), Derive (stateless, cross-grain capable)."""

from __future__ import annotations

import numpy as np
import pytest

from xscalers.compose import PerVariable
from xscalers.core import NonInvertibleError, from_spec
from xscalers.pipeline import Pipeline
from xscalers.preprocessing import Derive, FixedScaler, Func, MeanScaler, Selector, Splitter, ZScore

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
                PerVariable(
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

    def test_pipeline_affine_raises_because_of_split(self, split_tree):
        pipe = _pipe()
        state = pipe.fit(split_tree)
        with pytest.raises(TypeError):
            pipe.affine(state)


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


class TestDerive:
    """Derived metrics from coordinate values within one variable."""

    def _deriver(self):
        return Derive(
            name="level_ratio",
            numerator=("controls", {"control": "level"}),
            denominator=("controls", {"control": "reference"}),
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
            numerator=("controls", {"control": ["level", "reference"]}),
            denominator=("controls", {"control": "reference"}),
            op="ratio",
            reduce="sum",
        )
        state = near.fit(ds)
        out = near.transform(ds, state)
        expected = ds["controls"].sel(control=["level", "reference"]).sum(
            "control"
        ) / ds["controls"].sel(control="reference")
        assert out["level_share"].dims == ("date", "product")
        assert np.allclose(out["level_share"].values, expected.values)

    def test_scalar_coordinate_selection(self, mixed_grain_ds):
        ds = mixed_grain_ds
        derive = Derive(
            name="level_only",
            numerator=("controls", {"control": "level"}),
            denominator=("controls", {"control": "reference"}),
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
                ("scale", PerVariable({"level_ratio": ZScore(dims=("date",))})),
            ]
        )
        state = pipe.fit(mixed_grain_ds)
        scaled = pipe.transform(mixed_grain_ds, state)
        manual = mixed_grain_ds["controls"].sel(control="level") / mixed_grain_ds["controls"].sel(
            control="reference"
        )
        manual_scaled = (
            manual - manual.mean("date")
        ) / manual.std("date", ddof=0)
        assert np.allclose(scaled["level_ratio"].values, manual_scaled.values)
        back = pipe.inverse_transform(scaled, state)
        for var in mixed_grain_ds.data_vars:
            assert np.allclose(back[var].values, mixed_grain_ds[var].values)
        assert "level_ratio" not in back.data_vars

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
                        numerator=("controls", {"control": "level"}),
                        denominator=("controls", {"control": "reference"}),
                        op="ratio",
                    ),
                ),
                (
                    "derive_level_seasonality",
                    Derive(
                        name="level_vs_seasonality",
                        numerator=("controls", {"control": "level"}),
                        denominator=("controls", {"control": "seasonality"}),
                        op="ratio",
                    ),
                ),
                ("log", Func({"level_vs_reference": "log", "level_vs_seasonality": "log"})),
                (
                    "scale",
                    PerVariable(
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
            assert np.allclose(
                scaled[name].std(("date", "product"), ddof=0).values, 1.0, rtol=1e-6
            )

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
