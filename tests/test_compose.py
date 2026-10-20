"""compose.VariableTransformer: one recipe per variable, mixed grains, passthrough/drop defaults."""

from __future__ import annotations

import numpy as np
import pytest

from xscalers import preprocessing
from xscalers.compose import VariableTransformer


def _make():
    return VariableTransformer(
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
        assert state["media_digital"]["std"].dims == ("geo", "media")

    def test_keeps_variable_specific_dims(self, mixed_grain_ds):
        state = _make().fit(mixed_grain_ds)
        assert state["media_national"]["mean"].dims == ("media",)
        assert state["media_tv"]["scale"].dims == ()

    def test_dataarray_input_fits_named_variable(self, national_ds):
        state = preprocessing.MeanScaler(dims=("date",)).fit(national_ds["media"])
        assert state["mean"].dims == ("channel",)


class TestSplitPiecesAtMixedGrains:
    """The 'treat this media differently' case: one mapping, pieces that came
    out of a Splitter, each with its own recipe and its own remaining grain."""

    def _pieces(self):
        from xscalers.compose import VariableTransformer as VT
        from xscalers.datasets import load_panel
        from xscalers.pipeline import Pipeline

        raw = load_panel()
        # national series is broadcast to product grain upstream, digital is
        # already product grain -- the two now differ only in their split dim
        raw = raw.copy()
        raw["media_national"] = (
            raw["media_national"]
            .expand_dims({"product": raw["product"]})
            .transpose("date", "product", "channel")
        )
        split = Pipeline(
            [
                (
                    "split",
                    preprocessing.Splitter(
                        "media_national",
                        {"tv": {"channel": "tv"}, "ooh": {"channel": "ooh"}},
                    ),
                ),
                (
                    "split",
                    preprocessing.Splitter(
                        "media_digital",
                        {"search": {"digital": "search"}, "social": {"digital": "social"}},
                    ),
                ),
            ]
        )
        return raw, split.fit_transform(raw), VT

    def test_each_piece_scaled_by_its_own_recipe(self):
        _, pieces, VT = self._pieces()
        vt = VT(
            {
                "media_national_tv": preprocessing.MeanScaler(dims=("date",)),
                "media_national_ooh": preprocessing.FixedScaler(scale=100.0),
                "media_digital_search": preprocessing.ZScore(dims=("date",)),
                "media_digital_social": preprocessing.ZScore(dims=("date",)),
            }
        )
        state = vt.fit(pieces)
        out = vt.transform(pieces, state)
        tv = pieces["media_national_tv"]
        assert np.allclose(out["media_national_tv"].values, (tv / tv.mean("date")).values)
        ooh = pieces["media_national_ooh"]
        assert np.allclose(out["media_national_ooh"].values, ooh.values / 100.0)
        dig = pieces["media_digital_search"]
        assert np.allclose(
            out["media_digital_search"].values,
            (dig - dig.mean("date")) / dig.std("date", ddof=0),
        )

    def test_each_piece_keeps_its_own_state_grain(self):
        _, pieces, VT = self._pieces()
        vt = VT(
            {
                "media_national_tv": preprocessing.MeanScaler(dims=("date",)),
                "media_digital_search": preprocessing.MeanScaler(dims=("date",)),
            }
        )
        state = vt.fit(pieces)
        # the tv piece came from a broadcast national series, so its per-date
        # scale is the same number repeated per product ...
        assert state["media_national_tv"]["mean"].dims == ("product", "channel_tv")
        # ... while the digital piece varies over date and product on its own
        assert state["media_digital_search"]["mean"].dims == (
            "product",
            "digital_search",
        )

    def test_inverse_restores_every_piece(self):
        _, pieces, VT = self._pieces()
        vt = VT(
            {
                name: preprocessing.MeanScaler(dims=("date",))
                for name in pieces.data_vars
                if name.startswith("media_")
            }
        )
        out = vt.transform(pieces, vt.fit(pieces))
        back = vt.inverse_transform(out, vt.fit(pieces))
        for name in pieces.data_vars:
            if name.startswith("media_"):
                assert np.allclose(back[name].values, pieces[name].values)


class TestTransform:
    def test_each_variable_scaled_by_own_recipe(self, mixed_grain_ds):
        vt = _make()
        state = vt.fit(mixed_grain_ds)
        out = vt.transform(mixed_grain_ds, state)
        tv = mixed_grain_ds["media_tv"]
        assert np.allclose(out["media_tv"].values, tv.values / 1_000_000)
        dig = mixed_grain_ds["media_digital"]
        assert np.allclose(
            out["media_digital"].values,
            (dig - dig.mean("date")) / dig.std("date", ddof=0),
        )

    def test_unmapped_variables_pass_through(self, mixed_grain_ds):
        vt = _make()
        state = vt.fit(mixed_grain_ds)
        out = vt.transform(mixed_grain_ds, state)
        assert np.allclose(out["controls_nat"].values, mixed_grain_ds["controls_nat"].values)

    def test_inverse_restores_all(self, mixed_grain_ds):
        vt = _make()
        state = vt.fit(mixed_grain_ds)
        back = vt.inverse_transform(vt.transform(mixed_grain_ds, state), state)
        for var in mixed_grain_ds.data_vars:
            assert np.allclose(back[var].values, mixed_grain_ds[var].values)


class TestAffineViewPerVariable:
    """``affine_view`` is what a downstream graph binds to, so it has to be
    per variable *and* keep each variable's own grain — a pair broadcast over
    the wrong dims would silently mis-scale one series."""

    def test_one_pair_per_mapped_variable(self, mixed_grain_ds):
        view = _make().affine_view(_make().fit(mixed_grain_ds))
        assert set(view) == {"media_national", "media_digital", "media_tv"}
        for pair in view.values():
            assert set(pair) == {"center", "scale"}

    def test_each_pair_keeps_its_own_remaining_dims(self, mixed_grain_ds):
        vt = _make()
        view = vt.affine_view(vt.fit(mixed_grain_ds))
        assert view["media_national"]["scale"].dims == ("media",)
        assert view["media_digital"]["scale"].dims == ("geo", "media")
        # FixedScaler reduces to nothing -- a constant pair
        assert view["media_tv"]["scale"].dims == ()

    def test_applying_the_pair_reproduces_the_transform(self, mixed_grain_ds):
        vt = _make()
        state = vt.fit(mixed_grain_ds)
        out = vt.transform(mixed_grain_ds, state)
        view = vt.affine_view(state)
        for name, pair in view.items():
            by_hand = (mixed_grain_ds[name] - pair["center"]) / pair["scale"]
            assert np.allclose(by_hand.transpose(*out[name].dims).values, out[name].values), name

    def test_non_affine_recipe_raises_type_error(self, national_ds):
        """A non-affine recipe inside this composite must surface the documented
        TypeError from Pipeline.affine's contract, not an AttributeError from
        the lookup itself (log-then-scale is the common real-world case)."""
        from xscalers.pipeline import Pipeline

        vt = VariableTransformer({"target": preprocessing.FunctionTransformer({"target": "log"})})
        state = vt.fit(national_ds)
        with pytest.raises(TypeError, match="FunctionTransformer"):
            vt.affine_view(state)

        pipe = Pipeline(
            [
                ("choose", vt),
                ("scale", VariableTransformer({"media": preprocessing.MeanScaler(dims=("date",))})),
            ]
        )
        pipe.fit(national_ds)
        with pytest.raises(TypeError, match="FunctionTransformer"):
            pipe.affine(pipe.states_)

    def test_pair_inverts_the_transform(self, mixed_grain_ds):
        vt = _make()
        state = vt.fit(mixed_grain_ds)
        out = vt.transform(mixed_grain_ds, state)
        view = vt.affine_view(state)
        for name, pair in view.items():
            back = out[name] * pair["scale"] + pair["center"]
            assert np.allclose(
                back.transpose(*mixed_grain_ds[name].dims).values, mixed_grain_ds[name].values
            ), name


class TestDataTreeDispatch:
    def test_tree_input_fit_and_transform(self, split_tree):
        vt = VariableTransformer(
            {
                "media": preprocessing.MeanScaler(dims=("date",)),
                "controls": preprocessing.ZScore(dims=("date",)),
            }
        )
        state = vt.fit(split_tree)
        out = vt.transform(split_tree, state)
        media = split_tree.children["media"].dataset["media"]
        expected = media / media.mean("date")
        assert np.allclose(out.children["media"].dataset["media"].values, expected.values)


class TestDropDefault:
    def test_drop_removes_unmapped_variables(self, mixed_grain_ds):
        vt = VariableTransformer({"media_national": preprocessing.MeanScaler(dims=("date",))}, default="drop")
        state = vt.fit(mixed_grain_ds)
        out = vt.transform(mixed_grain_ds, state)
        assert set(out.data_vars) == {"media_national"}
        back = vt.inverse_transform(out, state)
        assert set(back.data_vars) == {"media_national"}
        assert np.allclose(back["media_national"].values, mixed_grain_ds["media_national"].values)

    def test_drop_plus_passthrough_explicit(self, mixed_grain_ds):
        vt = VariableTransformer(
            {
                "media_national": preprocessing.MeanScaler(dims=("date",)),
                "controls_nat": "passthrough",
            },
            default="drop",
        )
        state = vt.fit(mixed_grain_ds)
        out = vt.transform(mixed_grain_ds, state)
        assert set(out.data_vars) == {"media_national", "controls_nat"}


class TestMappingValidation:
    """An explicit mapping entry naming an unknown variable is a config bug — loud, not silent."""

    def test_unknown_mapping_key_warns(self, mixed_grain_ds):
        """A coordinate name ('date') is not a data variable — must warn, not scale silently."""
        vt = VariableTransformer({"date": preprocessing.MeanScaler(dims=("date",))})
        with pytest.warns(UserWarning, match="missing from the input"):
            vt.fit(mixed_grain_ds)

    def test_unmatched_keys_listed_with_suggestion(self, mixed_grain_ds):
        """Closest-name hint: typo-like mapping entries are discoverable."""
        vt = VariableTransformer({"media_natoinal": preprocessing.MeanScaler(dims=("date",))})
        with pytest.warns(UserWarning, match="media_natoinal"):
            vt.fit(mixed_grain_ds)


class TestUnknownInputVariable:
    """Transform against data whose variables the fitted recipe never saw."""

    def test_transform_of_unscaled_variable_passthrough_with_state(self, mixed_grain_ds):
        """Unmapped-but-present variables pass through even when other keys scale."""
        vt = VariableTransformer({"media_national": preprocessing.MeanScaler(dims=("date",))})
        state = vt.fit(mixed_grain_ds)
        out = vt.transform(mixed_grain_ds, state)
        assert np.allclose(out["media_tv"].values, mixed_grain_ds["media_tv"].values)
        assert np.allclose(
            out["media_national"].values,
            mixed_grain_ds["media_national"] / mixed_grain_ds["media_national"].mean("date"),
        )
