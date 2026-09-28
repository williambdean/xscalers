"""Basics: FixedScaler, ZScore, MeanScaler — state semantics, mutation, sklearn parity."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xscalers.preprocessing import FixedScaler, Func, MeanScaler, ZScore


class TestFixedScaler:
    def test_no_fit_required_state_is_constant(self, national_ds):
        scaler = FixedScaler(scale=100.0)
        state = scaler.fit(national_ds["target"])
        assert float(state["scale"].item()) == 100.0
        assert float(state["center"].item()) == 0.0

    def test_divide_by_hundred_style_scaling(self, national_ds):
        scaler = FixedScaler(scale=100.0, center=0.0)
        state = scaler.fit(national_ds["target"])
        out = scaler.transform(national_ds["target"], state)
        assert np.allclose(out.values, national_ds["target"].values / 100.0)

    def test_inverse_restores(self, national_ds):
        scaler = FixedScaler(scale=1000.0, center=-2.0)
        state = scaler.fit(national_ds["target"])
        out = scaler.transform(national_ds["target"], state)
        back = scaler.inverse_transform(out, state)
        assert np.allclose(back.values, national_ds["target"].values)

    def test_matches_sklearn_maxabs_when_scaled_by_abs_max(self, national_ds):
        """FixedScaler(scale=max|x|) reproduces sklearn's MaxAbsScaler."""
        from sklearn.preprocessing import MaxAbsScaler

        da = national_ds["target"]
        sk_scaled = MaxAbsScaler().fit_transform(da.values.reshape(-1, 1)).ravel()

        state = FixedScaler(scale=float(np.abs(da).max()), center=0.0).fit(da)
        out = FixedScaler().transform(da, state)
        assert np.allclose(out.values, sk_scaled)


class TestZScore:
    def test_state_carries_mean_and_std(self, national_ds):
        da = national_ds["target"]
        state = ZScore(dims=("date",)).fit(da)
        assert set(state.data_vars) == {"mean", "std"}
        assert np.isclose(float(state["mean"].item()), da.mean("date").item())
        assert np.isclose(float(state["std"].item()), da.std("date", ddof=0).item())

    def test_sklearn_parity_flattened_standard_scaler(self, national_ds):
        from sklearn.preprocessing import StandardScaler

        da = national_ds["media"]  # (date, channel)
        sk = StandardScaler().fit_transform(da.values)  # treats rows as samples, cols as features
        state = ZScore(dims=("date",)).fit(da)
        ours = ZScore(dims=("date",)).transform(da, state)
        assert np.allclose(ours.values, sk)

    def test_none_dims_reduces_all(self, national_ds):
        da = national_ds["target"]
        state = ZScore().fit(da)
        assert float(state["mean"].item()) == pytest.approx(da.mean("date").item())
        assert float(state["std"].item()) == pytest.approx(da.std("date", ddof=0).item())

    def test_kept_dims_carry_coords(self, mixed_grain_ds):
        da = mixed_grain_ds["media_digital"]  # (date, geo, media)
        state = ZScore(dims=("date",)).fit(da)
        assert set(state["std"].dims) == {"geo", "media"}
        assert list(state["std"]["geo"].values) == list(mixed_grain_ds.geo.values)

    def test_nan_handling_matches_sklearn(self, national_ds):
        from sklearn.preprocessing import StandardScaler

        da = national_ds["media"].copy()
        da[5, 2] = np.nan
        da[15, 0] = np.nan
        sk = StandardScaler().fit_transform(da.values)
        state = ZScore(dims=("date",)).fit(da)
        ours = ZScore(dims=("date",)).transform(da, state)
        both_nan = np.isnan(ours.values) & np.isnan(np.nan_to_num(sk, nan=np.nan, posinf=np.inf))
        # same NaN pattern as sklearn, same values where finite
        assert np.array_equal(np.isnan(ours.values), np.isnan(sk))
        assert not both_nan.all()
        finite = ~np.isnan(sk)
        assert np.allclose(ours.values[finite], sk[finite])

    def test_inverse_roundtrip(self, national_ds):
        da = national_ds["target"]
        state = ZScore().fit(da)
        back = ZScore().inverse_transform(ZScore().transform(da, state), state)
        assert np.allclose(back.values, da.values)


class TestMeanScaler:
    def test_divide_by_mean_along_dims(self, national_ds):
        da = national_ds["media"]
        scaler = MeanScaler(dims=("date",))
        state = scaler.fit(da)
        scaled = scaler.transform(da, state)
        assert np.allclose(scaled.values, da.values / da.mean("date").values)

    def test_divisive_state_carries_mean_only(self, national_ds):
        state = MeanScaler(dims=("date",)).fit(national_ds["target"])
        assert set(state.data_vars) == {"mean"}  # no center stored

    def test_kept_per_channel_scales(self, national_ds):
        state = MeanScaler(dims=("date",)).fit(national_ds["media"])
        assert state["mean"].dims == ("channel",)


class TestMutation:
    def test_editing_state_changes_transform(self, national_ds):
        da = national_ds["target"]
        state = MeanScaler(dims=("date",)).fit(da)
        base = MeanScaler(dims=("date",)).transform(da, state)
        assert np.allclose(base.values, da.values / da.mean("date").values)
        state["mean"] = state["mean"] * 2
        after = MeanScaler(dims=("date",)).transform(da, state)
        assert np.allclose(after.values, (da.values / da.mean("date").values) / 2)

    def test_transposed_state_entry_broadcasts_by_dim_name(self, mixed_grain_ds):
        da = mixed_grain_ds["media_digital"]
        scaler = ZScore(dims=("date",))
        state = scaler.fit(da)
        # store scale with dims transposed — transform must broadcast by dim name
        state["std"] = state["std"].transpose("media", "geo")
        out = scaler.transform(da, state)
        expected = (da - da.mean("date")) / da.std("date", ddof=0)
        assert np.allclose(out.values, expected.values)


class TestDatasetInputDispatch:
    """Dataset inputs are handled by per-variable dispatch at the scaler level."""

    def test_dataset_input_gets_per_variable_states(self, mixed_grain_ds):
        scaler = MeanScaler(dims=("date",))
        state = scaler.fit(mixed_grain_ds)
        assert hasattr(state, "get")
        assert set(state.get("media_national")["mean"].dims) == {"media"}

    def test_dataset_input_transforms_each_variable(self, mixed_grain_ds):
        scaler = MeanScaler(dims=("date",))
        state = scaler.fit(mixed_grain_ds)
        out = scaler.transform(mixed_grain_ds, state)
        assert set(out.data_vars) == set(mixed_grain_ds.data_vars)
        assert np.allclose(
            out["media_national"].values,
            mixed_grain_ds["media_national"].values
            / mixed_grain_ds["media_national"].mean("date").values,
        )

    def test_dataset_input_inverse_roundtrip(self, mixed_grain_ds):
        scaler = MeanScaler(dims=("date",))
        state = scaler.fit(mixed_grain_ds)
        out = scaler.transform(mixed_grain_ds, state)
        back = scaler.inverse_transform(out, state)
        for var in mixed_grain_ds.data_vars:
            assert np.allclose(back[var].values, mixed_grain_ds[var].values)


class TestFunctionTransformer:
    """Named elementwise non-affine maps: the ratio→log chain's second half."""

    def test_log_parity_per_variable(self, mixed_grain_ds):
        ft = Func({"level_ratio": "log"})
        # mapping keys reference *variables*; a dataset may be fit directly,
        # and the derived variable must be built without the coord-collision
        # NaN trap (scalar selections squeeze the `control` dim)
        probe = mixed_grain_ds.assign(
            level_ratio=mixed_grain_ds["controls"].sel(control="level")
            / mixed_grain_ds["controls"].sel(control="reference")
        )
        state = ft.fit(probe)
        out = ft.transform(probe, state)
        assert np.allclose(out["level_ratio"].values, np.log(probe["level_ratio"].values))

    def test_mapping_scopes_to_named_variables(self, national_ds):
        ft = Func({"target": "log"})
        state = ft.fit(national_ds)
        out = ft.transform(national_ds, state)
        assert np.allclose(out["target"].values, np.log(national_ds["target"].values))
        assert np.allclose(out["media"].values, national_ds["media"].values)

    def test_inverse_pairs(self, national_ds):
        for pair in (("log", "exp"), ("log1p", "expm1"), ("sqrt", "square")):
            ft = Func({"target": pair[0]})
            state = ft.fit(national_ds["target"])
            positive = np.abs(national_ds["target"]) + 1.0
            forward = ft.transform(positive, state)
            back = ft.inverse_transform(forward, state)
            assert np.allclose(back.values, positive.values), pair

    def test_state_records_op_metadata(self, national_ds):
        state = Func({"target": "log"}).fit(national_ds["target"])
        assert state.attrs["kind"] == "function"
        assert state.attrs["op"] == "log"  # mapping stays on the spec, not the artifact
        assert len(state.data_vars) == 0

    def test_xscalers_affine_refuses_non_affine_steps(self, mixed_grain_ds):
        from xscalers.compose import VariableTransformer
        from xscalers.pipeline import Pipeline

        pipe = Pipeline(
            [
                # media_tv is abs-gamma-mental — strictly positive, log-safe
                ("log", Func({"media_tv": "log"})),
                ("scale", VariableTransformer({"media_tv": MeanScaler(dims=("date",))})),
            ]
        )
        state = pipe.fit(mixed_grain_ds)
        with pytest.raises(TypeError):
            pipe.affine(state)

    def test_spec_roundtrip(self, national_ds):
        from xscalers.core import from_spec

        original = Func({"target": "log1p"})
        rebuilt = from_spec(original.to_spec())
        state = original.fit(national_ds["target"].clip(min=0))
        out = rebuilt.transform(xr.DataArray(national_ds["target"].clip(min=0), name="target"), state)
        assert np.allclose(out.values, np.log1p(national_ds["target"].clip(min=0).values))
