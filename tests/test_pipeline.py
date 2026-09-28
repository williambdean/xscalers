"""Pipeline: sklearn-shaped steps, chained states, reversed inverse, composed affine."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xscalers import preprocessing
from xscalers.compose import VariableTransformer
from xscalers.core import from_spec
from xscalers.pipeline import Pipeline, make_pipeline


def _make():
    return Pipeline(
        [
            ("target", VariableTransformer({"target": preprocessing.ZScore(dims=("date",))})),
            ("media", VariableTransformer({"media": preprocessing.MeanScaler(dims=("date",))})),
        ]
    )


class TestPipeline:
    def test_fit_returns_chained_states(self, national_ds):
        pipe = _make()
        state = pipe.fit(national_ds)
        assert set(state) == {"target", "media"}
        # composite steps contribute one level of nesting: step → variable → state
        assert np.isclose(
            float(state["target"]["target"]["std"].item()),
            national_ds["target"].std("date", ddof=0).item(),
        )

    def test_transform_then_inverse_roundtrip(self, national_ds):
        pipe = _make()
        state = pipe.fit(national_ds)
        out = pipe.transform(national_ds, state)
        back = pipe.inverse_transform(out, state)
        for var in national_ds.data_vars:
            assert np.allclose(back[var].values, national_ds[var].values)

    def test_named_step_transformer_reachable(self, national_ds):
        pipe = _make()
        steps = dict(pipe.steps)
        assert isinstance(steps["media"], VariableTransformer)

    def test_from_spec_roundtrip(self, national_ds):
        pipe = _make()
        spec = pipe.to_spec()
        rebuilt = from_spec(spec)
        state = rebuilt.fit(national_ds)
        out = rebuilt.transform(national_ds, state)
        back = rebuilt.inverse_transform(out, state)
        for var in national_ds.data_vars:
            assert np.allclose(back[var].values, national_ds[var].values)


class TestAffine:
    def test_composed_affine_matches_sklearn_zscore_plus_mean(self, national_ds):
        from sklearn.preprocessing import StandardScaler

        pipe = _make()
        state = pipe.fit(national_ds)

        view = pipe.affine(state)
        target_center = view["target"]["center"]
        target_scale = view["target"]["scale"]

        # manual composition: fit_transform target via ours and via sklearn
        da = national_ds["target"]
        sk = StandardScaler().fit_transform(da.values.reshape(-1, 1)).ravel()
        ours = (da - target_center) / target_scale
        assert np.allclose(ours.values, sk)

    def test_composed_affine_unscales_contributions(self, national_ds):
        """inverse_transform through the pipeline ≡ applying the composed affine pair."""
        pipe = _make()
        state = pipe.fit(national_ds)
        view = pipe.affine(state)

        contrib = xr.DataArray(np.full(30, 0.25), dims=("date",))
        direct = pipe.inverse_transform(xr.Dataset({"target": contrib}), state)
        expected = contrib * view["target"]["scale"] + view["target"]["center"]
        assert np.allclose(direct["target"].values, expected.values)

    def test_two_steps_on_one_variable_compose(self, mixed_grain_ds):
        """The per-variable composition math, across steps: two affine steps
        on the *same* variable must collapse to one pair."""
        pipe = Pipeline(
            [
                ("mean", VariableTransformer({"media_national": preprocessing.MeanScaler(dims=("date",))})),
                ("z", VariableTransformer({"media_national": preprocessing.ZScore(dims=("date",))})),
            ]
        )
        state = pipe.fit(mixed_grain_ds)
        pair = pipe.affine(state)["media_national"]
        out = pipe.transform(mixed_grain_ds, state)["media_national"]

        first = pipe["mean"].transformer.affine_view(pipe["mean"].state)["media_national"]
        second = pipe["z"].transformer.affine_view(pipe["z"].state)["media_national"]
        # y2 = (y1 - c2)/s2 with y1 = (x - c1)/s1
        assert np.allclose(pair["center"], (first["center"] + first["scale"] * second["center"]).values)
        assert np.allclose(pair["scale"], (first["scale"] * second["scale"]).values)
        # ... and that single pair is what the pipeline actually did
        by_hand = (mixed_grain_ds["media_national"] - pair["center"]) / pair["scale"]
        assert np.allclose(by_hand.transpose(*out.dims).values, out.values)
        # the composed pair is at the coarser grain, not the intermediate one
        assert pair["scale"].dims == ("media",)

    def test_unscale_matches_the_composed_pair_for_a_shared_variable(self, mixed_grain_ds):
        pipe = Pipeline(
            [
                ("mean", VariableTransformer({"media_national": preprocessing.MeanScaler(dims=("date",))})),
                ("z", VariableTransformer({"media_national": preprocessing.ZScore(dims=("date",))})),
            ]
        )
        state = pipe.fit(mixed_grain_ds)
        scaled = pipe.transform(mixed_grain_ds, state)["media_national"]
        contrib = xr.zeros_like(scaled) + 0.25
        back = pipe.unscale(
            xr.Dataset({"media_national": contrib}), var="media_national", state=state
        )
        pair = pipe.affine(state)["media_national"]
        expected = contrib * pair["scale"] + pair["center"]
        assert np.allclose(back["media_national"].values, expected.values)

    def test_affine_raises_for_non_affine_steps(self, national_ds):
        pipeline = Pipeline([("shape", NonAffineStep()), ("scale", _make())])
        state = pipeline.fit(national_ds)
        with pytest.raises(TypeError):
            pipeline.affine(state)


class NonAffineStep:
    """A step that changes values non-affinely — affine() must refuse."""

    def fit(self, X):
        return xr.Dataset()

    def transform(self, X, state):
        return X

    def inverse_transform(self, X, state):
        return X


class TestBoundStepAccess:
    """pipe["name"] = the fitted step: transformer + its state slice, usable on new data."""

    def test_step_access_returns_bound_view(self, national_ds):
        pipe = _make()
        pipe.fit(national_ds)
        step = pipe["target"]
        assert step.transformer is pipe.named_steps()["target"]
        assert step.state is not None
        assert set(step.state) == {"target"}

    def test_bound_step_transforms_new_data(self, national_ds):
        """The 'access transformations after the fact' path: apply one fitted step alone."""
        pipe = _make()
        pipe.fit(national_ds)
        conj = xr.Dataset({"target": xr.DataArray(np.ones(30) * 100.0, dims=("date",))})
        out = pipe["target"].transform(conj)
        scaled = conj["target"] - national_ds["target"].mean("date")
        scaled = scaled / national_ds["target"].std("date", ddof=0)
        assert np.allclose(out["target"].values, scaled.values)

    def test_bound_step_inverse(self, national_ds):
        pipe = _make()
        pipe.fit(national_ds)
        y = xr.Dataset({"target": xr.DataArray(np.full(30, 0.25), dims=("date",))})
        back = pipe["target"].inverse_transform(y)
        mean = national_ds["target"].mean("date").item()
        std = national_ds["target"].std("date", ddof=0).item()
        assert np.allclose(back["target"].values, y["target"].values * std + mean)

    def test_unfit_pipeline_raises(self, national_ds):
        pipe = _make()
        with pytest.raises(RuntimeError):
            pipe["target"]

    def test_unknown_step_raises(self, national_ds):
        pipe = _make()
        pipe.fit(national_ds)
        with pytest.raises(KeyError):
            pipe["nope"]


class TestPassthroughSteps:
    def test_string_passthrough_step_is_a_no_op(self, national_ds):
        pipeline = Pipeline(
            [
                ("identity", "passthrough"),
                ("scale", _make()),
            ]
        )
        state = pipeline.fit(national_ds)
        # the passthrough step contributes no state beyond an empty Dataset
        assert len(state["identity"].data_vars) == 0
        # downstream steps still chain: media scaled by the second step
        media = national_ds["media"]
        assert np.allclose(
            pipeline.transform(national_ds, state)["media"].values,
            (media / media.mean("date")).values,
        )
        back = pipeline.inverse_transform(pipeline.transform(national_ds, state), state)
        for var in national_ds.data_vars:
            assert np.allclose(back[var].values, national_ds[var].values)


class TestFittedStateAsCounterfactualBasis:
    """The scenario-analysis contract: fit once, then transform many perturbed
    copies of the raw data with the same state. Every counterfactual is scaled
    by the fit-time numbers, only the perturbed variable moves, and the step is
    pure -- no refit, no hidden state."""

    def test_one_state_serves_many_perturbations(self, mixed_grain_ds):
        pipe = Pipeline(
            [
                (
                    "scale",
                    VariableTransformer(
                        {
                            "media_national": preprocessing.MeanScaler(dims=("date",)),
                            "controls_prod": preprocessing.ZScore(dims=("date",)),
                        }
                    ),
                )
            ]
        )
        state = pipe.fit(mixed_grain_ds)
        base = pipe.transform(mixed_grain_ds, state)

        bumped = mixed_grain_ds.copy(deep=True)
        bumped["media_national"] = bumped["media_national"] * 1.10
        out = pipe.transform(bumped, state)

        # the perturbed variable moved, and by the same 10%
        assert np.allclose(
            out["media_national"].values, base["media_national"].values * 1.10
        )
        # every other variable is bit-identical to the unperturbed run
        for name in base.data_vars:
            if name != "media_national":
                assert out[name].identical(base[name]), name
        # and the step is pure: the same input gives the same output
        assert pipe.transform(bumped, state)["media_national"].identical(
            out["media_national"]
        )
        # the fit-time scale is still the original one, not the perturbed one
        assert np.allclose(
            state["scale"]["media_national"]["mean"].values,
            mixed_grain_ds["media_national"].mean("date").values,
        )

    def test_two_counterfactuals_are_independent_of_each_other(self, mixed_grain_ds):
        pipe = Pipeline(
            [("scale", VariableTransformer({"media_national": preprocessing.MeanScaler(dims=("date",))}))]
        )
        state = pipe.fit(mixed_grain_ds)
        up = mixed_grain_ds.copy(deep=True)
        up["media_national"] = up["media_national"] * 1.10
        down = mixed_grain_ds.copy(deep=True)
        down["media_national"] = down["media_national"] * 0.90
        # running one after the other must not leak state between them
        assert np.allclose(
            pipe.transform(down, state)["media_national"].values,
            pipe.transform(mixed_grain_ds, state)["media_national"].values * 0.90,
        )
        assert np.allclose(
            pipe.transform(up, state)["media_national"].values,
            pipe.transform(mixed_grain_ds, state)["media_national"].values * 1.10,
        )


class TestMakePipeline:
    def test_auto_names_from_class(self, national_ds):
        z = preprocessing.ZScore(dims=("date",))
        m = preprocessing.MeanScaler(dims=("date",))
        pipe = make_pipeline(z, m)
        assert [name for name, _ in pipe.steps] == ["zscore", "meanscaler"]

    def test_repeated_transformers_get_suffixed_names(self):
        z1, z2 = preprocessing.ZScore(), preprocessing.ZScore()
        pipe = make_pipeline(z1, z2)
        assert [name for name, _ in pipe.steps] == ["zscore", "zscore-1"]
