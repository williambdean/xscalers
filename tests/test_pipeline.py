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
            float(state["target"]["target"]["scale"].item()),
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
