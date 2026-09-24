"""Pipeline: sklearn-shaped steps, chained states, reversed inverse, composed affine."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xscalers import scalers
from xscalers.core import from_spec
from xscalers.dispatch import PerVariable
from xscalers.pipeline import Pipeline


def _make():
    return Pipeline(
        [
            ("target", PerVariable({"target": scalers.ZScore(dims=("date",))})),
            ("media", PerVariable({"media": scalers.MeanScaler(dims=("date",))})),
        ]
    )


class TestPipeline:
    def test_fit_returns_chained_states(self, national_ds):
        pipe = _make()
        state = pipe.fit(national_ds)
        assert set(state) == {"target", "media"}
        assert np.isclose(
            float(state["target"]["scale"].item()),
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
        assert isinstance(steps["media"], PerVariable)

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
        pipe = _make()
        state = pipe.fit(national_ds)
        view = pipe.affine(state)

        contrib = xr.DataArray(np.full(30, 0.25), dims=("date",))
        restored = contrib * view["target"]["scale"] + view["target"]["center"]
        direct = pipe.inverse_transform(restored, state)
        assert np.allclose(restored.values, direct["target"].values)

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
