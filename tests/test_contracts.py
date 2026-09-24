"""Contract tests — the library's core semantics: state shape, affine view, dispatch"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xscalers.core import (
    BaseTransformer,
    affine_pair,
    from_spec,
    register,
)


@register
class SetScale(BaseTransformer):
    """Minimal concrete transformer for contract tests: divide by a constant."""

    def __init__(self, scale: float = 4.0):
        self.scale = scale
        self.dims = None

    def fit(self, X):
        return self.compute(X)

    def compute(self, X):
        return xr.Dataset(
            {"center": 0.0, "scale": self.scale},
            attrs={"kind": "set_scale", "dims": (), "scaled_from": getattr(X, "name", None)},
        )

    def to_spec_kwargs(self):
        return {"scale": self.scale}

    def transform(self, X, state):
        return X / state["scale"]

    def inverse_transform(self, X, state):
        return X * state["scale"] + state["center"]


class TestStateConventions:
    def test_state_is_a_dataset(self, national_ds):
        state = SetScale().fit(national_ds["target"])
        assert isinstance(state, xr.Dataset)
        assert "center" in state and "scale" in state

    def test_state_attrs_metadata(self, national_ds):
        state = SetScale().fit(national_ds["target"])
        assert state.attrs["kind"] == "set_scale"
        assert state.attrs["scaled_from"] == "target"

    def test_scalar_values_are_0d_data_vars_not_attrs(self, national_ds):
        state = SetScale(scale=4.5).fit(national_ds["target"])
        assert float(state["scale"].item()) == 4.5

    def test_state_is_mapping_of_dataarrays(self, national_ds):
        state = SetScale().fit(national_ds["target"])
        state["scale"] = state["scale"] * 2  # Dataset is a setitem-able mapping
        out = SetScale().transform(national_ds["target"], state)
        assert np.allclose(out.values, national_ds["target"].values / 8.0)


class TestAffineView:
    def test_affine_pair_roundtrip(self, national_ds):
        da = national_ds["media"]
        state = SetScale(scale=2.0).fit(da)
        scaled = da / 2.0
        pair = affine_pair(state)
        restored = scaled * pair["scale"] + pair["center"]
        assert np.allclose(restored.values, da.values)

    def test_non_invertible_transformer_raises_on_inverse(self, national_ds):
        class Drop(BaseTransformer):
            dims = None

            def compute(self, X):
                return xr.Dataset()

            def _transform(self, X, state):
                return X

            def _inverse_transform(self, X, state):
                raise NotImplementedError

        step = Drop()
        state = step.fit(national_ds["target"])
        assert len(state.data_vars) == 0
        with pytest.raises(NotImplementedError):
            step.inverse_transform(national_ds["target"], state)

    def test_unknown_spec_kind_raises_value_error(self):
        with pytest.raises(ValueError):
            from_spec({"kind": "Nope", "kwargs": {}})


class TestSpecRoundtrip:
    def test_from_spec_rebuilds(self, national_ds, tmp_path):
        original = SetScale(scale=3.0)
        spec = original.to_spec()
        rebuilt = from_spec(spec)
        assert isinstance(rebuilt, SetScale)
        assert rebuilt.scale == 3.0

        path = tmp_path / "state.nc"
        original.fit(national_ds["target"]).to_netcdf(path)
        state = xr.open_dataset(path)
        out = original.transform(national_ds["target"], state)
        assert np.allclose(out.values, national_ds["target"].values / 3.0)
