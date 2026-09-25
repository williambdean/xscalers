"""Pipeline.unscale — one verb for real-terms conversion (the affine pair, internal)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xscalers import Func, MeanScaler, ZScore
from xscalers.compose import PerVariable
from xscalers.pipeline import Pipeline


def _national_pipeline():
    return Pipeline(
        [("scale", PerVariable({"target": ZScore(dims=("date",)), "media": MeanScaler(dims=("date",))}))],
    )


class TestUnscaleAffine:
    def test_manual_pair_equals_unscale(self, national_ds):
        """The hand-written algebra the API replaces — must be numerically identical."""
        pipe = _national_pipeline()
        pipe.fit(national_ds)

        contrib = xr.DataArray(np.full((2, 5, 104, 3), 0.25), dims=("chain", "draw", "date", "channel"))
        expected = (
            contrib * pipe["scale"].state["media"]["scale"]
            + pipe["scale"].state["media"]["center"]
        )
        out = pipe.unscale(contrib, var="media")
        assert np.allclose(out.values, expected.values)

    def test_target_center_travels(self, national_ds):
        """z-score's nonzero mean is part of the composed pair — carried by unscale."""
        pipe = _national_pipeline()
        state = pipe.fit(national_ds)
        contrib = xr.DataArray(np.full(104, 0.25), dims=("date",))
        view = pipe.affine(state)
        manual = contrib * view["target"]["scale"] + view["target"]["center"]
        assert np.allclose(pipe.unscale(contrib, var="target").values, manual.values)

    def test_symbolic_parity(self, national_ds):
        """Symbolic (pytensor.xtensor) contributions unscale identically."""
        pytest.importorskip("pytensor")
        import pytensor
        import pytensor.xtensor as ptx

        pipe = _national_pipeline()
        pipe.fit(national_ds)
        sym = ptx.xtensor("media_contrib", dims=("date", "channel"))
        f = pytensor.function([sym], pipe.unscale(sym, var="media").values)
        value = np.full((104, 3), 0.5)
        eager = pipe.unscale(
            xr.DataArray(value, dims=("date", "channel")), var="media"
        ).values
        assert np.allclose(f(value), eager)


class TestUnscaleNonAffineFallback:
    def test_log_chain_delegates_to_inverse_transform(self, mixed_grain_ds):
        from xscalers import FixedScaler
        from xscalers.preprocessing import Derive

        pipe = Pipeline(
            [
                (
                    "derive",
                    Derive(
                        name="level_rel_reference",
                        numerator=("controls", {"control": "level"}),
                        denominator=("controls", {"control": "reference"}),
                        op="ratio",
                    ),
                ),
                ("log", Func({"level_rel_reference": "log"})),
                ("scale", PerVariable({"level_rel_reference": FixedScaler(scale=1.0)})),
            ]
        )
        pipe.fit(mixed_grain_ds)
        contrib = xr.DataArray(np.full((104, 4), -1.0), dims=("date", "product"))

        # the affine shortcut would be *silently wrong* here (log is not affine);
        # unscale's fallback walks only THIS variable's own steps backwards —
        # un-scale (fixed, scale=1) then un-log — stopping before Derive, so
        # the result is the derived variable's real-space value: exp(−1)
        out = pipe.unscale(contrib, var="level_rel_reference")
        assert np.allclose(out.values, np.exp(contrib.values))


class TestUnscaleErrors:
    def test_missing_var_raises_keyerror(self, national_ds):
        pipe = _national_pipeline()
        pipe.fit(national_ds)
        with pytest.raises(KeyError):
            pipe.unscale(xr.DataArray(np.zeros(5), dims=("date",)), var="nope")

    def test_reload_requires_explicit_state(self, national_ds, tmp_path):
        """from_spec pipelines have no fitted states_ — the state must be explicit."""
        from xscalers.core import from_spec

        pipe = _national_pipeline()
        state = pipe.fit(national_ds)  # state is the artifact (a DataTree), reloaded below
        from xscalers.container import state_from_tree, state_tree

        path = tmp_path / "state.nc"
        state_tree(state).to_netcdf(path)
        reloaded = from_spec(pipe.to_spec())
        with pytest.raises(RuntimeError, match="no fitted state"):
            reloaded.unscale(
                xr.DataArray(np.zeros((104, 3)), dims=("date", "channel")), var="media"
            )
        # with the reloaded state bound explicitly, the same verb works
        reloaded_state = {"scale": state_from_tree(xr.open_datatree(path))["scale"]}
        contributions = xr.DataArray(np.full((104, 3), 0.5), dims=("date", "channel"))
        out_fit = pipe.unscale(contributions, var="media")
        out_reloaded = reloaded.unscale(contributions, var="media", state=reloaded_state)
        assert np.allclose(out_fit.values, out_reloaded.values)
