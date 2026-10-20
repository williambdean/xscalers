# xscalers

sklearn-shaped, dim-aware scaling transformers for `xarray` and
`pytensor.xtensor`.

**Fit is eager xarray**: the fitted state is an `xr.Dataset` you can inspect
and edit, and **transform is pure arithmetic by dimension name** that runs
identically on `DataArray`s and symbolic `pytensor.xtensor` variables. All v1
scalers are affine, so model code binds only to a composed `(center, scale)`
pair. Nothing custom: every container and serialization format is a native
ecosystem object.

## Why

Hand-writing `(x - mean) / std` is fine until the regression's inputs are an
`xr.Dataset` with mixed grains and the model is PyMC. Three familiar pains:

- sklearn's transformers assume stacked 2-D arrays and hide fitted state in
  `mean_`-style attributes, which neither preserves dims nor hands you the
  fitted constants;
- the scaling constants fed into a `pytensor.xtensor` graph are usually
  hand-copied, so converting a model contribution back to real terms goes
  silently wrong;
- xarray itself ships no scaling transformer.

xscalers fills exactly that gap: affine scaling over dimension names on
native xarray objects, fitted state as an inspectable `xr.Dataset` artifact,
the same arithmetic lifting into a symbolic `pytensor.xtensor` graph, and
one `unscale` verb that converts model contributions back to real units,
refusing where the algebra is not affine instead of guessing.

The frame is a regression loop you can toggle: point the same pipeline at a
different transformation (`MeanScaler` vs `ZScore` vs `FixedScaler`, per
variable) and regenerate the model-ready `xr.Dataset` without touching model
code, then invert predictions and contributions back to real terms with the
same fitted state.

## Install

```bash
pip install xscalers            # core (xarray, numpy)
pip install "xscalers[pytensor]"  # with the symbolic pytensor extra
```

## Quick start

```python
import xarray as xr

from xscalers.datasets import load_national
from xscalers.preprocessing import MeanScaler

national: xr.Dataset = load_national()

scaler = MeanScaler(("date",))
state = scaler.fit(national["media"])
scaled = scaler.transform(national["media"], state)
back = scaler.inverse_transform(scaled, state)   # round-trips exactly
```

## Where to next

- [Guides](guides/index.md) for the fit/transform/state workflow
- [API Reference](reference/index.md) for the full vocabulary
