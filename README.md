# xscalers

sklearn-shaped, dim-aware scaling transformers for `xarray` and `pytensor.xtensor`.

**Fit is eager xarray** — the fitted state is an `xr.Dataset` you can inspect
and edit — and **transform is pure arithmetic by dimension name** that runs
identically on `DataArray`s and symbolic `pytensor.xtensor` variables. All v1
scalers are affine, so model code binds only to a composed `(center, scale)`
pair. Nothing custom: every container and serialization format is a native
ecosystem object. Every example below is executed against this README by the
test suite, so it stays copy-paste-runnable.

## The contract

```text
class BaseTransformer:
    dims: tuple[str, ...] | None                       # dims to reduce over; None = all
    def fit(self, X) -> xr.Dataset                     # fitted state, inspectable + editable
    def transform(self, X, state) -> X                 # state: Mapping[str, DataArray]
    def inverse_transform(self, X, state) -> X
```

Dataset and DataTree inputs are handled by per-variable dispatch — each
variable keeps its own remaining dims, which is what makes mixed-grain inputs
work with no special casing.

## The basics

```python
from xscalers import MeanScaler, load_national, load_mixed_grain

national = load_national()

state = MeanScaler(dims=("date",)).fit(national["media"])
state         # {"center": 0.0, "scale": per-channel mean over date}

scaled = MeanScaler(dims=("date",)).transform(national["media"], state)

# state is editable — transforms honor it
state["scale"] = state["scale"] * 2
halved = MeanScaler(dims=("date",)).transform(national["media"], state)
```

Mixed grains need no special casing: `fit` a Dataset and each variable gets
its own state:

```python
from xscalers import FixedScaler, MeanScaler, ZScore, load_mixed_grain
from xscalers.dispatch import PerVariable

mixed = load_mixed_grain()
dispatch = PerVariable(
    {
        "media_national": MeanScaler(dims=("date",)),
        "media_digital": ZScore(dims=("date",)),
        "media_tv": FixedScaler(scale=1_000_000),
    },
    default="passthrough",
)
state = dispatch.fit(mixed)              # one Dataset state per variable
scaled = dispatch.transform(mixed, state)  # each var scaled by its own recipe
```

## Splitting one variable into differently-treated pieces

Different treatment *within* one media variable (digital vs TV) works by
splitting into pieces, scaling each, then recombining. Real MMM inputs are a
plain Dataset of variables, so pieces become *sibling variables*
(`media_digital`, `media_linear`), each keeping its slice of the coordinate
under a piece-specific dim name; `inverse_transform` recombines and restores
the original coordinate order:

```python
from xscalers import FixedScaler, MeanScaler, Pipeline, Splitter
from xscalers.dispatch import PerVariable
from xscalers.datasets import load_media_channels

ds = load_media_channels()

pipe = Pipeline(
    [
        (
            "split",
            Splitter(
                "media",
                {
                    "digital": {"channel": ["search", "social", "fb"]},
                    "linear": {"channel": ["tv", "stream"]},
                },
            ),
        ),
        (
            "scale",
            PerVariable(
                {
                    "media_digital": MeanScaler(dims=("date",)),
                    "media_linear": FixedScaler(scale=1_000),
                },
            ),
        ),
    ]
)
state = pipe.fit(ds)
scaled = pipe.transform(ds, state)

back = pipe.inverse_transform(scaled, state)   # pieces recombine; coords restored
import numpy as np
np.testing.assert_allclose(back["media"], ds["media"])
```

## Saving the fitted state

State is **data** — saving it is native xarray IO, in any of the three shapes
you'll meet in practice:

**1. A standalone one-file artifact** — `to_netcdf`, next to your recipe:

```python
import os
import xarray as xr
import numpy as np
from xscalers import MeanScaler
from xscalers.datasets import load_national

national = load_national()
scaler = MeanScaler(dims=("date",))
state = scaler.fit(national["media"])

state.to_netcdf("media_state.nc")                 # state is a Dataset — native IO
reloaded = xr.open_dataset("media_state.nc")      # ...read back natively
scaled = scaler.transform(national["media"], reloaded)
back = scaler.inverse_transform(scaled, reloaded)
np.testing.assert_allclose(back, national["media"])
os.remove("media_state.nc")                       # tidy up after this example
```

**2. Alongside your (large) zarr inputs** — append the state tree as groups
of the existing store; `append_state` exists only because
`DataTree.to_zarr(group=...)` is not implemented upstream — reading back is
native:

```python
from pathlib import Path
from xscalers import append_state
from xscalers.container import state_tree

store = Path("mmm.zarr")                     # your existing input store
mixed = load_mixed_grain()                   # pretend it already lives there
mixed.to_zarr(store)                         # ... here we create it first

dispatch = PerVariable({"media_national": MeanScaler(dims=("date",))}, default="drop")
state = dispatch.fit(mixed)
append_state(state_tree({"scale": state}), store, group="scalers")

import xarray as xr
reloaded_tree = xr.open_datatree(store, group="scalers")
reloaded_state = {
    name: node.dataset for name, node in reloaded_tree.children["scale"].children.items()
}
scaled_again = dispatch.transform(mixed, reloaded_state)
import shutil; shutil.rmtree(store)          # tidy up after this example
```

**3. Next to a pymc model** — the state and the `InferenceData` are separate
artifacts sharing a directory; recipe (JSON via `to_spec`) is versioned
config, state is data, never the two mixed:

```python
# requires-pymc
import pymc as pm
import arviz as az
from xscalers import MeanScaler
from xscalers.datasets import load_national

national = load_national()
scaler = MeanScaler(dims=("date",))
state = scaler.fit(national["target"])
y = scaler.transform(national["target"], state)   # scaled input into the model

with pm.Model() as model:
    beta = pm.Normal("beta", 0, 1)
    sigma = pm.HalfNormal("sigma", 1)
    pm.Normal("obs", beta * y.mean(), sigma, observed=y)
    idata = pm.sample_prior_predictive(draws=10, random_seed=0)

az.to_netcdf(idata, "posterior.nc")               # model results...
state.to_netcdf("scaler_state.nc")                # ...and its scaling, side by side
idata_back = az.from_netcdf("posterior.nc")
state_back = xr.open_dataset("scaler_state.nc")
contributions_original_scale = idata_back["prior"]["beta"].mean() * state["scale"] + state["center"]
import os
for name in ("posterior.nc", "scaler_state.nc"):
    os.remove(name)
```

The fitted pipeline serializes as *recipe + artifact*:

```python
import os
from xscalers.container import state_tree

state_tree({"scale": state}).to_netcdf("fitted_state.nc")   # whole pipeline: one artifact
os.remove("fitted_state.nc")                                # tidy up after this example
```

## The flagship flow: fit on xarray, unscale a pymc graph

Requires the optional `pytensor` extra (`pip install "xscalers[pytensor]"`).

```python
import pytensor.xtensor as ptx
from xscalers import MeanScaler
from xscalers.datasets import load_national

scaler = MeanScaler(dims=("date",))
state = scaler.fit(load_national()["media"])

contrib = ptx.xtensor("channel_contrib", dims=("date", "channel"))
spend_real = scaler.inverse_transform(contrib, state)      # symbolic: contrib * mean
```

After sampling, contributions arrive as xarray with `(chain, draw, ...)` —
the same state broadcasts over those extra dims by name.

## Writing it out: post-fit step access

```python
from xscalers import MeanScaler
from xscalers.pipeline import make_pipeline
from xscalers.datasets import load_national

pipe = make_pipeline(MeanScaler(dims=("date",)), MeanScaler(dims=("date",)))
assert [name for name, _ in pipe.steps] == ["meanscaler", "meanscaler-1"]
pipe.fit(load_national())                      # sklearn-style fitted pipeline

bound = pipe["meanscaler"]                     # transformer + its fitted state slice
scaled_again = bound.transform(load_national())  # one fitted step on fresh data
back = bound.inverse_transform(scaled_again)     # ...and its inverse
```

## For sklearn users

sklearn vocabulary: `fit` / `transform` / `inverse_transform` / `Pipeline`,
`ColumnTransformer`-style dispatch (that's `PerVariable`), `make_pipeline`,
`"passthrough"` steps. The deliberate departure: **state is explicit and
passed around**, not hidden as `scaler.mean_` — that is what lets you edit
fitted values and reuse them across backends.

## Install

```text
pip install xscalers            # core (xarray, numpy)
```

## License

MIT
