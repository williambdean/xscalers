# Saving the fitted state

State is **data**: saving it is native xarray IO, in any of the three shapes
you'll meet in practice.

## 1. A standalone one-file artifact

`to_netcdf`, next to your recipe:

```python
import os

import numpy as np

from xscalers.datasets import load_national
from xscalers.preprocessing import MeanScaler

national: xr.Dataset = load_national()
scaler = MeanScaler(dims=("date",))
state = scaler.fit(national["media"])

state.to_netcdf("media_state.nc")                 # state is a Dataset — native IO
reloaded = xr.open_dataset("media_state.nc")      # ...read back natively
scaled = scaler.transform(national["media"], reloaded)
back = scaler.inverse_transform(scaled, reloaded)
np.testing.assert_allclose(back, national["media"])
os.remove("media_state.nc")                       # tidy up after this example
```

## 2. Alongside your (large) zarr inputs

Append the state tree as groups of the existing store. `append_state` exists
only because `DataTree.to_zarr(group=...)` is not implemented upstream;
reading back is native.

```python
from pathlib import Path

import xarray as xr

from xscalers.compose import VariableTransformer
from xscalers.container import state_tree
from xscalers.datasets import load_mixed_grain
from xscalers.persistence import append_state

store: Path = Path("inputs.zarr")                 # your existing input store
mixed: xr.Dataset = load_mixed_grain()            # pretend it already lives there
mixed.to_zarr(store)                              # ... here we create it first

transformer: VariableTransformer = VariableTransformer(
    {"media_national": MeanScaler(dims=("date",))},
    remainder="drop",
)
state = transformer.fit(mixed)
append_state(state_tree({"scale": state}), store, group="scalers")

reloaded_tree = xr.open_datatree(store, group="scalers")
reloaded_state = {
    name: node.dataset for name, node in reloaded_tree.children["scale"].children.items()
}
scaled_again = transformer.transform(mixed, reloaded_state)
import shutil; shutil.rmtree(store)               # tidy up after this example
```

`mode="a"` means a refit *overwrites* the group: name groups per fit if you
want history.

## 3. Next to a PyMC model

The state and the `InferenceData` are separate artifacts sharing a
directory; the recipe (JSON via `to_spec`) is versioned config, state is
data, never the two mixed.

```python
# requires-pymc
import pymc as pm

from xscalers.datasets import load_national
from xscalers.preprocessing import MeanScaler

national = load_national()
scaler = MeanScaler(dims=("date",))
state = scaler.fit(national["target"])
y = scaler.transform(national["target"], state)   # scaled input into the model

with pm.Model() as model:
    beta = pm.Normal("beta", 0, 1)
    sigma = pm.HalfNormal("sigma", 1)
    # numpy views into the scaled dataset operate the same as scaled arrays
    pm.Normal("obs", beta * float(y.mean()), sigma, observed=np.asarray(y))
    idata = pm.sample_prior_predictive(draws=10, random_seed=0)
```

!!! note "Requires pymc"
    This block runs only with the `pytensor` extra installed
    (`pip install "xscalers[pytensor]"`); it is skipped otherwise.

The whole fitted pipeline serializes as *recipe + artifact*, one file:

```python
# continues above — `state` carries over
from xscalers.container import state_tree

state_tree({"scale": state}).to_netcdf("fitted_state.nc")   # one artifact
```
