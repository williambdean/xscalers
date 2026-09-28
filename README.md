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

Dataset and DataTree inputs are handled per variable — each variable keeps
its own remaining dims, which is what makes mixed-grain inputs work with no
special casing.

## The basics

```python
import xarray as xr

from xscalers.datasets import load_mixed_grain, load_national
from xscalers.preprocessing import MeanScaler

national: xr.Dataset = load_national()

scaler = MeanScaler(("date",))
state = scaler.fit(national["media"])
scaled = scaler.transform(national["media"], state)

state
```

```text
<xarray.Dataset> Size: 96B
Dimensions:  (channel: 3)
Coordinates:
  * channel  (channel) <U6 72B 'search' 'social' 'fb'
Data variables:
    mean     (channel) float64 24B 49.68 48.81 53.79
Attributes:
    kind:         mean
    dims:         ['date']
    scaled_from:  media
    spec_sha:     af4a187be0e9
    recipe:       MeanScaler(dims=['date'])
```

State is a value on the record — inspect it, save it, pass it to another
process. Hand-editing is a manual override (the stamp records the recipe,
not your edits).

Mixed grains need no special casing: `fit` a Dataset and each variable gets
its own state:

```python
import xarray as xr

from xscalers.compose import VariableTransformer
from xscalers.datasets import load_mixed_grain
from xscalers.preprocessing import FixedScaler, MeanScaler, ZScore

mixed: xr.Dataset = load_mixed_grain()
transformer = VariableTransformer(
    {
        "media_national": MeanScaler(dims=("date",)),
        "media_digital": ZScore(dims=("date",)),
        "media_tv": FixedScaler(scale=1_000_000),
    },
    remainder="passthrough",
)
state = transformer.fit(mixed)              # one Dataset state per variable
scaled = transformer.transform(mixed, state)  # each var scaled by its own recipe
```

## Derived metrics — new variables, even across grains

`Derive` turns selected pieces of existing variables into a *new* variable
that later steps treat like anything else. The two pieces don't have to come
from the same variable or the same grain — a national series over a
product-grain control broadcasts up to product grain, so `tv_per_control`
below is a product-grain variable built from a national input.

Ops are a closed, spec-serializable vocabulary — `op="ratio | diff | product"`
— and selector pieces take `reduce="sum | mean | first | last"` for
share-of-total style derivations:

```python
import xarray as xr

from xscalers.compose import VariableTransformer
from xscalers.datasets import load_mixed_grain
from xscalers.pipeline import Pipeline
from xscalers.preprocessing import Derive, FunctionTransformer, Splitter, ZScore

mixed: xr.Dataset = load_mixed_grain()
mixed["media_national"].dims                  # (date, media) — national grain
mixed["controls_prod"].dims                    # (date, product) — product grain

pipe = Pipeline(
    [
        (
            "derive",
            Derive(
                name="tv_per_control",
                left=("media_national", {"media": "tv"}),
                right=("controls_prod", {}),   # {} = the whole variable
                op="ratio",
            ),
        ),
        ("log", FunctionTransformer({"tv_per_control": "log"})),  # named funcs
        ("scale", VariableTransformer({"tv_per_control": ZScore(dims=("date",))})),
    ]
)
state = pipe.fit(mixed)
scaled = pipe.transform(mixed, state)
scaled["tv_per_control"].dims                  # (date, product) — broadcast, then scaled

back = pipe.inverse_transform(scaled, state)
import numpy as np
np.testing.assert_allclose(back["controls_prod"], mixed["controls_prod"])
```

Because the z-score reduces over `date` only, each product keeps its own
mean and standard deviation — the per-variable, per-group treatment that
mixed-grain data needs, with no reshaping.

`inverse_transform` drops the derived variable and restores the inputs — so
the whole chain (derive → log → z-score — unzscore → exp → un-derive)
round-trips. Note that a log makes the target path non-affine: `pipe.affine()`
will refuse such pipelines, and real-terms contributions must go through the
batched `inverse_transform` instead of the `(β, scale)` shortcut.

The same step derives within one variable too, when the variable carries an
extra coordinate dimension — `controls (date, product, control)` with
`control = [level, reference, seasonality]`:

```python
# continues above — `mixed` carries over
from xscalers.preprocessing import Derive

share = Derive(
    name="level_diff",                           # op="diff" this time:
    left=("controls", {"control": "level"}), # level minus reference,
    right=("controls", {"control": "reference"}),
    op="diff",                                   # not a ratio
)
derived = share.transform(mixed, share.fit(mixed))
derived["level_diff"].dims                     # (date, product)
```

## Dropping variables and coordinate values

`VariableDropper` is the drop verb — the xarray answer to sklego's `ColumnDropper`.
`vars=` removes whole variables; `labels=` removes values from a coordinate
dataset-wide (the complement of keeping, which is `Selector`'s job):

```python
# continues above — `mixed` carries over
from xscalers.preprocessing import VariableDropper

prune = VariableDropper(vars=["media_national", "controls_prod"], labels={"control": ["reference"]})
pruned = prune.transform(mixed, prune.fit(mixed))
"media_national" in pruned.data_vars          # False — raw inputs are gone
"reference" in pruned.coords["control"].values  # False — every variable shrank with it
```

The motivating shape is pruning raw inputs once a derived metric supersedes
them, so later steps never have to enumerate what they don't care about:

```python
# continues above — `mixed` carries over
from xscalers.compose import VariableTransformer
from xscalers.pipeline import Pipeline
from xscalers.preprocessing import Derive, VariableDropper, ZScore

pipe = Pipeline(
    [
        (
            "derive",
            Derive(
                name="tv_per_control",
                left=("media_national", {"media": "tv"}),
                right=("controls_prod", {}),
                op="ratio",
            ),
        ),
        ("prune", VariableDropper(vars=["media_national", "controls_prod"])),
        ("scale", VariableTransformer({"tv_per_control": ZScore(dims=("date",))})),
    ]
)
model_ready = pipe.fit_transform(mixed)
sorted(model_ready.data_vars)                 # raw inputs pruned, ratio kept
```

Dropping destroys data, so — like `Selector` — `VariableDropper.inverse_transform`
raises instead of guessing, and `pipe.inverse_transform` refuses for the whole
chain. Real-terms conversion is unaffected: `pipe.unscale(contrib,
var="tv_per_control")` still walks that one variable backwards. `fit` validates
rather than failing late (sklego's contract): unknown names raise `KeyError`,
dropping *every* variable raises `ValueError`, and a non-`Dataset` input raises
`TypeError` (DataTree node drops are deferred).

If you need the reverse, a pipeline that scales what it keeps, `Splitter` is
the invertible way to split one variable into differently-treated pieces, and
`VariableTransformer(..., remainder="drop")` is the leftover-dropping sibling.


## Splitting one variable into differently-treated pieces

Different treatment *within* one media variable (digital vs TV) works by
splitting into pieces, scaling each, then recombining. Real inputs are a
plain Dataset of variables, so pieces become *sibling variables*
(`media_digital`, `media_linear`), each keeping its slice of the coordinate
under a piece-specific dim name; `inverse_transform` recombines and restores
the original coordinate order:

```python
from xscalers.compose import VariableTransformer
from xscalers.datasets import load_media_channels
from xscalers.pipeline import Pipeline
from xscalers.preprocessing import FixedScaler, MeanScaler, Splitter

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
            VariableTransformer(
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

**2. Alongside your (large) zarr inputs** — append the state tree as groups
of the existing store; `append_state` exists only because
`DataTree.to_zarr(group=...)` is not implemented upstream — reading back is
native:

```python
# continues above — `xr`, `mixed` and the fitted `transformer` carry over
from pathlib import Path
from xscalers.compose import VariableTransformer
from xscalers.container import state_tree
from xscalers.datasets import load_mixed_grain
from xscalers.persistence import append_state
from xscalers.preprocessing import MeanScaler

store: Path = Path("inputs.zarr")             # your existing input store
mixed: xr.Dataset = load_mixed_grain()       # pretend it already lives there
mixed.to_zarr(store)                         # ... here we create it first

transformer: VariableTransformer = VariableTransformer({"media_national": MeanScaler(dims=("date",))}, remainder="drop")
state = transformer.fit(mixed)
append_state(state_tree({"scale": state}), store, group="scalers")  # state under "scale"

import xarray as xr
reloaded_tree = xr.open_datatree(store, group="scalers")
reloaded_state = {
    name: node.dataset for name, node in reloaded_tree.children["scale"].children.items()
}
scaled_again = transformer.transform(mixed, reloaded_state)
import shutil; shutil.rmtree(store)          # tidy up after this example
```

**3. Next to a pymc model** — the state and the `InferenceData` are separate
artifacts sharing a directory; recipe (JSON via `to_spec`) is versioned
config, state is data, never the two mixed:

```python
# requires-pymc
import pymc as pm
import arviz as az
from xscalers.datasets import load_national
from xscalers.preprocessing import MeanScaler

national = load_national()
scaler = MeanScaler(dims=("date",))
state = scaler.fit(national["target"])
y = scaler.transform(national["target"], state)   # scaled input into the model

import xarray as xr
with pm.Model() as model:
    beta = pm.Normal("beta", 0, 1)
    sigma = pm.HalfNormal("sigma", 1)
    # numpy views into the scaled dataset operate the same as scaled arrays
    pm.Normal("obs", beta * float(y.mean()), sigma, observed=np.asarray(y))
    idata = pm.sample_prior_predictive(draws=10, random_seed=0)

import numpy as np
idata.to_netcdf("posterior.nc")                   # model results...
state.to_netcdf("scaler_state.nc")                # ...and its scaling, side by side
idata_back = xr.open_datatree("posterior.nc")     # DataTree in, DataTree out
state_back = xr.open_dataset("scaler_state.nc")

# real-terms conversion: the affine pair (center, scale) — MeanScaler is
# divisive, so its composed view is (center=0, scale=mean):
posterior_beta: float = float(idata_back["prior"]["beta"].mean())
contribution_scaled: float = posterior_beta * float(np.asarray(y).mean())
contributions_original_scale = contribution_scaled * float(state_back["mean"])
import os
for name in ("posterior.nc", "scaler_state.nc"):
    os.remove(name)
```

The fitted pipeline serializes as *recipe + artifact*:

```python
# continues above — `state` carries over
import os
from xscalers.container import state_tree

state_tree({"scale": state}).to_netcdf("fitted_state.nc")   # whole pipeline: one artifact
```

## Converting back to real units — one verb

Model code never hand-writes `× scale + center`: the affine pair is an
*internal* detail. `Pipeline.unscale` applies the composed pair per variable
(eager xarray contributions or symbolic `pytensor.xtensor` in the graph);
when a pipeline is non-affine (e.g. a log-derived metric), `unscale` walks
only *that variable's own* steps backwards, stopping at the step that created
it — so the one call stays honest where hand-written `β·scale` algebra goes
silently wrong:

```python
import numpy as np
import xarray as xr
from xscalers.compose import VariableTransformer
from xscalers.datasets import load_mixed_grain
from xscalers.pipeline import Pipeline
from xscalers.preprocessing import Derive, FixedScaler, FunctionTransformer, ZScore
import os

pipe = Pipeline(
    [
        (
            "derive",
            Derive(
                name="tv_per_control",
                left=("media_national", {"media": "tv"}),
                right=("controls_prod", {}),
                op="ratio",
            ),
        ),
        ("log", FunctionTransformer({"tv_per_control": "log"})),
        ("scale", VariableTransformer({"tv_per_control": FixedScaler(scale=1.0)})),
    ]
)
state = pipe.fit(load_mixed_grain())
contrib = xr.DataArray(np.full((104, 4), -0.75), dims=("date", "product"))

y_real = pipe.unscale(contrib, var="tv_per_control")   # one verb; located = exp(−0.75)
np.testing.assert_allclose(y_real, np.exp(contrib))
```

## The flagship flow: fit on xarray, unscale a pymc graph

Requires the optional `pytensor` extra (`pip install "xscalers[pytensor]"`).

```python
import pytensor.xtensor as ptx
from xscalers.compose import VariableTransformer
from xscalers.datasets import load_national
from xscalers.pipeline import Pipeline
from xscalers.preprocessing import MeanScaler

scaler = MeanScaler(dims=("date",))
data = load_national()
pipe = Pipeline([("scale", VariableTransformer({"media": scaler}))])
state = pipe.fit(data)                                     # fitted pipeline state

contrib = ptx.xtensor("channel_contrib", dims=("date", "channel"))
spend_real = pipe.unscale(contrib, var="media", state=state)   # symbolic: contrib * mean
```

After sampling, contributions arrive as xarray with `(chain, draw, ...)` —
the same state broadcasts over those extra dims by name.

## Writing it out: post-fit step access

```python
from xscalers.datasets import load_national
from xscalers.pipeline import make_pipeline
from xscalers.preprocessing import MeanScaler

pipe = make_pipeline(MeanScaler(dims=("date",)), MeanScaler(dims=("date",)))
assert [name for name, _ in pipe.steps] == ["meanscaler", "meanscaler-1"]
pipe.fit(load_national())                      # sklearn-style fitted pipeline

bound = pipe["meanscaler"]                     # transformer + its fitted state slice
scaled_again = bound.transform(load_national())  # one fitted step on fresh data
back = bound.inverse_transform(scaled_again)     # ...and its inverse
```

## For sklearn users

sklearn vocabulary: `fit` / `transform` / `inverse_transform` / `Pipeline`,
`ColumnTransformer`-style per-variable treatment (that's `VariableTransformer`), `make_pipeline`,
**`make_union`** (many pipelines over the same dataset, merged with a
collision policy — covariates-only by design), `"passthrough"` steps. The
drop verb lives in `VariableDropper` — sklego's `ColumnDropper` over
`drop_vars`/`drop_sel`, with `Selector` covering the keeping side. The
deliberate departure: **state is explicit and passed around**, not hidden as
`scaler.mean_` — that is what lets you edit fitted values and reuse them
across backends. The counterpart to `transform` for model code is
**`unscale`**: one verb, correct for affine pipelines and honest for
non-affine ones.

## Install

```text
pip install xscalers            # core (xarray, numpy)
```

## License

MIT
