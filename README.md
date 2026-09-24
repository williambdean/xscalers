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
splitting into pieces, scaling each, then recombining:

```python
from xscalers import FixedScaler, MeanScaler, Pipeline, Splitter
from xscalers.dispatch import PerVariable
from xscalers.datasets import load_channel_tree

tree = load_channel_tree()

pipe = Pipeline(
    [
        ("split", Splitter("media", {"digital": {"channel": ["search", "social", "fb"]}})),
        (
            "scale",
            PerVariable(
                {"digital": MeanScaler(dims=("date",)), "controls": FixedScaler(scale=1.0)}
            ),
        ),
    ]
)
state = pipe.fit(tree)
scaled_tree = pipe.transform(tree, state)     # pieces as child nodes media/digital

back = pipe.inverse_transform(scaled_tree, state)   # pieces recombine; coords restored
media = back.children["media"].dataset["media"]
```

The fitted pipeline serializes as *recipe + artifact*:

From your recipe that same catalog config, the fitted pipeline becomes one artifact:

```python
state_tree = pipe.to_state_tree(state)        # xr.DataTree
state_tree.to_netcdf("fitted_state.nc")       # whole pipeline: one artifact
import os; os.remove("fitted_state.nc")       # tidy up after this example
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
