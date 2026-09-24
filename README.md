# xscalers

sklearn-shaped, dim-aware scaling transformers for `xarray` and `pytensor.xtensor`

The smaller sibling of the scaling question that stares every MMM at the PyMC
ecosystem in the face: sklearn's transformers don't speak dimensions, and
pymc-marketing's scaling helpers are scattered. xscalers gives a tiny contract
where **fit is eager xarray** (a Dataset you can inspect and edit), and
**transform is pure arithmetic by dimension name** that runs identically on
`DataArray`s and symbolic `pytensor.xtensor` variables. All v1 scalers are
affine, so consumers only bind to a composed `(center, scale)` pair.

## The contract

```python
class BaseTransformer:
    dims: tuple[str, ...] | None                       # dims to reduce over; None = all
    def fit(self, X) -> xr.Dataset                     # fitted state, inspectable + editable
    def transform(self, X, state) -> X                 # state: Mapping[str, DataArray]
    def inverse_transform(self, X, state) -> X
```

The faithful just asks for a DataArray; a Dataset or DataTree is handled by
per-variable dispatch — each variable keeps its own remaining dims, which is
what makes mixed-grain MMM inputs (national media, geo media, product
controls) work with no special-casing.

## The flagship flow: fit on xarray, unscale a pytensor graph

```python
import pytensor.xtensor as ptx
from xscalers import MeanScaler

scaler = MeanScaler(dims=("date",))
state = scaler.fit(mmm_ds["media"])           # eager xarray: state = {"center": 0, "scale": mean(date, media)}

spend_scaled = scaler.transform(mmm_ds["media"], state)   # xarray → into the MMM

contrib = ptx.xtensor("channel_contrib", dims=("date", "channel"))
spend_real = scaler.inverse_transform(contrib, state)      # symbolic: contrib * mean
```

After sampling, contributions arrive as xarray with `(chain, draw, ...)` — the
same state broadcasts over the extra dims by name. The scaling operations and
the transform are pinned by parity tests between more than one backend: the
test `tests/test_binding_pytensor.py::test_symbolic_zscore_forward_matches_xarray` fails if
they ever diverge.

## The basics

| Scaler | fit | semantics |
|---|---|---|
| `FixedScaler(scale=100, center=0)` | none (constants) | divide by a fixed amount — the `scale by 100` case |
| `ZScore(dims=...)` | mean, std (ddof=0, skipna) | sklearn's `StandardScaler` under dim-aware reduction |
| `MeanScaler(dims=...)` | center 0, scale mean | divide-by-mean, the MMM workhorse |

## Pipelines

```python
from xscalers import FixedScaler, MeanScaler, Pipeline, Splitter, ZScore
from xscalers.dispatch import PerVariable

pipe = Pipeline(
    [
        ("split", Splitter("media", {"digital": {"channel": DIGITAL}, "tv": {"channel": TV}})),
        ("scale", PerVariable({
            "digital": ZScore(dims=("date",)),
            "tv": FixedScaler(scale=1_000),
        })),
    ]
)
state = pipe.fit(ds_tree)            # per-node states nested as the input is structured
recipe = pipe.to_spec()              # plain-dict recipe (versionable config)
state.to_netcdf("fitted_state.nc")   # whole fitted pipeline: one artifact (DataTree)
back = pipe.inverse_transform(pipe.transform(ds_tree, state), state)
```

Aggregation/deduplication are deliberately *outside* the pipeline: they are
native xarray (`resample`, `groupby`, `drop_duplicates`) — run them upstream.

## For sklearn users

sklearn vocabulary: `fit/transform/inverse_transform/fit_transform`, and
`Pipeline` composition with named steps. The deliberate departure: **state is
explicit and passed around**, not hidden as `scaler.mean_` — that is what lets
you mutate the fitted values and reuse them across backends. A Dataset
implements `Mapping[str, DataArray]`, so `state["scale"]` is a plain dict
anyway.

## Maturity

v0.1 — contract prototyped via tests; the pymc-marketing facade
(over the `compute/transform/inverse_transform` of PR pymc-labs/pymc-marketing#2831)
and a data_twiller adapter are the planned consumers' side; see `AGENTS.md`
for the deferred list.

## License

MIT
