# xscalers Development Guide

## Overview

xscalers is a tiny library that defines **what a fitted scaler is** for
xarray-like objects: sklearn's transformer *vocabulary* (`fit`, `transform`,
`inverse_transform`, `Pipeline`) over xarray's objects (`DataArray`,
`Dataset`, `DataTree`), with a pure-arithmetic path into `pytensor.xtensor`
graphs. Nothing custom: every container, every serialization format is a
native ecosystem object.

Source lives in `src/xscalers/`, tests in `tests/`. The library is TDD-built:
the test suite in `tests/` **is** the specification of the main points of
interest — read it before changing behavior.

## Principles

1. **Ecosystem-native objects end-to-end.** States are `xr.Dataset`s
   (values as 0-d or kept-dim data_vars; metadata in `attrs` — values never
   hide in attrs). Pipeline state is an `xr.DataTree` (leaf per variable,
   child per step) serializing with one `to_netcdf`. Never invent containers
   that Dataset/DataTree/InferenceData already cover.
2. **Backend-agnostic core.** Scalers only ever read `state["center"]` /
   `state["scale"]` — no `isinstance(state, Dataset)` branches, no coords in
   core arithmetic, reductions over named dims only. `pytensor.xtensor`
   variables have dims, never coords: anything coord-dependent pytest xtensor
   path skips it is xarray-only.
3. **All v1 scalers are affine.** `y = (x - center) / scale`. The composed
   `affine()` view is the contract consumers (pymc-marketing) bind to — an
   MMM never needs to know which scaler kind produced the pair.
4. **Explicit state, not hidden fit.** Unlike sklearn's `estimator.mean_`,
   the fitted values are passed around and can be inspected or edited. The
   one deliberate sklearn departure; documented in the README.
5. **Recipe / state split.** The recipe serializes to plain-dict specs
   (`to_spec`/`from_spec`, zero-dep); fitted state serializes to netCDF
   artifacts (`state_tree.to_netcdf`). They never mix.
6. **Shape-changing steps are honest.** `Splitter` is invertible (pieces are
   DataTree child nodes; recombine = concat back). `Selector` and `Dropper`
   (and anything aggregation-like) are not — their `inverse_transform` raises
   rather than guessing. Aggregation/dedup stay upstream xarray ops
   (`resample`, `groupby`, `drop_duplicates`), outside the pipeline.

## The contract

```python
class BaseTransformer:
    dims: tuple[str, ...] | None          # dims to reduce over; None = all
    def fit(self, X) -> xr.Dataset        # data-derived state; fixed scalers return constants
    def transform(self, X, state) -> X    # state: Mapping[str, DataArray] (Dataset or dict)
    def inverse_transform(self, X, state) -> X
```

- Inputs: `DataArray` (direct), plus `Dataset` / `DataTree` / `dict` via
  per-variable treatment.
- Per-variable fits keep each variable's own remaining dims — mixed grains
  need no special casing.
- Canonical fitted state across a pipeline: `DataTree` with children per
  step, leaves per variable.
- Keys: canonical learned-state names are `center` and `scale` (z-score's
  `mean`/`std` map onto this affine pair).

## Module layout (mirrors sklearn)

| xscalers | sklearn | holds |
| --- | --- | --- |
| `preprocessing` | `preprocessing` | `ZScore`, `MeanScaler`, `FixedScaler`, `Func`, `Derive`, `Splitter`, `Selector`, `Dropper` |
| `compose` | `compose` | `VariableTransformer` (the ColumnTransformer analog) |
| `pipeline` | `pipeline` | `Pipeline`, `Union`, `make_pipeline`, `make_union` |
| `datasets` | `datasets` | `load_national`, `load_mixed_grain`, `load_media_channels` |
| `core`, `container`, `binding`, `persistence`, `serialize` | `base` + internals | the contract, DataTree/flatten plumbing, xtensor lifting, state IO, specs |

Public names are re-exported from the package root, but README/docs examples
import from the mirrored module the way sklearn users write sklearn.

## Adding a scaler

1. Subclass `BaseAffineTransformer` in `src/xscalers/preprocessing.py`; implement
   `compute(X) -> xr.Dataset` producing `center` / `scale` entries over the
   kept dims (reduce over `self._reduce_dims(X)`).
2. Register with `@register` so `from_spec` can rebuild it.
3. Tests: sklearn parity for the equivalent estimator
   (flatten → compare, pinting `ddof=0` for std), inverse round-trip, affine
   view test. Register the class in the spec registry test if serializable.

## State persistence

- **State is data** — save with native xarray IO: `state.to_netcdf(path)` /
  `xr.open_dataset(path)`; a whole pipeline is one artifact via `state_tree`
  (`xr.open_datatree` on the other side).
- **zarr-alongside**: MMM inputs live in a big zarr store; `append_state(
  tree, store, group="scalers")` appends the state tree as zarr groups
  (upstream gap: `DataTree.to_zarr(group=...)` raises NotImplementedError),
  reading back with `xr.open_datatree(store, group="scalers")`. `mode="a"`
  means a refit *overwrites* the group; name groups per fit for history.
- **pymc**: `az.to_netcdf(idata, "posterior.nc")` and
  `state.to_netcdf("scaler_state.nc")` — state and inference results are
  separate artifacts sharing a directory. The composed `affine()` view is
  what converts in-graph contributions to original scale.
- test philosophy: pytest runs with `filterwarnings = ["error"]` — new
  warnings fail tests; only two documented upstream notices are ignored
  (zarr consolidated metadata, netCDF4 numpy-2.5 binary-compat import note).

## Development

```bash
uv sync --group dev               # includes pytensor so flagship tests run
uv run pytest                     # the red→green suite
uv run pytest tests/test_binding_pytensor.py -v
uv run ruff check src tests
uv run mypy src
uvx prek run --all-files          # hooks (prek), when installed
```

Conventional, informative commit messages; tests live in `tests/`; keep the
flagship cross-backend parity test passing before any refactor.

## Ergonomics

- ``pipe["name"]`` → the fitted bound step (transformer + its state slice) —
  apply one fitted transformation to new data: ``pipe["media"].transform(...)``.
- ``make_pipeline(*steps)`` auto-names from transformer classes
  (``zscore``, ``zscore-1``, ...); ``"passthrough"`` is a valid step value;
  ``VariableTransformer(..., default="drop")`` drops unmapped variables while
  scaling the mapped ones, and ``Dropper(vars=..., labels=...)`` is the
  standalone drop verb (data-movement steps are not affine — see principle 6).

## Deferred (documented, not built)

- pymc-marketing facade over the PR-2831 `VariableScaling` contract
  (`MaxAbsScaling`/`ZScore`/`FixedScaling` delegating to xscalers); recipe in
  config, state as artifact beside InferenceData.
- data_twiller `ScalingPostProcessor` adapter (stacked-scales ↔ Dataset state).
- More scalers (MinMax, MaxAbs, Robust), learnable symbolic *fit*, getitem /
  bound-step views, recipe persistence sugar.
- Symbolic `compute` (reductions inside the graph) — xtensor already has
  `mean/std/min/max`, so this needs no new primitives when a use case appears.
