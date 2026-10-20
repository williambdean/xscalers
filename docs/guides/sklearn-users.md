# For sklearn users

The vocabulary is deliberately sklearn-shaped:

| sklearn | xscalers | notes |
| --- | --- | --- |
| `fit` / `transform` / `inverse_transform` | same | states are `xr.Dataset`s, not attributes |
| `ColumnTransformer` | `VariableTransformer` | per-variable treatment over named data variables |
| `ColumnDropper` (sklego) | `VariableDropper` | over `drop_vars` / `drop_sel` |
| `Pipeline` | `Pipeline` | named steps, chained states, reversed inverse |
| `make_pipeline` | `make_pipeline` | the composition verb |
| `FeatureUnion` | `make_union` | many pipelines over one dataset, collision policy, covariates-only by design |
| `"passthrough"` | `"passthrough"` | identical semantics |

## The deliberate departure

State is **explicit and passed around**, not hidden as `scaler.mean_`. That
is what lets you edit fitted values, reuse them across backends, and
serialize them with native xarray IO. It is also why `transform` takes the
state as an argument instead of reading private attributes.

## The counterpart to `transform`

For model code the companion verb is **`unscale`**: one call, correct for
affine pipelines and honest for non-affine ones (it refuses where the
algebra does not hold instead of returning a wrong pair).

## What is not here, on purpose

- `>>` / `|` operator chaining: `make_pipeline` / `make_union` are the way
  to compose (Kedro-style chaining reads like imperative execution and
  hides the recipe/state split).
- Aggregation and dedup: stay upstream xarray ops (`resample`, `groupby`,
  `drop_duplicates`), outside the pipeline.
