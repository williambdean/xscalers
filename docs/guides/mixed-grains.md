# Mixed grains

Dataset and DataTree inputs are handled per variable, so each variable keeps
its own fitted state. Mixed-grain inputs need no special casing: fit a
Dataset and each variable gets its own recipe and stats.

```python exec="on" source="above" html="on" session="mixed"
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
state = transformer.fit(mixed)                # one Dataset state per variable
scaled = transformer.transform(mixed, state)  # each var scaled by its own recipe
mixed
```

## VariableTransformer

`VariableTransformer` is the `ColumnTransformer` analog over named data
variables instead of column positions:

- explicit `mapping` of variable name to transformer;
- `remainder="passthrough"` leaves unknown variables untouched;
- `remainder="drop"` removes them, the leftover-dropping sibling of the
  standalone drop verb ([Dropping](dropping.md)).

## State per variable

Because each variable is fitted independently, the state is a mapping of
variable name to its own `xr.Dataset`. That is what lets a national series
and a per-product series live side by side with separate fitted constants,
and what `Pipeline.affine()` composes into one `(center, scale)` pair per
variable downstream.

```python exec="on" source="above" html="on" session="mixed"
state["media_national"]
```
