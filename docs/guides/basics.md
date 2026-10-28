# The basics

xscalers implements sklearn's transformer vocabulary, `fit` / `transform` /
`inverse_transform`, over xarray objects:

```text
class BaseTransformer:
    dims: tuple[str, ...] | None                       # dims to reduce over; None = all
    def fit(self, X) -> xr.Dataset                     # fitted state, inspectable + editable
    def transform(self, X, state) -> X                 # state: Mapping[str, DataArray]
    def inverse_transform(self, X, state) -> X
```

Dataset and DataTree inputs are handled per variable: each variable keeps its
own remaining dims, which is what makes mixed-grain inputs work with no
special casing.

## Fit and transform

```python exec="on" source="above" html="on" session="basics"
import xarray as xr

from xscalers.datasets import load_national
from xscalers.preprocessing import MeanScaler

national: xr.Dataset = load_national()

scaler = MeanScaler(("date",))
state = scaler.fit(national["media"])
scaled = scaler.transform(national["media"], state)

state
```

The fitted state is a value on the record: inspect it, save it, pass it to
another process. Hand-editing is a manual override (the stamp records the
recipe, not your edits).

```python exec="on" source="above" session="basics"
back = scaler.inverse_transform(scaled, state)

import numpy as np
np.testing.assert_allclose(back, national["media"])   # round-trips exactly
print(f"round-trip max abs error: {np.max(np.abs(back - national['media'])).item():.3g}")
```

## Fixed constants

`FixedScaler` consults no data: it pins the affine pair, so the same object
also serves as a consumer's fixed-scale contract (scale by 1000, divide by a
documented constant, and so on).

```python exec="on" source="above" session="basics"
from xscalers.preprocessing import FixedScaler
import xarray as xr

xr.set_options(display_expand_data=False)

fixed = FixedScaler(scale=1_000_000)
scaled = fixed.transform(national["media"], fixed.fit(national["media"]))
scaled
```

## See also

- [Mixed grains](mixed-grains.md): per-variable treatment in one container
- [Derived metrics](derived-metrics.md): new variables from pieces of others
- [Converting to real units](unscale.md): the `unscale` verb
