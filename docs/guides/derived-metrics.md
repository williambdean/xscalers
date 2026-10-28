# Derived metrics

`Derive` turns selected pieces of existing variables into a *new* variable
that later steps treat like anything else. The two pieces do not have to
come from the same variable or the same grain: a national series over a
product-grain control broadcasts up to product grain, so `tv_per_control`
below is a product-grain variable built from a national input.

Ops are a closed, spec-serializable vocabulary, `op="ratio | diff |
product"`, and selector pieces take `reduce="sum | mean | first | last"` for
share-of-total style derivations.

## Across grains

```python exec="on" source="above" html="on" session="derived"
import numpy as np

import xarray as xr

from xscalers.compose import VariableTransformer
from xscalers.datasets import load_mixed_grain
from xscalers.pipeline import Pipeline
from xscalers.preprocessing import Derive, FunctionTransformer, ZScore

mixed: xr.Dataset = load_mixed_grain()
print(f"<p>media_national dims: {mixed['media_national'].dims}</p>")
print(f"<p>controls_prod dims: {mixed['controls_prod'].dims}</p>")

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
print(f"<p>tv_per_control dims: {scaled['tv_per_control'].dims}</p>")

back = pipe.inverse_transform(scaled, state)
np.testing.assert_allclose(back["controls_prod"], mixed["controls_prod"])

scaled["tv_per_control"]
```

Because the z-score reduces over `date` only, each product keeps its own
mean and standard deviation: the per-variable, per-group treatment that
mixed-grain data needs, with no reshaping.

## Within one variable

The same step derives within one variable too, when the variable carries an
extra coordinate dimension. `controls (date, product, control)` with
`control = [level, reference, seasonality]`:

```python exec="on" source="above" html="on" session="derived"
# continues above — `mixed` carries over
from xscalers.preprocessing import Derive

share = Derive(
    name="level_diff",                        # op="diff" this time:
    left=("controls", {"control": "level"}),  # level minus reference,
    right=("controls", {"control": "reference"}),
    op="diff",                                # not a ratio
)
derived = share.transform(mixed, share.fit(mixed))
print(f"<p>level_diff dims: {derived['level_diff'].dims}</p>")

derived["level_diff"]
```

## The non-affine caveat

A log makes the target path non-affine: `pipe.affine()` refuses such
pipelines, and real-terms contributions must go through the batched
`inverse_transform` instead of the `(beta, scale)` shortcut. See
[Converting to real units](unscale.md) for the honest version.
