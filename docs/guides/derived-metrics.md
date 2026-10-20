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

```python
import numpy as np

import xarray as xr

from xscalers.compose import VariableTransformer
from xscalers.datasets import load_mixed_grain
from xscalers.pipeline import Pipeline
from xscalers.preprocessing import Derive, FunctionTransformer, ZScore

mixed: xr.Dataset = load_mixed_grain()
mixed["media_national"].dims                  # (date, media) — national grain
mixed["controls_prod"].dims                   # (date, product) — product grain

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
scaled["tv_per_control"].dims                 # (date, product) — broadcast, then scaled

back = pipe.inverse_transform(scaled, state)
np.testing.assert_allclose(back["controls_prod"], mixed["controls_prod"])
```

Because the z-score reduces over `date` only, each product keeps its own
mean and standard deviation: the per-variable, per-group treatment that
mixed-grain data needs, with no reshaping.

## Within one variable

The same step derives within one variable too, when the variable carries an
extra coordinate dimension. `controls (date, product, control)` with
`control = [level, reference, seasonality]`:

```python
# continues above — `mixed` carries over
from xscalers.preprocessing import Derive

share = Derive(
    name="level_diff",                              # op="diff" this time:
    left=("controls", {"control": "level"}),  # level minus reference,
    right=("controls", {"control": "reference"}),
    op="diff",                                      # not a ratio
)
derived = share.transform(mixed, share.fit(mixed))
derived["level_diff"].dims                       # (date, product)
```

## The non-affine caveat

A log makes the target path non-affine: `pipe.affine()` refuses such
pipelines, and real-terms contributions must go through the batched
`inverse_transform` instead of the `(beta, scale)` shortcut. See
[Converting to real units](unscale.md) for the honest version.
