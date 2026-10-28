# Converting to real units: `unscale`

Model code never hand-writes `x * scale + center`: the affine pair is an
*internal* detail. `Pipeline.unscale` applies the composed pair per variable
(eager xarray contributions or symbolic `pytensor.xtensor` in the graph).
When a pipeline is non-affine (e.g. a log-derived metric), `unscale` walks
only *that variable's own* steps backwards, stopping at the step that
created it, so the one call stays honest where hand-written `beta * scale`
algebra goes silently wrong.

## Eager: contributions as xarray

```python exec="on" source="above" html="on" session="unscale"
import numpy as np

import xarray as xr

from xscalers.compose import VariableTransformer
from xscalers.datasets import load_mixed_grain
from xscalers.pipeline import Pipeline
from xscalers.preprocessing import Derive, FixedScaler, FunctionTransformer

xr.set_options(display_expand_data=False)

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

y_real = pipe.unscale(contrib, var="tv_per_control")   # one verb; located = exp(-0.75)
np.testing.assert_allclose(y_real, np.exp(contrib))
y_real
```

## Symbolic: unscale inside the pymc graph

Requires the optional `pytensor` extra (`pip install
"xscalers[pytensor]"`).

```python exec="on" source="above" session="unscale"
# requires-pytensor
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
print(f"symbolic result: {type(spend_real).__name__}")
```

!!! note "Requires pytensor"
    This block runs only with the `pytensor` extra installed; it is skipped
    otherwise.

After sampling, contributions arrive as xarray with `(chain, draw, ...)`,
and the same state broadcasts over those extra dims by name.
