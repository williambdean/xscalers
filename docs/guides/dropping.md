# Dropping variables and coordinate values

`VariableDropper` is the drop verb: the xarray answer to sklego's
`ColumnDropper`. `vars=` removes whole variables; `labels=` removes values
from a coordinate dataset-wide (the complement of keeping, which is
[Selector](#selector)'s job).

```python
import xarray as xr

from xscalers.datasets import load_mixed_grain
from xscalers.preprocessing import VariableDropper

mixed: xr.Dataset = load_mixed_grain()
prune = VariableDropper(
    vars=["media_national", "controls_prod"],
    labels={"control": ["reference"]},
)
pruned = prune.transform(mixed, prune.fit(mixed))
"media_national" in pruned.data_vars            # False — raw inputs are gone
"reference" in pruned.coords["control"].values  # False — every variable shrank
```

The motivating shape is pruning raw inputs once a derived metric supersedes
them, so later steps never have to enumerate what they don't care about:

```python
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
sorted(model_ready.data_vars)                   # raw inputs pruned, ratio kept
```

## Honesty about dropped data

Dropping destroys data, so `VariableDropper.inverse_transform` raises
instead of guessing, and `pipe.inverse_transform` refuses for the whole
chain. Real-terms conversion is unaffected: `pipe.unscale(contrib,
var="tv_per_control")` still walks that one variable backwards.

`fit` validates rather than failing late (sklego's contract): unknown names
raise `KeyError`, dropping *every* variable raises `ValueError`, and a
non-Dataset input raises `TypeError` (DataTree node drops are deferred).

## Selector

If you need the reverse, a pipeline that scales what it keeps:

- `Splitter` is the invertible way to split one variable into
  differently-treated pieces (see [Splitting](splitting.md));
- `VariableTransformer(..., remainder="drop")` is the leftover-dropping
  sibling.
