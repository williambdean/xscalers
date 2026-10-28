# Splitting one variable into differently-treated pieces

Different treatment *within* one media variable (digital vs TV) works by
splitting into pieces, scaling each, then recombining. Real inputs are a
plain Dataset of variables, so pieces become *sibling variables*
(`media_digital`, `media_linear`), each keeping its slice of the coordinate
under a piece-specific dim name; `inverse_transform` recombines and restores
the original coordinate order.

```python exec="on" source="above" html="on" session="split"
import numpy as np

from xscalers.compose import VariableTransformer
from xscalers.datasets import load_media_channels
from xscalers.pipeline import Pipeline
from xscalers.preprocessing import FixedScaler, MeanScaler, Splitter

ds = load_media_channels()

pipe = Pipeline(
    [
        (
            "split",
            Splitter(
                "media",
                {
                    "digital": {"channel": ["search", "social", "fb"]},
                    "linear": {"channel": ["tv", "stream"]},
                },
            ),
        ),
        (
            "scale",
            VariableTransformer(
                {
                    "media_digital": MeanScaler(dims=("date",)),
                    "media_linear": FixedScaler(scale=1_000),
                },
            ),
        ),
    ]
)
state = pipe.fit(ds)
scaled = pipe.transform(ds, state)
back = pipe.inverse_transform(scaled, state)   # pieces recombine; coords restored
np.testing.assert_allclose(back["media"], ds["media"])

scaled
```

## Why pieces are siblings

A Dataset cannot hold two variables with disjoint values on the same dim
(merge outer-joins), so each piece's split dim is renamed
`{dim}_{piece}`, and `inverse_transform` renames back and restores the
original coordinates. On DataTree inputs, pieces become child nodes under
the source node instead, keeping the same invertibility.

## Splits are affine identities

The split is a value-preserving selection, so the composed affine answer
composes through it: `pipe.affine()` gives each created piece its own
`(center, scale)` pair, with the source variable (gone post-step) getting
no pair.
