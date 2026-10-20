# Post-fit step access

A fitted pipeline answers "what did this step actually do" with a bound
step: `pipe["name"]` is the transformer plus its slice of the pipeline
state, so one fitted transformation can be applied to new data without
touching the rest of the pipeline.

```python
from xscalers.datasets import load_national
from xscalers.pipeline import make_pipeline
from xscalers.preprocessing import MeanScaler

pipe = make_pipeline(MeanScaler(dims=("date",)), MeanScaler(dims=("date",)))
assert [name for name, _ in pipe.steps] == ["meanscaler", "meanscaler-1"]
pipe.fit(load_national())                          # sklearn-style fitted pipeline

bound = pipe["meanscaler"]                         # transformer + fitted state slice
scaled_again = bound.transform(load_national())    # one fitted step on fresh data
back = bound.inverse_transform(scaled_again)       # ...and its inverse
```

## Auto-naming

`make_pipeline` names steps from transformer classes (`zscore`,
`zscore-1`, ...), mirroring sklearn. `"passthrough"` is a valid step value:
it fits to an empty state and transforms identity.

## Reusing a fitted step elsewhere

Because a `BoundStep` carries its state, it serializes and moves the same
way any state does ([Saving the fitted state](saving-state.md)): the step's
fitted behavior is data, not a hidden attribute on an object.
