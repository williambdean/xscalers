# API Reference

The public vocabulary of xscalers, organized the way sklearn mirrors it:

| xscalers | sklearn | holds |
| --- | --- | --- |
| `preprocessing` | `preprocessing` | `ZScore`, `MeanScaler`, `FixedScaler`, `FunctionTransformer`, `Derive`, `Splitter`, `Selector`, `VariableDropper` |
| `compose` | `compose` | `VariableTransformer` (the `ColumnTransformer` analog) |
| `pipeline` | `pipeline` | `Pipeline`, `Union`, `make_pipeline`, `make_union` |
| `datasets` | `datasets` | `load_national`, `load_mixed_grain`, `load_media_channels` |
| `core`, `container`, `binding`, `persistence`, `serialize` | `base` + internals | the contract, DataTree plumbing, xtensor lifting, state IO, specs |

All public names are re-exported from the package root.
