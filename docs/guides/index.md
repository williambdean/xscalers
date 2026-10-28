# Guides

Hands-on pages for the xscalers workflow, adapted from the README. Every
executable code block runs at docs build time and shows its real output,
so the pages can't drift from the code.

| Page | What it covers |
| --- | --- |
| [The basics](basics.md) | fit / transform / inverse_transform, inspecting state, `FixedScaler` |
| [Mixed grains](mixed-grains.md) | per-variable treatment with `VariableTransformer` |
| [Derived metrics](derived-metrics.md) | `Derive` across and within grains, the non-affine caveat |
| [Dropping variables](dropping.md) | `VariableDropper`, `Selector`, and honest inversion |
| [Splitting variables](splitting.md) | `Splitter` for differently-treated pieces |
| [Saving the fitted state](saving-state.md) | netCDF, zarr-alongside, next to a PyMC model |
| [Converting to real units](unscale.md) | `unscale`, eager and symbolic |
| [Post-fit step access](step-access.md) | `pipe["name"]`, auto-naming, reusing steps |
| [For sklearn users](sklearn-users.md) | the vocabulary mapping and the deliberate departures |
