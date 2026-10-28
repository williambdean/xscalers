"""State persistence — xarray-native, except where zarr needs a workaround.

The general picture: **state is data**, so it saves with plain xarray IO —
``state.to_netcdf(path)`` / ``xr.open_datatree(path)`` for a standalone
artifact, and for pymc workflows the state rides in the same directory as the
``InferenceData`` (``az.to_netcdf(idata, "posterior.nc")``). The recipe is
versioned config (``to_spec``, a plain JSON-ish dict); state is data — never
mix the two.

The one case that is *not* a one-liner today is appending state into an
existing (large) zarr store: ``DataTree.to_zarr(group=...)`` is not
implemented upstream. [`append_state`][xscalers.persistence.append_state]
works around it by writing each
node's Dataset with ``Dataset.to_zarr(..., group=..., mode="a")``; reading
back is native: ``xr.open_datatree(store, group="scalers")``.
"""

from __future__ import annotations

from pathlib import Path

import xarray as xr

__all__ = ["append_state"]


def append_state(
    state: xr.Dataset | xr.DataTree,
    store: str | Path,
    group: str = "scalers",
) -> str | Path:
    """Append a fitted state into an existing zarr store under ``group``.

    A bare Dataset is written directly under ``group``; a DataTree nests one
    child per step (``scalers/scale/<variable>``). ``mode="a"`` means a
    refit overwrites the group (override, not duplicate) — name groups per
    fit (``scalers/2024-02``) if you want history instead.
    """
    if isinstance(state, xr.Dataset):
        state.to_zarr(store, group=group, mode="a")
        return store
    for node in state.subtree:
        data_vars = list(node.dataset.data_vars)
        if not data_vars:
            continue  # empty intermediate nodes materialize from nested groups
        node_group = "/".join(filter(None, (group, node.path.strip("/"))))
        node.dataset.to_zarr(store, group=node_group, mode="a")
    return store
