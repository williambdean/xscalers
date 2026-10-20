"""Container normalization: Dataset / DataTree / dict / DataArray → one dispatch surface.

Internally everything becomes a flat mapping ``key → DataArray``. Dataset keys
are variable names; DataTree keys are node paths (the variable name joins the
key only when a node holds several variables). ``reassemble`` rebuilds the
original container type; new piece nodes created by shape steps appear the
same way. ``state_tree`` nests per-step/per-variable state Datasets into a
DataTree, which is what makes whole-pipeline state a single netCDF artifact.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import xarray as xr

__all__ = ["flatten", "reassemble", "state_from_tree", "state_tree"]


def _node_path_parts(node: xr.DataTree) -> list[str]:
    return [p for p in node.path.split("/") if p]


def flatten(X: Any) -> dict[str, xr.DataArray]:
    """Flatten any accepted container into ``key → DataArray``."""
    if isinstance(X, xr.DataArray):
        return {str(X.name) if X.name is not None else "__array__": X}
    if isinstance(X, xr.Dataset):
        return {str(var): da for var, da in X.data_vars.items()}
    if isinstance(X, xr.DataTree):
        out: dict[str, xr.DataArray] = {}
        for node in X.subtree:
            data_vars = [(str(var), da) for var, da in node.dataset.data_vars.items()]
            if not data_vars:
                continue
            parts = _node_path_parts(node)
            multi = len(data_vars) > 1
            for var, da in data_vars:
                if multi:
                    out["/".join([*parts, var]) if parts else var] = da
                elif parts:
                    out["/".join(parts)] = da
                else:
                    out[var] = da
        return out
    raise TypeError(f"Cannot flatten {type(X).__name__}")


def reassemble(X: Any, out: Mapping[str, xr.DataArray]) -> Any:
    """Rebuild the input container type from transformed flat items."""
    if isinstance(X, xr.DataArray):
        (only,) = out.values()
        return only
    if isinstance(X, xr.Dataset):
        # flat keys are exactly var names; new keys (e.g. splitter pieces)
        # appear alongside kept ones, dropped keys are absent from `out`
        return xr.Dataset(dict(out))
    if isinstance(X, xr.DataTree):
        groups = _groups_from_keys(X, out)
        root = xr.Dataset()
        tree_kwargs = {f"/{path}": xr.Dataset(vars_) for path, vars_ in groups.items()}
        return xr.DataTree.from_dict({"": root, **tree_kwargs})
    raise TypeError(f"Cannot reassemble {type(X).__name__}")


def _groups_from_keys(
    reference: xr.DataTree, out: Mapping[str, xr.DataArray]
) -> dict[str, dict[str, xr.DataArray]]:
    """Group flat keys into ``node path → {var: da}``, reusing reference names."""
    groups: dict[str, dict[str, xr.DataArray]] = {}
    used: set[str] = set()

    for node in reference.subtree:
        data_vars = [str(var) for var in node.dataset.data_vars]
        if not data_vars:
            continue
        parts = _node_path_parts(node)
        path = "/".join(parts)
        for var in data_vars:
            if len(data_vars) > 1:
                key = "/".join([*parts, var]) if parts else var
            else:
                key = path if path else var
            if key in out:
                groups.setdefault(path, {})[var] = out[key]
                used.add(key)

    # remaining keys are new nodes (e.g. split pieces): key "a/b" is the node
    # path itself; the node's var takes the last segment's name
    for key, da in out.items():
        if key in used or key in flatten(reference):
            continue
        groups.setdefault(key, {})[key.split("/")[-1]] = da

    return groups


def state_tree(states: Mapping[str, Any]) -> xr.DataTree:
    """Nest fitted states into a DataTree: child per step, leaf per variable.

    ``states`` is a mapping step name → Dataset, or → another mapping
    (composite states recurse, e.g. a Union's per-branch pipeline states).
    This is what makes whole-pipeline state a single netCDF artifact.
    """
    groups: dict[str, xr.Dataset] = {}

    def _walk(prefix: str, mapping: Mapping[str, Any]) -> None:
        for name, value in mapping.items():
            path = "/".join(filter(None, (prefix, name)))
            if isinstance(value, xr.Dataset):
                groups[path] = value
            elif isinstance(value, Mapping):
                _walk(path, value)
            else:
                raise TypeError(f"Cannot place state at {path!r}: {type(value).__name__}")

    _walk("", states)
    root = xr.Dataset()
    paths = {f"/{k}": v for k, v in groups.items()}
    return xr.DataTree.from_dict({"": root, **paths})


def state_from_tree(tree: xr.DataTree) -> dict[str, Any]:
    """Invert ``state_tree``: a DataTree → nested mapping with Dataset leaves."""
    out: dict[str, Any] = {}
    for node in tree.subtree:
        # attr-only Datasets count too (e.g. shape-step states) — only
        # fully-empty root/intermediates are skipped
        meaningful = bool(list(node.dataset.data_vars)) or bool(node.dataset.attrs)
        if not meaningful or node.path == "/":
            continue
        parts = [p for p in node.path.split("/") if p]
        branch = out
        for part in parts[:-1]:
            branch = branch.setdefault(part, {})
        branch[parts[-1]] = node.dataset
    return out
