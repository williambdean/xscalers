"""Shape steps — data movement, honestly labeled as such.

``Splitter`` extracts pieces of one variable along a coordinate as DataTree
child nodes (each keeping its own sliced coords), letting a ``PerVariable``
step treat pieces differently; its inverse concats the pieces back and
restores the original coordinate order from the fitted state. DataTree aligns
children with join='exact', so during a split the source var leaves the source
node's dataset — it reemerges on ``inverse_transform``.

``Selector`` selects within coordinates per variable. *Selection destroys
data*, so its ``inverse_transform`` raises rather than guessing — the same
honesty sklearn applies when a transformer can't invert. Aggregation-like
steps stay outside the pipeline (they are upstream xarray ops).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import xarray as xr

from .container import flatten, reassemble
from .core import NonInvertibleError, as_state, register

__all__ = ["Selector", "Splitter"]


def _positions(labels: Any, values: list[Any]) -> np.ndarray:
    lookup = {label: pos for pos, label in enumerate(list(labels))}
    return np.array([lookup[v] for v in values], dtype="int64")


# -- small node helpers used above ----------------------------------------------


def _node_of(X: xr.DataTree, key: str) -> xr.DataTree:
    """Walk key-as-node-path down the tree: ``"media/digital"`` → node that path."""
    node = X
    for part in key.split("/"):
        if not part:
            continue
        if part in node.children:
            node = node.children[part]
    return node


@register
class Splitter:
    """Split one variable along a coordinate into named piece nodes.

    ``pieces`` maps piece name → sel-dict along one dim, e.g.

        Splitter("media", {"digital": {"channel": ["search", "social"]},
                           "tv": {"channel": ["tv"]}})

    The original variable is replaced by child nodes ``media/digital``,
    ``media/tv`` (var renamed to the piece name), so a following
    ``PerVariable`` step scales each piece independently. DataTree requires
    child coords to not conflict with the parent's, so ``transform`` removes
    the source var from the source node's dataset.
    """

    def __init__(self, source: str, pieces: Mapping[str, Mapping[str, Any]]):
        self.source = source
        self.pieces = dict(pieces)
        split_dims = {dim for sel in self.pieces.values() for dim in sel}
        if len(split_dims) != 1:
            raise ValueError(
                f"Splitter splits along exactly one dim; pieces give {sorted(split_dims)}"
            )
        self.split_dim = next(iter(split_dims))

    def to_spec_kwargs(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "pieces": {
                piece: {dim: self._jsonable(value) for dim, value in sel.items()}
                for piece, sel in self.pieces.items()
            },
        }

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if isinstance(value, tuple):
            return list(value)
        return value

    # -- transformer API ---------------------------------------------------

    def fit(self, X: xr.DataTree) -> xr.Dataset:
        if not isinstance(X, xr.DataTree):
            raise TypeError("Splitter works on DataTree inputs")
        node = _node_of(X, self.source)
        dim = self.split_dim
        labels = np.asarray(node[self.source][dim].values)
        entries: dict[str, Any] = {f"labels_{dim}": xr.DataArray(labels, dims=("labels_index",))}
        for piece, sel in self.pieces.items():
            values = self._jsonable(sel[dim])
            entries[f"positions_{piece}"] = xr.DataArray(
                _positions(labels, values), dims=(f"index_{piece}",)
            )
        return as_state(
            "splitter",
            None,
            scaled_from=self.source,
            attrs={"source": self.source, "split_dim": dim, "pieces": sorted(self.pieces)},
            **entries,
        )

    def transform(self, X: xr.DataTree, state: xr.Dataset):
        if not isinstance(X, xr.DataTree):
            raise TypeError("Splitter works on DataTree inputs")
        node = _node_of(X, self.source)
        var = var_of(node)
        full = node[var]
        out: dict[str, xr.DataArray] = {}
        for key, da in flatten(X).items():
            if key == self.source:
                continue  # replaced by the pieces
            out[key] = da
        for piece, sel in self.pieces.items():
            piece_da = full.sel({self.split_dim: list(sel[self.split_dim])})
            out[f"{self.source}/{piece}"] = piece_da.rename(piece)
        return reassemble(X, out)

    def inverse_transform(self, X: xr.DataTree, state: xr.Dataset):
        if not isinstance(X, xr.DataTree):
            raise TypeError("Splitter works on DataTree inputs")
        dim = self.split_dim
        parent = _node_of(X, self.source)
        stored = list(state[f"labels_{dim}"].values)
        pieces_by_order = [
            next(iter(parent.children[piece].dataset.data_vars.values()))
            for piece in state.attrs["pieces"]
        ]
        combined = xr.concat(pieces_by_order, dim=dim).reindex({dim: stored})
        out = {
            key: da
            for key, da in flatten(X).items()
            if not key.startswith(f"{self.source}/")
        }
        out[self.source] = combined.rename(self.source)
        return reassemble(X, out)


@register
class Selector:
    """Select within coordinates per variable. Selection is NOT invertible."""

    def __init__(self, mapping: Mapping[str, Mapping[str, Any]]):
        self.mapping = dict(mapping)

    def to_spec_kwargs(self) -> dict[str, Any]:
        return {"mapping": {k: self._jsonable_sel(v) for k, v in self.mapping.items()}}

    @staticmethod
    def _jsonable_sel(sel: Mapping[str, Any]) -> dict[str, Any]:
        return {
            dim: list(value) if isinstance(value, (list, tuple)) else value
            for dim, value in sel.items()
        }

    def fit(self, X: Any) -> xr.Dataset:
        return as_state("selector", None, scaled_from=None)

    def transform(self, X: Any, state: xr.Dataset) -> Any:
        out = {}
        for key, da in flatten(X).items():
            sel = dict(self.mapping).get(key)
            if sel is not None:
                da = da.sel(**{dim: sel[dim] for dim in sel})
            out[key] = da
        return reassemble(X, out)

    def inverse_transform(self, X: Any, state: xr.Dataset) -> Any:
        raise NonInvertibleError(
            "Selection drops data; inverse_transform cannot restore it. Pass "
            "pieces as separate variables, or use Splitter for a reversible split."
        )


def var_of(node: xr.DataTree) -> str:
    """Sole variable name of a node's dataset (single-var nodes assumed)."""
    vars_ = list(node.dataset.data_vars)
    if len(vars_) != 1:
        raise ValueError(f"Node {node.path!r} must hold exactly one variable; has {vars_}")
    return str(vars_[0])
