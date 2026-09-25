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

__all__ = ["Derive", "Selector", "Splitter"]


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

    def fit(self, X: xr.DataTree | xr.Dataset) -> xr.Dataset:
        var = self.source.split("/")[-1]
        if isinstance(X, xr.Dataset):
            labels = np.asarray(X[var][self.split_dim].values)
        elif isinstance(X, xr.DataTree):
            node = _node_of(X, self.source)
            labels = np.asarray(node[var][self.split_dim].values)
        else:
            raise TypeError("Splitter works on Dataset or DataTree inputs")
        entries: dict[str, Any] = {
            f"labels_{self.split_dim}": xr.DataArray(labels, dims=("labels_index",))
        }
        for piece, sel in self.pieces.items():
            values = self._jsonable(sel[self.split_dim])
            entries[f"positions_{piece}"] = xr.DataArray(
                _positions(labels, values), dims=(f"index_{piece}",)
            )
        return as_state(
            "splitter",
            None,
            scaled_from=self.source,
            attrs={"source": self.source, "split_dim": self.split_dim, "pieces": sorted(self.pieces)},
            **entries,
        )

    def transform(self, X: xr.DataTree | xr.Dataset, state: xr.Dataset):
        var = self.source.split("/")[-1]
        if isinstance(X, xr.Dataset):
            # Dataset inputs: pieces become sibling variables named
            # ``{source}_{piece}``. A Dataset cannot hold two variables with
            # disjoint values on the same dim (merge outer-joins), so each
            # piece's split dim is renamed ``{dim}_{piece}`` — ``inverse_
            # transform`` renames back and restores the original coordinates.
            out: dict[str, xr.DataArray] = {}
            for key, da in flatten(X).items():
                if key == var:
                    continue  # replaced by the pieces
                out[key] = da
            for piece, sel in self.pieces.items():
                piece_da = X[var].sel({self.split_dim: list(sel[self.split_dim])})
                piece_da = piece_da.rename({self.split_dim: f"{self.split_dim}_{piece}"})
                out[f"{var}_{piece}"] = piece_da.rename(f"{var}_{piece}")
            return reassemble(X, out)
        if not isinstance(X, xr.DataTree):
            raise TypeError("Splitter works on Dataset or DataTree inputs")
        node = _node_of(X, self.source)
        full = node.dataset[var]
        out = {}
        for key, da in flatten(X).items():
            if key == self.source:
                continue  # replaced by the pieces
            out[key] = da
        for piece, sel in self.pieces.items():
            piece_da = full.sel({self.split_dim: list(sel[self.split_dim])})
            out[f"{self.source}/{piece}"] = piece_da.rename(piece)
        return reassemble(X, out)

    def inverse_transform(self, X: xr.DataTree | xr.Dataset, state: xr.Dataset):
        dim = self.split_dim
        stored = list(state[f"labels_{dim}"].values)
        if isinstance(X, xr.Dataset):
            var = self.source.split("/")[-1]
            pieces = []
            for piece in state.attrs["pieces"]:
                piece_var = f"{var}_{piece}"
                if piece_var not in X.data_vars:
                    continue
                pieces.append(X[piece_var].rename({f"{dim}_{piece}": dim}))
            combined = xr.concat(pieces, dim=dim).reindex({dim: stored})
            out = {
                key: da for key, da in flatten(X).items() if not key.startswith(f"{var}_")
            }
            out[var] = combined.rename(var)
            return reassemble(X, out)
        parent = _node_of(X, self.source)
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
class Derive:
    """Derive a variable from coordinate-referenced pieces of one variable.

    The ratio of two coordinate values within one variable, e.g. a price
    relative to a competitor's price inside ``controls (date, product,
    control)``:

        Derive(
            name="relative_price",
            numerator=("controls", {"control": ["price"]}),
            denominator=("controls", {"control": ["comp_price"]}),
            op="ratio",
        )

    - pieces reference ``(var_name, sel)``: the selector is the same
      ``Splitter``-style label dict, so lists + ``reduce`` (sum | mean |
      first | last) cover share-of-total style derivations, while scalar
      selections pass through directly.
    - ops: ``ratio`` | ``diff`` | ``product`` — a closed named vocabulary,
      because specs must stay plain JSON (no lambdas).
    - stateless: ``fit`` records only the recipe; ``inverse_transform``
      drops the derived variable (its inputs survive untouched), so a
      pipeline containing it still round-trips.
    - Dataset inputs only (DataTree derivation is deferred; the derived var
      needs a flat variable namespace).
    """

    def __init__(
        self,
        name: str,
        numerator: tuple[str, Mapping[str, Any]],
        denominator: tuple[str, Mapping[str, Any]] | None = None,
        op: str = "ratio",
        reduce: str | None = None,
    ):
        if op not in {"ratio", "diff", "product"}:
            raise ValueError(f"Derive ops are ratio | diff | product; got {op!r}")
        if denominator is None:
            raise ValueError(f"Derive op={op!r} needs a denominator piece")
        if reduce is not None and reduce not in {"sum", "mean", "first", "last"}:
            raise ValueError(f"unsupported reduce {reduce!r}; use sum | mean | first | last")
        self.name: str = name
        self.numerator = self._normalize_piece(numerator)
        self.denominator = self._normalize_piece(denominator)
        self.op = op
        self.reduce = reduce

    @staticmethod
    def _normalize_piece(piece: Any) -> tuple[str, dict[str, Any]]:
        """Accept ``(var, sel)`` tuples or serialized ``{"var", "sel"}`` dicts."""
        if isinstance(piece, dict):
            return str(piece["var"]), dict(piece["sel"])
        return str(piece[0]), dict(piece[1])

    def to_spec_kwargs(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "numerator": {"var": self.numerator[0], "sel": self._jsonable_sel(self.numerator[1])},
            "denominator": None if self.denominator is None else {
                "var": self.denominator[0], "sel": self._jsonable_sel(self.denominator[1])
            },
            "op": self.op,
            "reduce": self.reduce,
        }

    @staticmethod
    def _jsonable_sel(sel: Mapping[str, Any]) -> dict[str, Any]:
        return {
            dim: list(value) if isinstance(value, (list, tuple)) else value
            for dim, value in sel.items()
        }

    def to_spec(self) -> dict[str, Any]:
        """Serialize the derive recipe."""
        from .serialize import spec_of

        return spec_of(self)

    # -- pieces --------------------------------------------------------------

    def _piece(self, X: xr.Dataset, piece: tuple[str, Mapping[str, Any]] | None) -> xr.DataArray:
        var, sel = piece if piece is not None else (None, {})
        da = X[var]
        sel_args: dict[str, Any] = {
            str(dim): list(value) if isinstance(value, (list, tuple)) else value
            for dim, value in sel.items()
        }
        for dim, value in sel_args.items():
            da = da.sel(**{dim: value})
        reduce = self.reduce
        needs_reduce = any(
            isinstance(v, (list, tuple)) and len(v) > 1 for v in sel.values()
        )
        if reduce and needs_reduce:
            da = getattr(da, reduce)(dim=list(sel))
        elif reduce is not None and needs_reduce:
            raise ValueError(f"reduce={reduce!r} is not a recognized reducer")
        return da

    # -- transformer API -------------------------------------------------------

    def fit(self, X: Any) -> xr.Dataset:
        if not isinstance(X, xr.Dataset):
            raise TypeError("Derive works on Dataset inputs (DataTree derivation deferred)")
        for piece in (self.numerator, self.denominator):
            if piece is None:
                continue
            var = piece[0]
            if var not in X.data_vars:
                raise KeyError(f"Derive piece references unknown variable {var!r}")
        derived = self.name
        if X is not None and derived in X.data_vars:
            raise ValueError(f"Derived variable name {derived!r} already exists")
        return as_state(
            "derive",
            None,
            scaled_from=None,
            attrs={
                "derived": self.name,
                "op": self.op,
                "reduce": self.reduce,
            },
        )

    def transform(self, X: Any, state: xr.Dataset) -> Any:
        if not isinstance(X, xr.Dataset):
            raise TypeError("Derive works on Dataset inputs")
        first = self._piece(X, self.numerator)
        second = self._piece(X, self.denominator)
        if self.op == "ratio":
            derived = first / second
        elif self.op == "diff":
            derived = first - second
        else:
            derived = first * second
        out = {key: da for key, da in flatten(X).items()}
        out[self.name] = derived.rename(self.name)
        return reassemble(X, out)

    def inverse_transform(self, X: Any, state: xr.Dataset) -> Any:
        out = {key: da for key, da in flatten(X).items() if key != state.attrs["derived"]}
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
