"""Transformers, mirroring ``sklearn.preprocessing``.

Two families live here, both of which sklearn also keeps in
``sklearn.preprocessing``:

- the scalers — ``ZScore``, ``MeanScaler``, ``FixedScaler`` (all affine,
  ``y = (x - center) / scale``) and ``Func`` for named elementwise ops
  (``log``, ``log1p``, ... — the FunctionTransformer analog);
- the steps that build, split or select *variables* rather than rescaling
  them — ``Derive`` (a new variable from selected pieces, the
  PolynomialFeatures analog), ``Splitter`` and ``Selector``.

Everything affine exposes state keys ``center`` and ``scale`` (z-score's
mean/std map onto this pair, matching PR-2831).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import xarray as xr

from .container import flatten, reassemble
from .core import (
    BaseAffineTransformer,
    BaseTransformer,
    NonInvertibleError,
    as_state,
    register,
)

__all__ = [
    "Derive",
    "FixedScaler",
    "Func",
    "MeanScaler",
    "Selector",
    "Splitter",
    "ZScore",
]


_FUNC_INVERSES: dict[str, str] = {
    "log": "exp",
    "log1p": "expm1",
    "sqrt": "square",
    "exp": "log",
    "expm1": "log1p",
    "square": "sqrt",
}


@register
class Func(BaseTransformer):
    """Named elementwise transform per variable — non-affine on purpose.

    The second half of the "ratio, then log" chain: after a ``Derive`` step
    produces a derived metric, apply a named unary function (e.g. ``log``) to
    selected variables. Unmapped variables pass through unchanged.

    Because it is not affine, ``Pipeline.affine()`` refuses pipelines that
    contain it (by design): if the *target* is log-transformed, real-terms
    contributions must go through ``inverse_transform`` — the batched path —
    rather than the ``β·scale`` shortcut.
    """

    def __init__(self, mapping: dict[str, str]):
        unknown = {op for op in mapping.values() if op not in _FUNC_INVERSES}
        if unknown:
            raise ValueError(
                f"Unknown functions {sorted(unknown)}; known: {sorted(_FUNC_INVERSES)}"
            )
        self.mapping = dict(mapping)

    def to_spec_kwargs(self) -> dict[str, Any]:
        return {"mapping": dict(self.mapping)}

    def compute(self, X: Any) -> xr.Dataset:
        name = getattr(X, "name", None)
        scaled_from = str(name) if name is not None else ""
        # the mapping lives on the *recipe*; each per-variable state carries
        # only the scalar op decision for that variable (netCDF-safe)
        return as_state(
            "function",
            None,
            scaled_from,
            attrs={
                # None → "" for netCDF-safety (scalar string attrs only)
                "op": self.mapping.get(scaled_from, ""),
            },
        )

    def _transform(self, X: Any, state: Any) -> Any:
        op = state.attrs.get("op", "")
        if not op:
            return X
        return getattr(np, op)(X)

    def _inverse_transform(self, X: Any, state: Any) -> Any:
        op = state.attrs.get("op", "")
        if not op:
            return X
        return getattr(np, _FUNC_INVERSES[op])(X)


@register
class FixedScaler(BaseAffineTransformer):
    """Constant affine pair — scale by a fixed amount (e.g. scale by 100).

    No data-derived stats: ``compute`` returns constants, so the same object
    works as pymc-marketing's ``FixedScaling``.
    """

    def __init__(self, scale: float = 100.0, center: float = 0.0):
        super().__init__(dims=None)
        self.scale = scale
        self.center = center

    def to_spec_kwargs(self) -> dict[str, Any]:
        return {"scale": self.scale, "center": self.center}

    def compute(self, X: Any) -> xr.Dataset:
        scaled_from = getattr(X, "name", None)
        return as_state(
            "fixed", None, scaled_from, center=self.center, scale=self.scale
        )


class _DataDerivedScaler(BaseAffineTransformer):
    """Shared plumbing: compute affine stats from data over the reduce dims."""

    kind: str

    def to_spec_kwargs(self) -> dict[str, Any]:
        return {"dims": self.dims}

    def _stats(self, da: Any) -> tuple[Any, Any]:
        raise NotImplementedError

    def compute(self, X: Any) -> xr.Dataset:
        scaled_from = getattr(X, "name", None)
        center, scale = self._stats(X)
        return as_state(self.kind, self.dims, scaled_from, center=center, scale=scale)


@register
class ZScore(_DataDerivedScaler):
    """Standardize along the reduce dims: center = mean, scale = std (ddof=0).

    Matches sklearn's ``StandardScaler`` when the kept dim is the feature axis.
    NaNs are skipped in the statistics.
    """

    kind = "zscore"

    def _stats(self, da: Any):
        dims = self._reduce_dims(da)
        center = da.mean(dim=dims, skipna=True)
        scale = da.std(dim=dims, ddof=0, skipna=True)
        return center, scale


@register
class MeanScaler(_DataDerivedScaler):
    """Divide by the mean along the reduce dims: center = 0, scale = mean."""

    kind = "mean"

    def _stats(self, da: Any):
        dims = self._reduce_dims(da)
        center = xr.DataArray(0.0)
        scale = da.mean(dim=dims, skipna=True)
        return center, scale


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
    """Derive a new variable from selected pieces of existing variables.

    Pieces reference ``(var_name, sel)``, so the two sides need not come from
    the same variable or the same grain — xarray broadcasting does the rest.
    A national series over a product-grain control, giving a product-grain
    derived variable:

        Derive(
            name="tv_per_control",
            numerator=("media_national", {"media": "tv"}),
            denominator=("controls_prod", {}),
            op="ratio",
        )

    The within-one-variable shape works the same way, selecting coordinate
    values out of a variable that carries an extra coordinate dimension:

        Derive(
            name="level_ratio",
            numerator=("controls", {"control": "level"}),
            denominator=("controls", {"control": "reference"}),
            op="ratio",
        )

    - selectors are the same ``Splitter``-style label dict, so lists +
      ``reduce`` (sum | mean | first | last) cover share-of-total style
      derivations, while scalar selections pass through directly.
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
                "reduce": self.reduce or "" ,
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
