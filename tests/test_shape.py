"""Shape steps: Splitter (invertible) and Selector (honestly non-invertible)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xscalers import scalers
from xscalers.dispatch import PerVariable
from xscalers.pipeline import Pipeline
from xscalers.core import NonInvertibleError
from xscalers.shape import Selector, Splitter

DIGITAL = ["search", "social", "fb"]
TV = ["tv", "stream"]


def _splitter():
    return Splitter(
        source="media",
        pieces={"digital": {"channel": DIGITAL}, "tv": {"channel": TV}},
    )


def _pipe():
    return Pipeline(
        [
            ("split", _splitter()),
            (
                "scale",
                PerVariable(
                    {
                        "digital": scalers.MeanScaler(dims=("date",)),
                        "tv": scalers.FixedScaler(scale=1_000),
                    }
                ),
            ),
        ]
    )


class TestSplitter:
    def test_fit_records_source_coord_order(self, split_tree):
        state = _splitter().fit(split_tree)
        assert state.attrs["source"] == "media"
        assert state.attrs["split_dim"] == "channel"
        assert set(state.attrs["pieces"]) == {"digital", "tv"}
        # full original coordinate saved as a data var (netCDF-safe), not attrs
        assert list(state["channel_labels"].values) == ["search", "social", "fb", "tv", "stream"]

    def test_transform_replaces_source_with_piece_nodes(self, split_tree):
        tree = _splitter().transform(split_tree, _splitter().fit(split_tree))
        assert set(tree.children["media"].children) == {"digital", "tv"}
        piece = tree.children["media"].children["digital"].dataset["digital"]
        assert list(piece.channel.values) == DIGITAL

    def test_inverse_recombines_and_restores_coord_order(self, split_tree):
        splitter = _splitter()
        state = splitter.fit(split_tree)
        pieces = splitter.transform(split_tree, state)
        back = splitter.inverse_transform(pieces, state)
        media = back.children["media"].dataset["media"]
        assert list(media.channel.values) == ["search", "social", "fb", "tv", "stream"]
        assert np.allclose(
            media.values,
            split_tree.children["media"].dataset["media"].values,
        )

    def test_split_then_pipe_inverse_returns_original(self, split_tree):
        pipe = _pipe()
        state = pipe.fit(split_tree)
        out = pipe.transform(split_tree, state)
        back = pipe.inverse_transform(out, state)
        media = back.children["media"].dataset["media"]
        assert np.allclose(media.values, split_tree.children["media"].dataset["media"].values)

    def test_pipeline_affine_raises_because_of_split(self, split_tree):
        pipe = _pipe()
        state = pipe.fit(split_tree)
        with pytest.raises(TypeError):
            pipe.affine(state)


class TestSelector:
    def test_select_within_a_coordinate(self, split_tree):
        selector = Selector({"media": {"channel": DIGITAL}})
        tree = selector.transform(split_tree, selector.fit(split_tree))
        media = tree.children["media"].dataset["media"]
        assert list(media.channel.values) == DIGITAL

    def test_inverse_raises(self, split_tree):
        selector = Selector({"media": {"channel": DIGITAL}})
        state = selector.fit(split_tree)
        with pytest.raises(NonInvertibleError):
            selector.inverse_transform(split_tree, state)
