"""provenance(spec): the machine-readable lineage table, config-side.

One pure function over a recipe spec (plain dict, no fitted objects)
answers identity (what was done), affine citizenship (the flag), and
lineage (roles) as a (step, variable) Dataset. Rows exist only for
variables a step touches; absence IS the join for "untouched by".
"""

from __future__ import annotations

import json

import numpy as np

from xscalers import preprocessing as pp
from xscalers.compose import VariableTransformer
from xscalers.core import BaseAffineTransformer, BaseTransformer, as_state, register
from xscalers.pipeline import Pipeline
from xscalers.serialize import provenance


def national_spec():
    pipe = Pipeline(
        [
            (
                "split",
                pp.Splitter(
                    "media",
                    {
                        "direct": {"channel": ["tv", "stream"]},
                        "digital": {"channel": ["search", "social", "fb"]},
                    },
                ),
            ),
            (
                "scale_media",
                VariableTransformer(
                    {
                        "media_direct": pp.FixedScaler(scale=1_000),
                        "media_digital": pp.MeanScaler(dims=("date",)),
                    }
                ),
            ),
            ("scale_controls", VariableTransformer({"controls": pp.ZScore(dims=("date",))})),
        ]
    )
    return pipe.to_spec()


def national_pipe():
    return Pipeline(
        [
            (
                "split",
                pp.Splitter(
                    "media",
                    {
                        "direct": {"channel": ["tv", "stream"]},
                        "digital": {"channel": ["search", "social", "fb"]},
                    },
                ),
            ),
            (
                "scale_media",
                VariableTransformer(
                    {
                        "media_direct": pp.FixedScaler(scale=1_000),
                        "media_digital": pp.MeanScaler(dims=("date",)),
                    }
                ),
            ),
            ("scale_controls", VariableTransformer({"controls": pp.ZScore(dims=("date",))})),
        ]
    )


def derived_spec():
    pipe = Pipeline(
        [
            (
                "derive",
                pp.Derive(
                    name="tv_per_control",
                    left=("media_national", {"media": "tv"}),
                    right=("controls_prod", {}),
                    op="ratio",
                ),
            ),
            ("prune", pp.VariableDropper(vars=["media_national", "controls_prod"])),
            ("log", pp.FunctionTransformer({"tv_per_control": "log"})),
            ("scale", VariableTransformer({"tv_per_control": pp.ZScore(dims=("date",))})),
        ]
    )
    return pipe.to_spec()


class TestTable:
    def test_rows_shaped_step_times_variable(self):
        table = provenance(national_spec())
        assert table.sizes["step"] == 3
        assert {"scale_media", "scale_controls"} <= set(table.step.values)
        assert {"media_direct", "media_digital", "controls"} <= set(table.variable.values)

    def test_transformed_rows_carry_the_recipe(self):
        table = provenance(national_spec())
        row = table.sel(step="scale_media", variable="media_digital")
        assert row["kind"].item() == "MeanScaler"
        assert row["recipe"].item() == "MeanScaler(dims=['date'])"
        assert bool(row["affine"].item()) is True
        assert row["role"].item() == "transformed"

    def test_split_creates_pieces_and_marks_source(self):
        table = provenance(national_spec())
        source = table.sel(step="split", variable="media")
        assert source["role"].item() == "split"
        piece = table.sel(step="split", variable="media_direct")
        assert piece["role"].item() == "created"

    def test_untouched_rows_are_absent(self):
        """scale_media never touches controls: no row for it."""
        table = provenance(national_spec())
        mask = table.recipe.sel(step="scale_controls").values != ""
        step_vars = {str(v) for v in table.variable.values[mask]}
        assert step_vars == {"controls"}

    def test_derived_pipeline_roles(self):
        """derive creates, dropper drops, the log lands op + non-affine."""
        table = provenance(derived_spec())
        created = table.sel(step="derive", variable="tv_per_control")
        assert created["role"].item() == "created"
        assert created["op"].item() == "ratio"
        for dropped in ("media_national", "controls_prod"):
            row = table.sel(step="prune", variable=dropped)
            assert row["role"].item() == "dropped"
        logged = table.sel(step="log", variable="tv_per_control")
        assert logged["op"].item() == "log"
        assert bool(logged["affine"].item()) is False
        assert logged["recipe"].item() == "FunctionTransformer(mapping={'tv_per_control': 'log'})"

    def test_users_make_their_own(self):
        """A user-made registered transformer is flagged affine by protocol
        (it subclasses BaseAffineTransformer, so it HAS affine_view); no
        name-list in serialize.py to extend."""

        @register
        class WaveScaler(BaseAffineTransformer):
            """Divide by a user constant."""

            def to_spec_kwargs(self):
                return {"amp": 3.0}

            def compute(self, X):
                return as_state("wave", self._reduce_dims(X), "", scale=3.0)

        spec = {
            "kind": "Pipeline",
            "kwargs": {
                "steps": [
                    [
                        "s",
                        {
                            "kind": "VariableTransformer",
                            "kwargs": {
                                "mapping": {"x": {"kind": "WaveScaler", "kwargs": {"amp": 3.0}}},
                                "remainder": "passthrough",
                            },
                        },
                    ]
                ]
            },
        }
        table = provenance(spec)
        row = table.sel(step="s", variable="x")
        assert row["kind"].item() == "WaveScaler"
        assert bool(row["affine"].item()) is True

    def test_custom_step_declarare_their_rows(self):
        """Step-shaped customs declare their rows via the describe protocol;
        no kind-branch in provenance to extend."""

        @register
        class Tagger(BaseTransformer):
            """Demo step: tags one variable, changes nothing else."""

            def to_spec_kwargs(self):
                return {"tag": "x"}

            @classmethod
            def describe(cls, spec):
                return [
                    {
                        "variable": spec["tag"],
                        "role": "transformed",
                        "kind": "Tagger",
                        "op": "",
                        "affine": True,
                    }
                ]

        spec = {
            "kind": "Pipeline",
            "kwargs": {
                "steps": [
                    ["s", {"kind": "Tagger", "kwargs": {"tag": "x"}}],
                ]
            },
        }
        table = provenance(spec)
        row = table.sel(step="s", variable="x")
        assert row["kind"].item() == "Tagger"
        assert row["role"].item() == "transformed"
        assert bool(row["affine"].item()) is True

    def test_pure_data_protocol(self):
        """The protocol is over plain dicts: survives a JSON roundtrip."""
        spec = json.loads(json.dumps(national_spec()))
        table = provenance(spec)
        row = table.sel(step="scale_media", variable="media_digital")
        assert row["recipe"].item() == "MeanScaler(dims=['date'])"


class TestComposesThroughDataMovement:
    """Identity affine_views on data-movement steps make the composed pair
    real for the pieces they create; citizenship is one protocol."""

    def test_pipe_affine_works_on_splitter_pipeline(self):
        from xscalers.datasets import load_media_channels

        pipe = national_pipe()
        ds = load_media_channels()
        state = pipe.fit(ds)
        composed = pipe.affine(state)
        assert np.allclose(composed["media_direct"]["scale"].values, 1_000.0)
        # the MeanScaler's per-channel scale carries through the splitter
        assert np.allclose(
            composed["media_digital"]["scale"].values,
            state["scale_media"]["media_digital"]["mean"].values,
        )

    def test_data_movement_rows_flag_by_protocol(self):
        table = provenance(national_spec())
        for variable in ("media", "media_direct"):
            row = table.sel(step="split", variable=variable)
            assert bool(row["affine"].item()) is True
        derived = provenance(derived_spec())
        prune = derived.sel(step="prune", variable="media_national")
        assert bool(prune["affine"].item()) is False  # dropping removes values
