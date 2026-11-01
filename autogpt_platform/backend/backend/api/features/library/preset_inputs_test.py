"""A preset's stored inputs are bounded where its rows are written."""

import ast
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest

from backend.api.features.library.db import check_preset_input_count, preset_input_rows
from backend.api.features.library.model import LibraryAgentPreset
from backend.data.includes import (
    AGENT_PRESET_INCLUDE,
    MAX_NAMED_INPUTS,
    MAX_NODE_INPUT_OUTPUT_FETCH,
)
from backend.util.exceptions import InvalidInputError

_ROW_TYPE = "AgentNodeExecutionInputOutputCreateWithoutRelationsInput"
_BACKEND_ROOT = Path(__file__).resolve().parents[4] / "backend"


def test_a_preset_at_the_bound_is_written_in_full():
    entries = {f"in_{i}": i for i in range(MAX_NAMED_INPUTS)}

    rows = preset_input_rows(entries)

    assert len(rows) == MAX_NAMED_INPUTS
    assert {row["name"] for row in rows} == set(entries)


def test_a_preset_over_the_bound_is_refused():
    entries = {f"in_{i}": i for i in range(MAX_NAMED_INPUTS + 1)}

    with pytest.raises(InvalidInputError, match="at most"):
        preset_input_rows(entries)


def test_a_read_brings_back_a_preset_at_the_write_bound_in_full():
    # Were the write bound the higher of the two, a stored preset would come
    # back short and from_db would rebuild it with inputs missing.
    fetch_args = cast(dict, AGENT_PRESET_INCLUDE["InputPresets"])
    assert MAX_NAMED_INPUTS <= fetch_args["take"]


def test_every_preset_input_row_is_built_by_the_bounded_builder():
    """A second writer of these rows would skip the bound, so there is one."""
    builders = set()
    for path in _BACKEND_ROOT.rglob("*.py"):
        if path.name.endswith("_test.py") or path.name.startswith("test_"):
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr == _ROW_TYPE
                ):
                    builders.add(
                        (path.relative_to(_BACKEND_ROOT).as_posix(), node.name)
                    )

    assert builders == {("api/features/library/db.py", "preset_input_rows")}


def test_from_db_warns_when_the_stored_rows_hit_the_read_cap(caplog):
    preset = _preset_row(
        [_input_row(f"in_{i}") for i in range(MAX_NODE_INPUT_OUTPUT_FETCH)]
    )

    with caplog.at_level(logging.WARNING, logger="backend.api.features.library.model"):
        result = LibraryAgentPreset.from_db(preset)

    assert any(
        "InputPresets rows hit MAX_NODE_INPUT_OUTPUT_FETCH" in m
        for m in caplog.messages
    )
    assert len(result.inputs) == MAX_NODE_INPUT_OUTPUT_FETCH


def test_from_db_stays_quiet_below_the_read_cap(caplog):
    preset = _preset_row([_input_row("in_0")])

    with caplog.at_level(logging.WARNING, logger="backend.api.features.library.model"):
        LibraryAgentPreset.from_db(preset)

    assert not any("MAX_NODE_INPUT_OUTPUT_FETCH" in m for m in caplog.messages)


def test_the_check_accepts_the_bound_and_refuses_one_more():
    check_preset_input_count({f"in_{i}": i for i in range(MAX_NAMED_INPUTS)})

    with pytest.raises(InvalidInputError):
        check_preset_input_count({f"in_{i}": i for i in range(MAX_NAMED_INPUTS + 1)})


def _input_row(name: str) -> MagicMock:
    row = MagicMock()
    row.name = name
    row.data = "v"
    return row


def _preset_row(input_rows: list) -> MagicMock:
    preset = MagicMock()
    preset.id = "preset-1"
    preset.userId = "u-1"
    preset.createdAt = datetime.now(UTC)
    preset.updatedAt = datetime.now(UTC)
    preset.agentGraphId = "g-1"
    preset.agentGraphVersion = 1
    preset.name = "p"
    preset.description = ""
    preset.isActive = True
    preset.webhookId = None
    preset.Webhook = None
    preset.expertId = None
    preset.organizationId = None
    preset.teamId = None
    preset.InputPresets = input_rows
    return preset
