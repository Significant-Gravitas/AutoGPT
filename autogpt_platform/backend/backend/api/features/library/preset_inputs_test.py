"""A preset's stored inputs are bounded where its rows are written."""

import ast
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import prisma.models
import pytest

from backend.api.features.library import db
from backend.api.features.library.db import preset_input_rows
from backend.api.features.library.model import LibraryAgentPreset
from backend.data.includes import AGENT_PRESET_INCLUDE, MAX_NODE_INPUT_OUTPUT_FETCH
from backend.util.exceptions import InvalidInputError

_ROW_TYPE = "AgentNodeExecutionInputOutputCreateWithoutRelationsInput"
_BUILDER = "preset_input_rows"
_RELATION = "InputPresets"
_BACKEND_ROOT = Path(__file__).resolve().parents[4] / "backend"


def test_a_preset_at_the_bound_is_written_in_full():
    entries = {f"in_{i}": i for i in range(MAX_NODE_INPUT_OUTPUT_FETCH)}

    rows = preset_input_rows(entries)

    assert len(rows) == MAX_NODE_INPUT_OUTPUT_FETCH
    assert {row["name"] for row in rows} == set(entries)


def test_a_preset_over_the_bound_is_refused():
    entries = {f"in_{i}": i for i in range(MAX_NODE_INPUT_OUTPUT_FETCH + 1)}

    with pytest.raises(InvalidInputError, match="at most"):
        preset_input_rows(entries)


def test_a_read_brings_back_a_preset_at_the_write_bound_in_full():
    # A read of fewer rows than a write may store would rebuild a stored preset
    # with inputs missing.
    fetch_args = cast(dict, AGENT_PRESET_INCLUDE[_RELATION])
    assert fetch_args["take"] == MAX_NODE_INPUT_OUTPUT_FETCH


@pytest.mark.asyncio
async def test_update_refuses_to_replace_inputs_past_the_read_cap(mocker):
    """A caller's view of InputPresets came from a read capped at
    MAX_NODE_INPUT_OUTPUT_FETCH, so replacing it from that view would silently
    drop whatever rows the read never surfaced. Past the cap, not at it: the
    read still returns every row of a preset sized exactly to it."""
    mocker.patch.object(
        db, "get_preset", new=AsyncMock(return_value=MagicMock(expert_id=None))
    )

    @asynccontextmanager
    async def fake_tx():
        yield AsyncMock()

    mocker.patch.object(db, "transaction", fake_tx)
    mock_row_client = AsyncMock()
    mock_row_client.count.return_value = MAX_NODE_INPUT_OUTPUT_FETCH + 1
    mocker.patch.object(
        prisma.models.AgentNodeExecutionInputOutput,
        "prisma",
        return_value=mock_row_client,
    )
    mock_preset_client = AsyncMock()
    mocker.patch.object(
        prisma.models.AgentPreset, "prisma", return_value=mock_preset_client
    )

    with pytest.raises(InvalidInputError, match="past what a read returns"):
        await db.update_preset(
            user_id="owner",
            preset_id="preset-1",
            inputs={"a": 1},
            credentials={},
        )

    mock_row_client.delete_many.assert_not_called()
    mock_preset_client.update.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stored_row_count",
    [MAX_NODE_INPUT_OUTPUT_FETCH - 1, MAX_NODE_INPUT_OUTPUT_FETCH],
    ids=["under-the-cap", "at-the-cap"],
)
async def test_update_replaces_inputs_at_or_under_the_read_cap(
    mocker, stored_row_count
):
    """A preset sized exactly to the cap is still fully returned by the read
    (check_preset_input_count allows creating one that size), so it must
    still be editable — not only a preset under it."""
    mocker.patch.object(
        db, "get_preset", new=AsyncMock(return_value=MagicMock(expert_id=None))
    )

    @asynccontextmanager
    async def fake_tx():
        yield AsyncMock()

    mocker.patch.object(db, "transaction", fake_tx)
    mock_row_client = AsyncMock()
    mock_row_client.count.return_value = stored_row_count
    mocker.patch.object(
        prisma.models.AgentNodeExecutionInputOutput,
        "prisma",
        return_value=mock_row_client,
    )
    mock_preset_client = AsyncMock()
    mock_preset_client.update.return_value = MagicMock()
    mocker.patch.object(
        prisma.models.AgentPreset, "prisma", return_value=mock_preset_client
    )
    mocker.patch.object(LibraryAgentPreset, "from_db", return_value=MagicMock())

    await db.update_preset(
        user_id="owner",
        preset_id="preset-1",
        inputs={"a": 1},
        credentials={},
    )

    mock_row_client.delete_many.assert_awaited_once()
    mock_preset_client.update.assert_awaited_once()


def test_every_preset_input_row_is_built_by_the_bounded_builder():
    """A second writer of these rows would skip the bound, so there is one.

    Prisma takes the typed row constructor or a plain dict, so this counts both.
    """
    unbounded = set()
    for path in sorted(_BACKEND_ROOT.rglob("*.py")):
        if path.name.endswith("_test.py") or path.name.startswith("test_"):
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for scope in ast.walk(tree):
            if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if _writes_rows_itself(scope):
                unbounded.add((path.relative_to(_BACKEND_ROOT).as_posix(), scope.name))

    assert unbounded == {("api/features/library/db.py", _BUILDER)}


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


def _writes_rows_itself(scope: ast.AST) -> bool:
    for node in ast.walk(scope):
        if _is_typed_row_call(node) or _is_unbounded_relation_write(node):
            return True
    return False


def _is_typed_row_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == _ROW_TYPE
    )


def _is_unbounded_relation_write(node: ast.AST) -> bool:
    """An `InputPresets=` or `[...]["InputPresets"] =` whose rows we don't build."""
    written = _relation_write_value(node)
    return written is not None and not _is_builder_call(written)


def _relation_write_value(node: ast.AST) -> ast.expr | None:
    """The `create` value of a write to the relation, where there is one."""
    if isinstance(node, ast.keyword) and node.arg == _RELATION:
        return _create_entry(node.value)
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        if (
            isinstance(target, ast.Subscript)
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == _RELATION
        ):
            return _create_entry(node.value)
    if isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and key.value == _RELATION:
                return _create_entry(value)
    return None


def _create_entry(value: ast.expr) -> ast.expr | None:
    if not isinstance(value, ast.Dict):
        return None
    for key, entry in zip(value.keys, value.values):
        if isinstance(key, ast.Constant) and key.value == "create":
            return entry
    return None


def _is_builder_call(value: ast.expr) -> bool:
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == _BUILDER
    )


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
