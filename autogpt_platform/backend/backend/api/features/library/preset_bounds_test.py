"""A preset's stored inputs, and its page size, are bounded.

`LibraryAgentPreset` reads stored rows through `LibraryAgentPresetCreatable`,
so the bound lives at the write rather than on that model. The page size is
capped in the database layer, so every caller is covered rather than one route.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import prisma.models
import pytest

from backend.api.features.library.db import (
    MAX_PRESET_DESCRIPTION_LENGTH,
    MAX_PRESET_INPUT_BYTES,
    MAX_PRESET_NAME_LENGTH,
    MAX_PRESETS_PER_PAGE,
    check_preset_text,
    create_preset,
    list_presets,
    preset_input_rows,
    update_preset,
)
from backend.api.features.library.model import LibraryAgentPresetCreatable
from backend.data.graph import GRAPH_TEXT_MAX_LENGTHS
from backend.data.includes import MAX_NODE_INPUT_OUTPUT_FETCH
from backend.util.exceptions import InvalidInputError


def test_inputs_over_the_payload_bound_are_refused():
    with pytest.raises(InvalidInputError, match="bytes"):
        preset_input_rows({"essay": "x" * (MAX_PRESET_INPUT_BYTES + 1)})


def test_one_input_under_the_payload_bound_is_accepted():
    rows = preset_input_rows({"essay": "x" * (MAX_PRESET_INPUT_BYTES // 2)})

    assert len(rows) == 1


@pytest.mark.parametrize(
    "field_name, max_length",
    [
        ("name", MAX_PRESET_NAME_LENGTH),
        ("description", MAX_PRESET_DESCRIPTION_LENGTH),
    ],
)
def test_preset_text_over_the_bound_is_refused(field_name: str, max_length: int):
    over = {field_name: "x" * (max_length + 1)}
    at = {field_name: "x" * max_length}

    with pytest.raises(InvalidInputError, match=f"Preset {field_name} is too long"):
        check_preset_text(over.get("name"), over.get("description"))
    check_preset_text(at.get("name"), at.get("description"))


def test_preset_text_may_be_absent():
    check_preset_text(None, None)


@pytest.mark.parametrize(
    "preset_field, graph_field",
    [("name", "name"), ("description", "description")],
)
def test_a_preset_accepts_whatever_its_graph_can_carry(
    preset_field: str, graph_field: str
):
    """`save_as_preset` and the legacy-trigger migration name a preset after
    its graph, so a bound below the graph's would refuse a graph that saves."""
    preset_bound = {
        "name": MAX_PRESET_NAME_LENGTH,
        "description": MAX_PRESET_DESCRIPTION_LENGTH,
    }[preset_field]
    graph_bound = GRAPH_TEXT_MAX_LENGTHS[graph_field]

    assert preset_bound >= graph_bound
    at_the_graph_bound = {preset_field: "x" * graph_bound}
    check_preset_text(
        at_the_graph_bound.get("name"), at_the_graph_bound.get("description")
    )


@pytest.mark.asyncio
async def test_creating_a_preset_refuses_an_oversized_name():
    """The check is at the write, not only in the helper it calls."""
    preset = LibraryAgentPresetCreatable(
        graph_id="g-1",
        graph_version=1,
        inputs={},
        credentials={},
        name="x" * (MAX_PRESET_NAME_LENGTH + 1),
        description="",
    )

    # Refused before the first query, so no oversized row can reach the
    # database on any caller's path.
    with pytest.raises(InvalidInputError, match="Preset name is too long"):
        await create_preset("user-1", preset)


class _ReachedTheCountCheck(Exception):
    pass


@pytest.mark.asyncio
async def test_creating_a_preset_named_after_a_long_graph_is_accepted(mocker):
    mocker.patch(
        "backend.api.features.library.db.check_preset_count",
        side_effect=_ReachedTheCountCheck,
    )
    preset = LibraryAgentPresetCreatable(
        graph_id="g-1",
        graph_version=1,
        inputs={},
        credentials={},
        name="x" * GRAPH_TEXT_MAX_LENGTHS["name"],
        description="y" * GRAPH_TEXT_MAX_LENGTHS["description"],
    )

    # The count check is the next step, so reaching it means the text passed.
    with pytest.raises(_ReachedTheCountCheck):
        await create_preset("user-1", preset)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "inputs, expected",
    [
        (
            {f"in_{i}": i for i in range(MAX_NODE_INPUT_OUTPUT_FETCH + 1)},
            "inputs and credentials together",
        ),
        ({"essay": "x" * (MAX_PRESET_INPUT_BYTES + 1)}, "inputs hold at most"),
    ],
    ids=["too many", "too large"],
)
async def test_updating_a_preset_refuses_oversized_inputs(
    mocker, inputs: dict, expected: str
):
    """The update path writes the same rows, so it carries the same bound."""
    mocker.patch(
        "backend.api.features.library.db.get_preset",
        new=mocker.AsyncMock(
            return_value=mocker.MagicMock(name="preset", expert_id=None)
        ),
    )
    mocker.patch("backend.api.features.library.db.transaction", new=_fake_transaction())
    # The stored-row count the merged update path reads under its row lock,
    # before it builds the new rows; under the read cap, so the refusal below
    # is the input bound rather than that check.
    row_client = mocker.AsyncMock()
    row_client.count.return_value = 1
    mocker.patch.object(
        prisma.models.AgentNodeExecutionInputOutput, "prisma", return_value=row_client
    )

    with pytest.raises(InvalidInputError, match=expected):
        await update_preset(
            user_id="user-1", preset_id="p-1", inputs=inputs, credentials={}
        )

    row_client.delete_many.assert_not_called()


def _fake_transaction():
    """`update_preset` wraps its writes in `transaction()`; this keeps the test
    off a real connection, which would poison the pool for later tests."""

    @asynccontextmanager
    async def _tx():
        yield AsyncMock()

    return _tx


@pytest.mark.asyncio
async def test_the_listing_refuses_a_page_bigger_than_its_rows_allow():
    with pytest.raises(InvalidInputError, match="At most 100 presets per page"):
        await list_presets(user_id="user-1", page=1, page_size=MAX_PRESETS_PER_PAGE + 1)
