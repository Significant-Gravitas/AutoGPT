"""A preset's stored inputs, and its page size, are bounded.

`LibraryAgentPreset` reads stored rows through `LibraryAgentPresetCreatable`,
so the bound lives at the write rather than on that model. The page size is
capped in the database layer, so every caller is covered rather than one route.
"""

import pytest

from backend.api.features.library.db import (
    MAX_PRESET_INPUT_BYTES,
    MAX_PRESET_INPUTS,
    MAX_PRESETS_PER_PAGE,
    check_preset_text,
    create_preset,
    list_presets,
    preset_input_rows,
)
from backend.api.features.library.model import LibraryAgentPresetCreatable
from backend.util.exceptions import InvalidInputError
from backend.util.text_limits import MAX_DESCRIPTION_LENGTH, MAX_NAME_LENGTH


def test_inputs_at_the_count_bound_are_accepted():
    rows = preset_input_rows({f"in_{i}": i for i in range(MAX_PRESET_INPUTS)})

    assert len(rows) == MAX_PRESET_INPUTS


def test_inputs_over_the_count_bound_are_refused():
    with pytest.raises(InvalidInputError, match="at most 500 inputs"):
        preset_input_rows({f"in_{i}": i for i in range(MAX_PRESET_INPUTS + 1)})


def test_inputs_over_the_payload_bound_are_refused():
    with pytest.raises(InvalidInputError, match="bytes"):
        preset_input_rows({"essay": "x" * (MAX_PRESET_INPUT_BYTES + 1)})


def test_one_input_under_the_payload_bound_is_accepted():
    rows = preset_input_rows({"essay": "x" * (MAX_PRESET_INPUT_BYTES // 2)})

    assert len(rows) == 1


@pytest.mark.parametrize(
    "field_name, max_length",
    [("name", MAX_NAME_LENGTH), ("description", MAX_DESCRIPTION_LENGTH)],
)
def test_preset_text_over_the_bound_is_refused(field_name: str, max_length: int):
    over = {field_name: "x" * (max_length + 1)}
    at = {field_name: "x" * max_length}

    with pytest.raises(InvalidInputError, match=f"Preset {field_name} is too long"):
        check_preset_text(over.get("name"), over.get("description"))
    check_preset_text(at.get("name"), at.get("description"))


def test_preset_text_may_be_absent():
    check_preset_text(None, None)


@pytest.mark.asyncio
async def test_creating_a_preset_refuses_an_oversized_name():
    """The check is at the write, not only in the helper it calls."""
    preset = LibraryAgentPresetCreatable(
        graph_id="g-1",
        graph_version=1,
        inputs={},
        credentials={},
        name="x" * (MAX_NAME_LENGTH + 1),
        description="",
    )

    # Refused before the first query, so no oversized row can reach the
    # database on any caller's path.
    with pytest.raises(InvalidInputError, match="Preset name is too long"):
        await create_preset("user-1", preset)


@pytest.mark.asyncio
async def test_the_listing_refuses_a_page_bigger_than_its_rows_allow():
    with pytest.raises(InvalidInputError, match="At most 100 presets per page"):
        await list_presets(user_id="user-1", page=1, page_size=MAX_PRESETS_PER_PAGE + 1)
