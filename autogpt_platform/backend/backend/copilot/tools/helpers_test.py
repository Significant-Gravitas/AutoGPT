"""Tests for execute_block type coercion in helpers.py.

Verifies that execute_block() coerces string input values to match the block's
expected input types, mirroring the executor's validate_exec() logic.
This is critical for @@agptfile: expansion, where file content is always a string
but the block may expect structured types (e.g. list[list[str]]).
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.tools.helpers import execute_block
from backend.copilot.tools.models import BlockOutputResponse


def _make_block_schema(annotations: dict[str, Any]) -> MagicMock:
    """Create a mock input_schema with model_fields matching the given annotations."""
    schema = MagicMock()
    # coerce_inputs_to_schema uses model_fields (Pydantic v2 API)
    model_fields = {}
    for name, ann in annotations.items():
        field = MagicMock()
        field.annotation = ann
        model_fields[name] = field
    schema.model_fields = model_fields
    return schema


def _make_block(
    block_id: str,
    name: str,
    annotations: dict[str, Any],
    outputs: dict[str, list[Any]] | None = None,
) -> MagicMock:
    """Create a mock block with typed annotations and a simple execute method."""
    block = MagicMock()
    block.id = block_id
    block.name = name
    block.input_schema = _make_block_schema(annotations)

    captured_inputs: dict[str, Any] = {}

    async def mock_execute(input_data: dict, **_kwargs: Any):
        captured_inputs.update(input_data)
        for output_name, values in (outputs or {"result": ["ok"]}).items():
            for v in values:
                yield output_name, v

    block.execute = mock_execute
    block._captured_inputs = captured_inputs
    return block


_TEST_SESSION_ID = "test-session-coerce"
_TEST_USER_ID = "test-user-coerce"


@pytest.mark.asyncio(loop_scope="session")
async def test_coerce_json_string_to_nested_list():
    """JSON string → list[list[str]] (Google Sheets CSV import case)."""
    block = _make_block(
        "sheets-write",
        "Google Sheets Write",
        {"values": list[list[str]], "spreadsheet_id": str},
    )

    mock_workspace_db = MagicMock()
    mock_workspace_db.get_or_create_workspace = AsyncMock(
        return_value=MagicMock(id="ws-1")
    )

    with patch(
        "backend.copilot.tools.helpers.workspace_db",
        return_value=mock_workspace_db,
    ):
        response = await execute_block(
            block=block,
            block_id="sheets-write",
            input_data={
                "values": '[["Name","Score"],["Alice","90"],["Bob","85"]]',
                "spreadsheet_id": "abc123",
            },
            user_id=_TEST_USER_ID,
            session_id=_TEST_SESSION_ID,
            node_exec_id="exec-1",
            matched_credentials={},
        )

    assert isinstance(response, BlockOutputResponse)
    assert response.success is True
    # Verify the input was coerced from string to list[list[str]]
    assert block._captured_inputs["values"] == [
        ["Name", "Score"],
        ["Alice", "90"],
        ["Bob", "85"],
    ]
    assert isinstance(block._captured_inputs["values"], list)
    assert isinstance(block._captured_inputs["values"][0], list)


@pytest.mark.asyncio(loop_scope="session")
async def test_coerce_json_string_to_list():
    """JSON string → list[str]."""
    block = _make_block(
        "list-block",
        "List Block",
        {"items": list[str]},
    )

    mock_workspace_db = MagicMock()
    mock_workspace_db.get_or_create_workspace = AsyncMock(
        return_value=MagicMock(id="ws-1")
    )

    with patch(
        "backend.copilot.tools.helpers.workspace_db",
        return_value=mock_workspace_db,
    ):
        response = await execute_block(
            block=block,
            block_id="list-block",
            input_data={"items": '["a","b","c"]'},
            user_id=_TEST_USER_ID,
            session_id=_TEST_SESSION_ID,
            node_exec_id="exec-2",
            matched_credentials={},
        )

    assert isinstance(response, BlockOutputResponse)
    assert block._captured_inputs["items"] == ["a", "b", "c"]


@pytest.mark.asyncio(loop_scope="session")
async def test_coerce_json_string_to_dict():
    """JSON string → dict[str, str]."""
    block = _make_block(
        "dict-block",
        "Dict Block",
        {"config": dict[str, str]},
    )

    mock_workspace_db = MagicMock()
    mock_workspace_db.get_or_create_workspace = AsyncMock(
        return_value=MagicMock(id="ws-1")
    )

    with patch(
        "backend.copilot.tools.helpers.workspace_db",
        return_value=mock_workspace_db,
    ):
        response = await execute_block(
            block=block,
            block_id="dict-block",
            input_data={"config": '{"key": "value", "foo": "bar"}'},
            user_id=_TEST_USER_ID,
            session_id=_TEST_SESSION_ID,
            node_exec_id="exec-3",
            matched_credentials={},
        )

    assert isinstance(response, BlockOutputResponse)
    assert block._captured_inputs["config"] == {"key": "value", "foo": "bar"}


@pytest.mark.asyncio(loop_scope="session")
async def test_no_coercion_when_type_matches():
    """Already-correct types pass through without coercion."""
    block = _make_block(
        "pass-through",
        "Pass Through",
        {"values": list[list[str]], "name": str},
    )

    original_values = [["a", "b"], ["c", "d"]]
    mock_workspace_db = MagicMock()
    mock_workspace_db.get_or_create_workspace = AsyncMock(
        return_value=MagicMock(id="ws-1")
    )

    with patch(
        "backend.copilot.tools.helpers.workspace_db",
        return_value=mock_workspace_db,
    ):
        response = await execute_block(
            block=block,
            block_id="pass-through",
            input_data={"values": original_values, "name": "test"},
            user_id=_TEST_USER_ID,
            session_id=_TEST_SESSION_ID,
            node_exec_id="exec-4",
            matched_credentials={},
        )

    assert isinstance(response, BlockOutputResponse)
    assert block._captured_inputs["values"] == original_values
    assert block._captured_inputs["name"] == "test"


@pytest.mark.asyncio(loop_scope="session")
async def test_coerce_string_to_int():
    """String number → int."""
    block = _make_block(
        "int-block",
        "Int Block",
        {"count": int},
    )

    mock_workspace_db = MagicMock()
    mock_workspace_db.get_or_create_workspace = AsyncMock(
        return_value=MagicMock(id="ws-1")
    )

    with patch(
        "backend.copilot.tools.helpers.workspace_db",
        return_value=mock_workspace_db,
    ):
        response = await execute_block(
            block=block,
            block_id="int-block",
            input_data={"count": "42"},
            user_id=_TEST_USER_ID,
            session_id=_TEST_SESSION_ID,
            node_exec_id="exec-5",
            matched_credentials={},
        )

    assert isinstance(response, BlockOutputResponse)
    assert block._captured_inputs["count"] == 42
    assert isinstance(block._captured_inputs["count"], int)


@pytest.mark.asyncio(loop_scope="session")
async def test_coerce_skips_none_values():
    """None values are not coerced (they may be optional fields)."""
    block = _make_block(
        "optional-block",
        "Optional Block",
        {"data": list[str], "label": str},
    )

    mock_workspace_db = MagicMock()
    mock_workspace_db.get_or_create_workspace = AsyncMock(
        return_value=MagicMock(id="ws-1")
    )

    with patch(
        "backend.copilot.tools.helpers.workspace_db",
        return_value=mock_workspace_db,
    ):
        response = await execute_block(
            block=block,
            block_id="optional-block",
            input_data={"label": "test"},
            user_id=_TEST_USER_ID,
            session_id=_TEST_SESSION_ID,
            node_exec_id="exec-6",
            matched_credentials={},
        )

    assert isinstance(response, BlockOutputResponse)
    # 'data' was not provided, so it should not appear in captured inputs
    assert "data" not in block._captured_inputs


@pytest.mark.asyncio(loop_scope="session")
async def test_coerce_union_type_preserves_valid_member():
    """Union-typed fields should not be coerced when the value matches a member."""
    block = _make_block(
        "union-block",
        "Union Block",
        {"content": str | list[str]},
    )

    mock_workspace_db = MagicMock()
    mock_workspace_db.get_or_create_workspace = AsyncMock(
        return_value=MagicMock(id="ws-1")
    )

    with patch(
        "backend.copilot.tools.helpers.workspace_db",
        return_value=mock_workspace_db,
    ):
        response = await execute_block(
            block=block,
            block_id="union-block",
            input_data={"content": ["a", "b"]},
            user_id=_TEST_USER_ID,
            session_id=_TEST_SESSION_ID,
            node_exec_id="exec-7",
            matched_credentials={},
        )

    assert isinstance(response, BlockOutputResponse)
    # list[str] should NOT be stringified to '["a", "b"]'
    assert block._captured_inputs["content"] == ["a", "b"]
    assert isinstance(block._captured_inputs["content"], list)


@pytest.mark.asyncio(loop_scope="session")
async def test_coerce_inner_elements_of_generic():
    """Inner elements of generic containers are recursively coerced."""
    block = _make_block(
        "inner-coerce",
        "Inner Coerce",
        {"values": list[str]},
    )

    mock_workspace_db = MagicMock()
    mock_workspace_db.get_or_create_workspace = AsyncMock(
        return_value=MagicMock(id="ws-1")
    )

    with patch(
        "backend.copilot.tools.helpers.workspace_db",
        return_value=mock_workspace_db,
    ):
        response = await execute_block(
            block=block,
            block_id="inner-coerce",
            # Inner elements are ints, but target is list[str]
            input_data={"values": [1, 2, 3]},
            user_id=_TEST_USER_ID,
            session_id=_TEST_SESSION_ID,
            node_exec_id="exec-8",
            matched_credentials={},
        )

    assert isinstance(response, BlockOutputResponse)
    # Inner elements should be coerced from int to str
    assert block._captured_inputs["values"] == ["1", "2", "3"]
    assert all(isinstance(v, str) for v in block._captured_inputs["values"])
