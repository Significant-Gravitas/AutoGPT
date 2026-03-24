"""Tests for execute_block, prepare_block_for_execution, and check_hitl_review."""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks._base import BlockType
from backend.copilot.constants import COPILOT_NODE_PREFIX, COPILOT_SESSION_PREFIX
from backend.copilot.tools.helpers import (
    BlockPreparation,
    check_hitl_review,
    execute_block,
    prepare_block_for_execution,
)
from backend.copilot.tools.models import (
    BlockOutputResponse,
    ErrorResponse,
    InputValidationErrorResponse,
    ReviewRequiredResponse,
    SetupRequirementsResponse,
)

_USER = "test-user-helpers"
_SESSION = "test-session-helpers"


def _make_block(block_id: str = "block-1", name: str = "TestBlock"):
    """Create a minimal mock block for execute_block()."""
    mock = MagicMock()
    mock.id = block_id
    mock.name = name
    mock.block_type = BlockType.STANDARD

    mock.input_schema = MagicMock()
    mock.input_schema.get_credentials_fields_info.return_value = {}

    async def _execute(
        input_data: dict, **kwargs: Any
    ) -> AsyncIterator[tuple[str, Any]]:
        yield "result", "ok"

    mock.execute = _execute
    return mock


def _patch_workspace():
    """Patch workspace_db to return a mock workspace."""
    mock_workspace = MagicMock()
    mock_workspace.id = "ws-1"
    mock_ws_db = MagicMock()
    mock_ws_db.get_or_create_workspace = AsyncMock(return_value=mock_workspace)
    return patch("backend.copilot.tools.helpers.workspace_db", return_value=mock_ws_db)


def _patch_credit_db(
    get_credits_return: int = 100,
    spend_credits_side_effect: Any = None,
):
    """Patch credit_db accessor to return a mock credit adapter."""
    mock_credit = MagicMock()
    mock_credit.get_credits = AsyncMock(return_value=get_credits_return)
    if spend_credits_side_effect is not None:
        mock_credit.spend_credits = AsyncMock(side_effect=spend_credits_side_effect)
    else:
        mock_credit.spend_credits = AsyncMock()
    return (
        patch(
            "backend.copilot.tools.helpers.credit_db",
            return_value=mock_credit,
        ),
        mock_credit,
    )


# ---------------------------------------------------------------------------
# Credit charging tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
class TestExecuteBlockCreditCharging:
    async def test_charges_credits_when_cost_is_positive(self):
        """Block with cost > 0 should call spend_credits after execution."""
        block = _make_block()
        credit_patch, mock_credit = _patch_credit_db(get_credits_return=100)

        with (
            _patch_workspace(),
            patch(
                "backend.copilot.tools.helpers.block_usage_cost",
                return_value=(10, {"key": "val"}),
            ),
            credit_patch,
        ):
            result = await execute_block(
                block=block,
                block_id="block-1",
                input_data={"text": "hello"},
                user_id=_USER,
                session_id=_SESSION,
                node_exec_id="exec-1",
                matched_credentials={},
            )

        assert isinstance(result, BlockOutputResponse)
        assert result.success is True
        mock_credit.spend_credits.assert_awaited_once()
        call_kwargs = mock_credit.spend_credits.call_args.kwargs
        assert call_kwargs["cost"] == 10
        assert call_kwargs["metadata"].reason == "copilot_block_execution"

    async def test_returns_error_when_insufficient_credits_before_exec(self):
        """Pre-execution check should return ErrorResponse when balance < cost."""
        block = _make_block()
        credit_patch, mock_credit = _patch_credit_db(get_credits_return=5)

        with (
            _patch_workspace(),
            patch(
                "backend.copilot.tools.helpers.block_usage_cost",
                return_value=(10, {}),
            ),
            credit_patch,
        ):
            result = await execute_block(
                block=block,
                block_id="block-1",
                input_data={},
                user_id=_USER,
                session_id=_SESSION,
                node_exec_id="exec-1",
                matched_credentials={},
            )

        assert isinstance(result, ErrorResponse)
        assert "Insufficient credits" in result.message

    async def test_no_charge_when_cost_is_zero(self):
        """Block with cost 0 should not call spend_credits."""
        block = _make_block()
        credit_patch, mock_credit = _patch_credit_db()

        with (
            _patch_workspace(),
            patch(
                "backend.copilot.tools.helpers.block_usage_cost",
                return_value=(0, {}),
            ),
            credit_patch,
        ):
            result = await execute_block(
                block=block,
                block_id="block-1",
                input_data={},
                user_id=_USER,
                session_id=_SESSION,
                node_exec_id="exec-1",
                matched_credentials={},
            )

        assert isinstance(result, BlockOutputResponse)
        assert result.success is True
        # Credit functions should not be called at all for zero-cost blocks
        mock_credit.get_credits.assert_not_awaited()
        mock_credit.spend_credits.assert_not_awaited()

    async def test_returns_output_on_post_exec_insufficient_balance(self):
        """If charging fails after execution, output is still returned (block already ran)."""
        from backend.util.exceptions import InsufficientBalanceError

        block = _make_block()
        credit_patch, mock_credit = _patch_credit_db(
            get_credits_return=15,
            spend_credits_side_effect=InsufficientBalanceError(
                "Low balance", _USER, 5, 10
            ),
        )

        with (
            _patch_workspace(),
            patch(
                "backend.copilot.tools.helpers.block_usage_cost",
                return_value=(10, {}),
            ),
            credit_patch,
        ):
            result = await execute_block(
                block=block,
                block_id="block-1",
                input_data={},
                user_id=_USER,
                session_id=_SESSION,
                node_exec_id="exec-1",
                matched_credentials={},
            )

        # Block already executed (with side effects), so output is returned
        assert isinstance(result, BlockOutputResponse)
        assert result.success is True


# ---------------------------------------------------------------------------
# Type coercion tests
# ---------------------------------------------------------------------------


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


def _make_coerce_block(
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
    block = _make_coerce_block(
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
    block = _make_coerce_block(
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
    block = _make_coerce_block(
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
    block = _make_coerce_block(
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
    block = _make_coerce_block(
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
    block = _make_coerce_block(
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
    block = _make_coerce_block(
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
    block = _make_coerce_block(
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


# ---------------------------------------------------------------------------
# prepare_block_for_execution tests
# ---------------------------------------------------------------------------

_PREP_USER = "prep-user"
_PREP_SESSION = "prep-session"


def _make_prep_session(session_id: str = _PREP_SESSION) -> MagicMock:
    session = MagicMock()
    session.session_id = session_id
    return session


def _make_simple_block(
    block_id: str = "blk-1",
    name: str = "Simple Block",
    disabled: bool = False,
    required: list[str] | None = None,
    properties: dict[str, Any] | None = None,
) -> MagicMock:
    block = MagicMock()
    block.id = block_id
    block.name = name
    block.disabled = disabled
    block.description = ""
    block.block_type = MagicMock()

    schema = {
        "type": "object",
        "properties": properties or {"text": {"type": "string"}},
        "required": required or [],
    }
    block.input_schema.jsonschema.return_value = schema
    block.input_schema.get_credentials_fields.return_value = {}
    block.input_schema.get_credentials_fields_info.return_value = {}
    return block


def _patch_excluded(block_ids: set | None = None, block_types: set | None = None):
    return (
        patch(
            "backend.copilot.tools.find_block.COPILOT_EXCLUDED_BLOCK_IDS",
            new=block_ids or set(),
            create=True,
        ),
        patch(
            "backend.copilot.tools.find_block.COPILOT_EXCLUDED_BLOCK_TYPES",
            new=block_types or set(),
            create=True,
        ),
    )


@pytest.mark.asyncio
async def test_prepare_block_not_found() -> None:
    excl_ids, excl_types = _patch_excluded()
    with (
        patch("backend.copilot.tools.helpers.get_block", return_value=None),
        excl_ids,
        excl_types,
    ):
        result = await prepare_block_for_execution(
            block_id="missing",
            input_data={},
            user_id=_PREP_USER,
            session=_make_prep_session(),
            session_id=_PREP_SESSION,
        )
    assert isinstance(result, ErrorResponse)
    assert "not found" in result.message


@pytest.mark.asyncio
async def test_prepare_block_disabled() -> None:
    block = _make_simple_block(disabled=True)
    excl_ids, excl_types = _patch_excluded()
    with (
        patch("backend.copilot.tools.helpers.get_block", return_value=block),
        excl_ids,
        excl_types,
    ):
        result = await prepare_block_for_execution(
            block_id="blk-1",
            input_data={},
            user_id=_PREP_USER,
            session=_make_prep_session(),
            session_id=_PREP_SESSION,
        )
    assert isinstance(result, ErrorResponse)
    assert "disabled" in result.message


@pytest.mark.asyncio
async def test_prepare_block_unrecognized_fields() -> None:
    block = _make_simple_block(properties={"text": {"type": "string"}})
    excl_ids, excl_types = _patch_excluded()
    with (
        patch("backend.copilot.tools.helpers.get_block", return_value=block),
        excl_ids,
        excl_types,
        patch(
            "backend.copilot.tools.helpers.resolve_block_credentials",
            AsyncMock(return_value=({}, [])),
        ),
        patch(
            "backend.copilot.tools.helpers.expand_file_refs_in_args",
            AsyncMock(side_effect=lambda d, *a, **kw: d),
        ),
    ):
        result = await prepare_block_for_execution(
            block_id="blk-1",
            input_data={"text": "hi", "unknown_field": "oops"},
            user_id=_PREP_USER,
            session=_make_prep_session(),
            session_id=_PREP_SESSION,
        )
    assert isinstance(result, InputValidationErrorResponse)
    assert "unknown_field" in result.unrecognized_fields


@pytest.mark.asyncio
async def test_prepare_block_missing_credentials() -> None:
    block = _make_simple_block()
    mock_cred = MagicMock()
    excl_ids, excl_types = _patch_excluded()
    with (
        patch("backend.copilot.tools.helpers.get_block", return_value=block),
        excl_ids,
        excl_types,
        patch(
            "backend.copilot.tools.helpers.resolve_block_credentials",
            AsyncMock(return_value=({}, [mock_cred])),
        ),
        patch(
            "backend.copilot.tools.helpers.build_missing_credentials_from_field_info",
            return_value={"cred_key": mock_cred},
        ),
    ):
        result = await prepare_block_for_execution(
            block_id="blk-1",
            input_data={},
            user_id=_PREP_USER,
            session=_make_prep_session(),
            session_id=_PREP_SESSION,
        )
    assert isinstance(result, SetupRequirementsResponse)


@pytest.mark.asyncio
async def test_prepare_block_success_returns_preparation() -> None:
    block = _make_simple_block(
        required=["text"], properties={"text": {"type": "string"}}
    )
    excl_ids, excl_types = _patch_excluded()
    with (
        patch("backend.copilot.tools.helpers.get_block", return_value=block),
        excl_ids,
        excl_types,
        patch(
            "backend.copilot.tools.helpers.resolve_block_credentials",
            AsyncMock(return_value=({}, [])),
        ),
        patch(
            "backend.copilot.tools.helpers.expand_file_refs_in_args",
            AsyncMock(side_effect=lambda d, *a, **kw: d),
        ),
    ):
        result = await prepare_block_for_execution(
            block_id="blk-1",
            input_data={"text": "hello"},
            user_id=_PREP_USER,
            session=_make_prep_session(),
            session_id=_PREP_SESSION,
        )
    assert isinstance(result, BlockPreparation)
    assert result.required_non_credential_keys == {"text"}
    assert result.provided_input_keys == {"text"}


# ---------------------------------------------------------------------------
# check_hitl_review tests
# ---------------------------------------------------------------------------


def _make_hitl_prep(
    block_id: str = "blk-hitl",
    input_data: dict | None = None,
    session_id: str = "hitl-sess",
    needs_review: bool = False,
) -> BlockPreparation:
    block = MagicMock()
    block.id = block_id
    block.name = "HITL Block"
    data = input_data if input_data is not None else {"action": "delete"}
    block.is_block_exec_need_review = AsyncMock(return_value=(needs_review, data))
    return BlockPreparation(
        block=block,
        block_id=block_id,
        input_data=data,
        matched_credentials={},
        input_schema={},
        credentials_fields=set(),
        required_non_credential_keys=set(),
        provided_input_keys=set(),
        synthetic_graph_id=f"{COPILOT_SESSION_PREFIX}{session_id}",
        synthetic_node_id=f"{COPILOT_NODE_PREFIX}{block_id}",
    )


@pytest.mark.asyncio
async def test_check_hitl_no_review_needed() -> None:
    prep = _make_hitl_prep(input_data={"action": "read"}, needs_review=False)
    mock_rdb = MagicMock()
    mock_rdb.get_pending_reviews_for_execution = AsyncMock(return_value=[])

    with patch("backend.copilot.tools.helpers.review_db", return_value=mock_rdb):
        result = await check_hitl_review(prep, "user1", "hitl-sess")

    assert isinstance(result, tuple)
    node_exec_id, returned_data = result
    assert node_exec_id.startswith(f"{COPILOT_NODE_PREFIX}blk-hitl")
    assert returned_data == {"action": "read"}


@pytest.mark.asyncio
async def test_check_hitl_review_required() -> None:
    prep = _make_hitl_prep(input_data={"action": "delete"}, needs_review=True)
    mock_rdb = MagicMock()
    mock_rdb.get_pending_reviews_for_execution = AsyncMock(return_value=[])

    with patch("backend.copilot.tools.helpers.review_db", return_value=mock_rdb):
        result = await check_hitl_review(prep, "user1", "hitl-sess")

    assert isinstance(result, ReviewRequiredResponse)
    assert result.block_id == "blk-hitl"


@pytest.mark.asyncio
async def test_check_hitl_reuses_existing_waiting_review() -> None:
    prep = _make_hitl_prep(input_data={"action": "delete"}, needs_review=False)

    existing = MagicMock()
    existing.node_id = prep.synthetic_node_id
    existing.status.value = "WAITING"
    existing.payload = {"action": "delete"}
    existing.node_exec_id = "existing-review-42"

    mock_rdb = MagicMock()
    mock_rdb.get_pending_reviews_for_execution = AsyncMock(return_value=[existing])

    with patch("backend.copilot.tools.helpers.review_db", return_value=mock_rdb):
        result = await check_hitl_review(prep, "user1", "hitl-sess")

    assert isinstance(result, ReviewRequiredResponse)
    assert result.review_id == "existing-review-42"


@pytest.mark.asyncio
async def test_prepare_block_excluded_by_type() -> None:
    """prepare_block_for_execution returns ErrorResponse for excluded block types."""
    from backend.blocks import BlockType

    block = _make_simple_block()
    block.block_type = BlockType.AGENT

    excl_ids, excl_types = _patch_excluded(block_types={BlockType.AGENT})
    with (
        patch("backend.copilot.tools.helpers.get_block", return_value=block),
        excl_ids,
        excl_types,
    ):
        result = await prepare_block_for_execution(
            block_id="blk-agent",
            input_data={},
            user_id=_PREP_USER,
            session=_make_prep_session(),
            session_id=_PREP_SESSION,
        )
    assert isinstance(result, ErrorResponse)
    assert "cannot be run directly" in result.message


@pytest.mark.asyncio
async def test_prepare_block_excluded_by_id() -> None:
    """prepare_block_for_execution returns ErrorResponse for excluded block IDs."""
    block = _make_simple_block(block_id="blk-excluded")

    excl_ids, excl_types = _patch_excluded(block_ids={"blk-excluded"})
    with (
        patch("backend.copilot.tools.helpers.get_block", return_value=block),
        excl_ids,
        excl_types,
    ):
        result = await prepare_block_for_execution(
            block_id="blk-excluded",
            input_data={},
            user_id=_PREP_USER,
            session=_make_prep_session(),
            session_id=_PREP_SESSION,
        )
    assert isinstance(result, ErrorResponse)
    assert "cannot be run directly" in result.message


@pytest.mark.asyncio
async def test_prepare_block_file_ref_expansion_error() -> None:
    """prepare_block_for_execution returns ErrorResponse when file-ref expansion fails."""
    from backend.copilot.sdk.file_ref import FileRefExpansionError

    block = _make_simple_block(properties={"text": {"type": "string"}})
    excl_ids, excl_types = _patch_excluded()
    with (
        patch("backend.copilot.tools.helpers.get_block", return_value=block),
        excl_ids,
        excl_types,
        patch(
            "backend.copilot.tools.helpers.resolve_block_credentials",
            AsyncMock(return_value=({}, [])),
        ),
        patch(
            "backend.copilot.tools.helpers.expand_file_refs_in_args",
            AsyncMock(
                side_effect=FileRefExpansionError("@@agptfile:missing.txt not found")
            ),
        ),
    ):
        result = await prepare_block_for_execution(
            block_id="blk-1",
            input_data={"text": "@@agptfile:missing.txt"},
            user_id=_PREP_USER,
            session=_make_prep_session(),
            session_id=_PREP_SESSION,
        )
    assert isinstance(result, ErrorResponse)
    assert "file reference" in result.message.lower()
