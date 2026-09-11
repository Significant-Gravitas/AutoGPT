"""Resolved block names are available before execution and setup responses."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import ReviewStatus

from backend.blocks._base import BlockType
from backend.copilot.constants import COPILOT_SESSION_PREFIX
from backend.copilot.model import ChatSession
from backend.copilot.tool_display import tool_display_context
from backend.copilot.tools.continue_run_block import ContinueRunBlockTool
from backend.copilot.tools.models import BlockOutputResponse
from backend.copilot.tools.run_block import RunBlockTool

_BLOCK_ID = "b71fd24c-7623-4a73-a196-0538e436f4b8"
_BLOCK_NAME = "HTTPDownloadBlock"
_USER_ID = "block-display-user"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario,expected_type",
    [
        ("execute", "block_output"),
        ("credentials", "setup_requirements"),
        ("unknown_input", "input_validation_error"),
        ("invalid_schema", "error"),
        ("validate_only", "block_details"),
    ],
)
async def test_run_block_emits_name_before_execution_or_setup(
    scenario: str, expected_type: str
):
    session = ChatSession.new(_USER_ID, dry_run=False)
    block = _block()
    published: list[str] = []
    if scenario == "invalid_schema":
        block.input_schema.jsonschema.side_effect = ValueError("Invalid schema")

    async def resolve_credentials(*args):
        assert published == [_BLOCK_NAME]
        return {}, [MagicMock()] if scenario == "credentials" else []

    execute = AsyncMock(return_value=_output(session))
    with (
        tool_display_context(published.append),
        patch("backend.copilot.tools.helpers.get_block", return_value=block),
        patch(
            "backend.copilot.tools.helpers.resolve_block_credentials",
            new=AsyncMock(side_effect=resolve_credentials),
        ),
        patch(
            "backend.copilot.tools.helpers.expand_file_refs_in_args",
            new=AsyncMock(side_effect=lambda data, *args, **kwargs: data),
        ),
        patch(
            "backend.copilot.tools.run_block.check_hitl_review",
            new=AsyncMock(return_value=("execution-id", {})),
        ),
        patch("backend.copilot.tools.run_block.execute_block", new=execute),
    ):
        result = await RunBlockTool()._execute(
            user_id=_USER_ID,
            session=session,
            block_id=_BLOCK_ID,
            input_data={"unknown": "value"} if scenario == "unknown_input" else {},
            validate_only=scenario == "validate_only",
        )

    assert published == [_BLOCK_NAME]
    assert result.type == expected_type
    if scenario == "execute":
        execute.assert_awaited_once()
    else:
        execute.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("unavailable", ["missing", "disabled", "excluded"])
async def test_unavailable_block_does_not_emit_a_name(unavailable: str):
    session = ChatSession.new(_USER_ID, dry_run=False)
    block = _block()
    block.disabled = unavailable == "disabled"
    if unavailable == "excluded":
        block.block_type = BlockType.INPUT
    published: list[str] = []
    with (
        tool_display_context(published.append),
        patch(
            "backend.copilot.tools.helpers.get_block",
            return_value=None if unavailable == "missing" else block,
        ),
        patch(
            "backend.copilot.tools.helpers.resolve_block_credentials",
            new_callable=AsyncMock,
        ) as resolve,
    ):
        result = await RunBlockTool()._execute(
            user_id=_USER_ID, session=session, block_id=_BLOCK_ID
        )

    assert result.type == "error"
    assert published == []
    resolve.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_credentials", [False, True])
async def test_approved_continuation_emits_name_before_credentials(
    missing_credentials: bool,
):
    session = ChatSession.new(_USER_ID, dry_run=False)
    review_id = f"copilot-node-{_BLOCK_ID}:12345678"
    database = _review_database(session, review_id)
    published: list[str] = []

    async def resolve_credentials(*args):
        assert published == [_BLOCK_NAME]
        return {}, [MagicMock()] if missing_credentials else []

    execute = AsyncMock(return_value=_output(session))
    with (
        tool_display_context(published.append),
        patch(
            "backend.copilot.tools.continue_run_block.review_db", return_value=database
        ),
        patch(
            "backend.copilot.tools.continue_run_block.get_block", return_value=_block()
        ),
        patch(
            "backend.copilot.tools.continue_run_block.resolve_block_credentials",
            new=AsyncMock(side_effect=resolve_credentials),
        ),
        patch("backend.copilot.tools.continue_run_block.execute_block", new=execute),
    ):
        result = await ContinueRunBlockTool()._execute(
            user_id=_USER_ID, session=session, review_id=review_id
        )

    assert published == [_BLOCK_NAME]
    assert result.type == ("error" if missing_credentials else "block_output")
    database.get_reviews_by_node_exec_ids.assert_awaited_once_with(
        [review_id], _USER_ID
    )
    if missing_credentials:
        execute.assert_not_awaited()
        database.delete_review_by_node_exec_id.assert_not_awaited()
    else:
        execute.assert_awaited_once()
        database.delete_review_by_node_exec_id.assert_awaited_once_with(
            review_id, _USER_ID
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "unavailable",
    ["review_missing", "wrong_session", "waiting", "rejected", "block_missing"],
)
async def test_unavailable_continuation_does_not_emit_a_name(unavailable: str):
    session = ChatSession.new(_USER_ID, dry_run=False)
    review_id = f"copilot-node-{_BLOCK_ID}:12345678"
    database = _review_database(session, review_id)
    review = database.get_reviews_by_node_exec_ids.return_value[review_id]
    if unavailable == "review_missing":
        database.get_reviews_by_node_exec_ids.return_value = {}
    elif unavailable == "wrong_session":
        review.graph_exec_id = "another-session"
    elif unavailable in ("waiting", "rejected"):
        review.status = (
            ReviewStatus.WAITING if unavailable == "waiting" else ReviewStatus.REJECTED
        )
    published: list[str] = []
    with (
        tool_display_context(published.append),
        patch(
            "backend.copilot.tools.continue_run_block.review_db", return_value=database
        ),
        patch(
            "backend.copilot.tools.continue_run_block.get_block", return_value=None
        ) as lookup,
        patch(
            "backend.copilot.tools.continue_run_block.resolve_block_credentials",
            new_callable=AsyncMock,
        ) as resolve,
    ):
        result = await ContinueRunBlockTool()._execute(
            user_id=_USER_ID, session=session, review_id=review_id
        )

    assert result.type == "error"
    assert published == []
    resolve.assert_not_awaited()
    if unavailable != "block_missing":
        lookup.assert_not_called()


def _block() -> MagicMock:
    block = MagicMock()
    block.id = _BLOCK_ID
    block.name = _BLOCK_NAME
    block.description = "Downloads HTTP content"
    block.disabled = False
    block.block_type = BlockType.STANDARD
    block.input_schema.get_credentials_fields.return_value = {}
    block.input_schema.get_credentials_fields_info.return_value = {}
    block.input_schema.jsonschema.return_value = {"properties": {}, "required": []}
    block.output_schema.jsonschema.return_value = {"properties": {}}
    return block


def _output(session: ChatSession) -> BlockOutputResponse:
    return BlockOutputResponse(
        message="Completed",
        session_id=session.session_id,
        block_id=_BLOCK_ID,
        block_name=_BLOCK_NAME,
        outputs={"result": ["downloaded"]},
        success=True,
    )


def _review_database(session: ChatSession, review_id: str) -> MagicMock:
    database = MagicMock()
    database.get_reviews_by_node_exec_ids = AsyncMock(
        return_value={
            review_id: MagicMock(
                graph_exec_id=f"{COPILOT_SESSION_PREFIX}{session.session_id}",
                status=ReviewStatus.APPROVED,
                payload={},
            )
        }
    )
    database.delete_review_by_node_exec_id = AsyncMock()
    return database
