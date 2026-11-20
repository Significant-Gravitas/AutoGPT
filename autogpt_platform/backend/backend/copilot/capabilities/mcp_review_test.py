"""An MCP call waiting for review reuses that review only for the same call."""

from unittest.mock import AsyncMock, MagicMock, patch

from backend.copilot.capabilities.mcp_review import (
    MCPReviewPayload,
    mcp_review_node_id,
    open_mcp_review,
)

HOST = "mcp.example.com"
WAITING_ID = f"{mcp_review_node_id(HOST)}:ab12cd34"


def _payload(arguments: dict) -> MCPReviewPayload:
    return MCPReviewPayload(
        server_url=f"https://{HOST}/mcp", tool="delete_thing", arguments=arguments
    )


def _waiting(arguments: dict) -> MagicMock:
    review = MagicMock()
    review.node_id = mcp_review_node_id(HOST)
    review.status.value = "WAITING"
    review.payload = _payload(arguments).model_dump()
    review.node_exec_id = WAITING_ID
    return review


async def _open(arguments: dict, pending: list) -> tuple[str, MagicMock]:
    db = MagicMock()
    db.get_pending_reviews_for_chat_session = AsyncMock(return_value=pending)
    db.get_or_create_human_review = AsyncMock()
    with patch("backend.copilot.capabilities.mcp_review.review_db", return_value=db):
        review_id = await open_mcp_review(
            user_id="u-1",
            session_id="s-1",
            host=HOST,
            payload=_payload(arguments),
            organization_id=None,
            team_id=None,
        )
    return review_id, db


async def test_an_identical_waiting_call_reuses_its_review():
    review_id, db = await _open({"id": 1}, [_waiting({"id": 1})])

    assert review_id == WAITING_ID
    db.get_or_create_human_review.assert_not_awaited()


async def test_a_call_that_differs_only_in_type_gets_its_own_review():
    """Python's ``1 == True``, but the server receives ``true``: the waiting
    card shows ``1``, so the call with ``true`` is not folded into it."""
    review_id, db = await _open({"id": True}, [_waiting({"id": 1})])

    assert review_id != WAITING_ID
    stored = db.get_or_create_human_review.await_args.kwargs["input_data"]
    assert stored["arguments"]["id"] is True
