"""One approval runs its call once, however many resumes name it.

These run against the real review table, with the block or MCP call mocked.
"""

import asyncio
from uuid import uuid4

import pytest
from prisma.enums import ReviewStatus
from prisma.errors import UniqueViolationError
from prisma.models import PendingHumanReview, User

from backend.blocks._base import BlockType
from backend.copilot.capabilities.mcp_review import COPILOT_MCP_NODE_PREFIX
from backend.copilot.context import set_execution_context
from backend.copilot.tools._test_data import make_session
from backend.copilot.tools.models import (
    BlockOutputResponse,
    ErrorResponse,
    MCPToolOutputResponse,
)
from backend.copilot.tools.resume_capability import ResumeCapabilityTool
from backend.data import human_review
from backend.data.human_review import get_or_create_human_review
from backend.util.test import SpinTestServer

pytestmark = pytest.mark.asyncio(loop_scope="session")

RESUMES = 4


@pytest.fixture
async def user_id(server: SpinTestServer):
    user_id = f"resume-claim-{uuid4()}"
    try:
        await User.prisma().create(
            data={"id": user_id, "email": f"{user_id}@example.com", "name": "Claim"}
        )
    except UniqueViolationError:
        pass
    yield user_id
    await PendingHumanReview.prisma().delete_many(where={"userId": user_id})
    await User.prisma().delete_many(where={"id": user_id})


@pytest.fixture
def session(user_id):
    session = make_session(user_id)
    set_execution_context(user_id, session)
    yield session
    set_execution_context(None, None)


async def _approved_review(user_id: str, session_id: str, review_id: str, payload):
    await get_or_create_human_review(
        user_id=user_id,
        node_exec_id=review_id,
        chat_session_id=session_id,
        input_data=payload,
        message="Send",
        editable=True,
    )
    await PendingHumanReview.prisma().update(
        where={"nodeExecId": review_id}, data={"status": ReviewStatus.APPROVED}
    )


def _all_read_before_any_runs(monkeypatch, parties: int) -> None:
    """Every resume reads the review before any of them goes on."""
    barrier = asyncio.Barrier(parties)
    read = human_review.get_reviews_by_node_exec_ids

    async def read_then_wait(node_exec_ids, user_id):
        reviews = await read(node_exec_ids, user_id)
        await asyncio.wait_for(barrier.wait(), timeout=30)
        return reviews

    monkeypatch.setattr(
        human_review, "get_reviews_by_node_exec_ids", read_then_wait, raising=True
    )


def _block_mocks(monkeypatch, outcome):
    runs: list[dict] = []

    async def execute_block(**kwargs):
        runs.append(kwargs["input_data"])
        return outcome(len(runs))

    class _Block:
        id = "send-email"
        name = "Send Email"
        block_type = BlockType.STANDARD
        disabled = False

    async def no_credentials(*args, **kwargs):
        return {}, {}

    monkeypatch.setattr(
        "backend.copilot.tools.continue_run_block.execute_block", execute_block
    )
    monkeypatch.setattr(
        "backend.copilot.tools.continue_run_block.get_block", lambda _: _Block()
    )
    monkeypatch.setattr(
        "backend.copilot.tools.continue_run_block.resolve_block_credentials",
        no_credentials,
    )
    return runs


def _sent(_: int):
    return BlockOutputResponse(
        message="sent", block_id="b", block_name="Send Email", outputs={}
    )


async def test_concurrent_resumes_of_one_block_approval_run_it_once(
    monkeypatch, user_id, session
):
    review_id = f"copilot-node-send-email:{uuid4().hex[:8]}"
    await _approved_review(user_id, session.session_id, review_id, {"to": "a@b.c"})
    runs = _block_mocks(monkeypatch, _sent)
    _all_read_before_any_runs(monkeypatch, RESUMES)

    results = await asyncio.gather(
        *[
            ResumeCapabilityTool()._execute(user_id, session, review_id=review_id)
            for _ in range(RESUMES)
        ]
    )

    assert len(runs) == 1
    assert sum(isinstance(r, BlockOutputResponse) for r in results) == 1
    refused = [r for r in results if isinstance(r, ErrorResponse)]
    assert len(refused) == RESUMES - 1
    assert all("already executed" in r.message for r in refused)
    assert (
        await PendingHumanReview.prisma().find_unique(where={"nodeExecId": review_id})
        is None
    )


async def test_a_failed_block_run_leaves_the_approval_for_a_retry(
    monkeypatch, user_id, session
):
    review_id = f"copilot-node-send-email:{uuid4().hex[:8]}"
    await _approved_review(user_id, session.session_id, review_id, {"to": "a@b.c"})

    def fail_then_send(attempt: int):
        if attempt == 1:
            return ErrorResponse(message="smtp down", session_id=session.session_id)
        return _sent(attempt)

    runs = _block_mocks(monkeypatch, fail_then_send)
    tool = ResumeCapabilityTool()

    first = await tool._execute(user_id, session, review_id=review_id)
    retry = await tool._execute(user_id, session, review_id=review_id)
    again = await tool._execute(user_id, session, review_id=review_id)

    assert isinstance(first, ErrorResponse)
    assert isinstance(retry, BlockOutputResponse)
    assert isinstance(again, ErrorResponse) and "already executed" in again.message
    assert len(runs) == 2


async def test_concurrent_resumes_of_one_mcp_approval_run_it_once(
    monkeypatch, user_id, session
):
    review_id = f"{COPILOT_MCP_NODE_PREFIX}mcp.example.com:{uuid4().hex[:8]}"
    await _approved_review(
        user_id,
        session.session_id,
        review_id,
        {
            "server_url": "https://mcp.example.com/mcp",
            "tool": "create_issue",
            "arguments": {"title": "x"},
        },
    )
    calls: list[dict] = []

    async def run_mcp(self, user_id, session, **kwargs):
        calls.append(kwargs)
        return MCPToolOutputResponse(
            message="done", server_url=kwargs["server_url"], tool_name="create_issue"
        )

    monkeypatch.setattr(
        "backend.copilot.tools.resume_capability.RunMCPToolTool._execute", run_mcp
    )
    _all_read_before_any_runs(monkeypatch, RESUMES)

    results = await asyncio.gather(
        *[
            ResumeCapabilityTool()._execute(user_id, session, review_id=review_id)
            for _ in range(RESUMES)
        ]
    )

    assert len(calls) == 1
    assert sum(isinstance(r, MCPToolOutputResponse) for r in results) == 1
    refused = [r for r in results if isinstance(r, ErrorResponse)]
    assert len(refused) == RESUMES - 1
    assert all("already executed" in r.message for r in refused)


async def test_a_failed_mcp_call_leaves_the_approval_for_a_retry(
    monkeypatch, user_id, session
):
    review_id = f"{COPILOT_MCP_NODE_PREFIX}mcp.example.com:{uuid4().hex[:8]}"
    await _approved_review(
        user_id,
        session.session_id,
        review_id,
        {
            "server_url": "https://mcp.example.com/mcp",
            "tool": "create_issue",
            "arguments": {"title": "x"},
        },
    )
    calls: list[dict] = []

    async def run_mcp(self, user_id, session, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return ErrorResponse(message="server down", session_id=session.session_id)
        return MCPToolOutputResponse(
            message="done", server_url=kwargs["server_url"], tool_name="create_issue"
        )

    monkeypatch.setattr(
        "backend.copilot.tools.resume_capability.RunMCPToolTool._execute", run_mcp
    )
    tool = ResumeCapabilityTool()

    first = await tool._execute(user_id, session, review_id=review_id)
    retry = await tool._execute(user_id, session, review_id=review_id)

    assert isinstance(first, ErrorResponse)
    assert isinstance(retry, MCPToolOutputResponse)
    assert len(calls) == 2
