"""A review belongs to a graph execution or to a chat, never to both.

Chat reviews used to be stored under a synthetic ``copilot-session-<id>``
graph execution; these tests pin the chat shape, the graph shape, and the
reading of rows (and callers) still in the old shape.
"""

from uuid import uuid4

import pytest
from prisma.enums import ReviewStatus
from prisma.errors import PrismaError, UniqueViolationError
from prisma.models import PendingHumanReview, User
from pydantic import BaseModel

from backend.data.human_review import (
    check_approval,
    create_auto_approval_record,
    get_or_create_human_review,
    get_pending_reviews_for_chat_session,
    get_pending_reviews_for_execution,
)
from backend.util.json import SafeJson
from backend.util.test import SpinTestServer

pytestmark = pytest.mark.asyncio(loop_scope="session")


@pytest.fixture
async def user_id(server: SpinTestServer):
    user_id = f"review-scope-{uuid4()}"
    try:
        await User.prisma().create(
            data={"id": user_id, "email": f"{user_id}@example.com", "name": "Scope"}
        )
    except UniqueViolationError:
        pass
    yield user_id
    await PendingHumanReview.prisma().delete_many(where={"userId": user_id})
    await User.prisma().delete_many(where={"id": user_id})


async def test_chat_review_has_no_graph_fields_and_is_found_by_its_chat(user_id):
    session_id = f"chat-{uuid4()}"

    await get_or_create_human_review(
        user_id=user_id,
        node_exec_id=f"copilot-node-blk:{uuid4().hex[:8]}",
        chat_session_id=session_id,
        input_data={"path": "/reports"},
        message="Create Folder",
        editable=True,
    )

    [review] = await get_pending_reviews_for_chat_session(session_id, user_id)
    assert review.session_id == session_id
    assert (review.graph_exec_id, review.graph_id, review.graph_version) == (
        None,
        None,
        None,
    )
    assert review.node_id == "copilot-node-blk"
    row = await PendingHumanReview.prisma().find_unique(
        where={"nodeExecId": review.node_exec_id}
    )
    assert row and row.chatSessionId == session_id
    assert row.graphExecId == f"copilot-session-{session_id}"


async def test_graph_review_keeps_its_graph_execution(user_id):
    graph_exec_id = str(uuid4())

    await get_or_create_human_review(
        user_id=user_id,
        node_exec_id=str(uuid4()),
        graph_exec_id=graph_exec_id,
        graph_id="graph-1",
        graph_version=3,
        input_data={"a": 1},
        message="Send",
        editable=False,
    )

    [review] = await get_pending_reviews_for_execution(graph_exec_id, user_id)
    assert (review.graph_exec_id, review.graph_id, review.graph_version) == (
        graph_exec_id,
        "graph-1",
        3,
    )
    assert review.session_id is None


async def test_a_caller_passing_the_old_synthetic_id_writes_a_chat_review(user_id):
    session_id = f"chat-{uuid4()}"

    await get_or_create_human_review(
        user_id=user_id,
        node_exec_id=f"copilot-node-gate-bash_exec:{uuid4().hex[:8]}",
        graph_exec_id=f"copilot-session-{session_id}",
        graph_id=f"copilot-session-{session_id}",
        graph_version=1,
        input_data={},
        message="Run a command",
        editable=False,
    )

    [review] = await get_pending_reviews_for_chat_session(session_id, user_id)
    assert review.graph_exec_id is None
    row = await PendingHumanReview.prisma().find_unique(
        where={"nodeExecId": review.node_exec_id}
    )
    assert row and row.chatSessionId == session_id
    assert row.graphId == f"copilot-session-{session_id}"
    # ...and the same caller finds it again by the id it passed.
    old_id = f"copilot-session-{session_id}"
    found = await get_pending_reviews_for_execution(old_id, user_id)
    assert [r.node_exec_id for r in found] == [review.node_exec_id]


async def test_a_row_in_the_old_shape_still_resolves_to_its_chat(user_id):
    session_id = f"chat-{uuid4()}"
    old_id = f"copilot-session-{session_id}"
    await PendingHumanReview.prisma().create(
        data={
            "nodeExecId": f"copilot-node-blk:{uuid4().hex[:8]}",
            "userId": user_id,
            "graphExecId": old_id,
            "graphId": old_id,
            "graphVersion": 1,
            "payload": SafeJson({}),
            "status": ReviewStatus.WAITING,
        }
    )

    [review] = await get_pending_reviews_for_chat_session(session_id, user_id)
    assert review.session_id == session_id
    assert review.graph_exec_id is None and review.graph_id is None
    assert [
        r.node_exec_id for r in await get_pending_reviews_for_execution(old_id, user_id)
    ] == [review.node_exec_id]


async def test_auto_approval_holds_for_its_chat_only(user_id):
    session_id = f"chat-{uuid4()}"
    node_id = "copilot-node-blk"

    await create_auto_approval_record(
        user_id=user_id, node_id=node_id, payload={}, chat_session_id=session_id
    )

    approved = await check_approval(
        node_exec_id=f"{node_id}:new",
        node_id=node_id,
        user_id=user_id,
        chat_session_id=session_id,
    )
    other_chat = await check_approval(
        node_exec_id=f"{node_id}:new",
        node_id=node_id,
        user_id=user_id,
        chat_session_id=f"chat-{uuid4()}",
    )
    assert approved is not None and approved.status == ReviewStatus.APPROVED
    assert other_chat is None


async def test_a_backfilled_row_reads_as_its_chat(user_id):
    # The migration keeps the legacy graph columns beside the new chatSessionId.
    session_id = f"chat-{uuid4()}"
    old_id = f"copilot-session-{session_id}"
    await PendingHumanReview.prisma().create(
        data={
            "nodeExecId": f"copilot-node-blk:{uuid4().hex[:8]}",
            "userId": user_id,
            "graphExecId": old_id,
            "graphId": old_id,
            "graphVersion": 1,
            "chatSessionId": session_id,
            "payload": SafeJson({}),
            "status": ReviewStatus.WAITING,
        }
    )

    [review] = await get_pending_reviews_for_chat_session(session_id, user_id)
    assert review.session_id == session_id
    assert (review.graph_exec_id, review.graph_id, review.graph_version) == (
        None,
        None,
        None,
    )


async def test_an_auto_approval_made_before_the_migration_still_holds(user_id):
    session_id = f"chat-{uuid4()}"
    node_id = "copilot-node-blk"
    old_id = f"copilot-session-{session_id}"
    await PendingHumanReview.prisma().create(
        data={
            "nodeExecId": f"auto_approve_{old_id}_{node_id}",
            "userId": user_id,
            "graphExecId": old_id,
            "graphId": old_id,
            "graphVersion": 1,
            "chatSessionId": session_id,
            "payload": SafeJson({}),
            "status": ReviewStatus.APPROVED,
            "processed": True,
        }
    )

    approved = await check_approval(
        node_exec_id=f"{node_id}:new",
        node_id=node_id,
        user_id=user_id,
        chat_session_id=session_id,
    )
    assert approved is not None and approved.status == ReviewStatus.APPROVED


class _PreviousDeployReview(BaseModel):
    """The graph fields as the review model before this change required them."""

    node_exec_id: str
    graph_exec_id: str
    graph_id: str
    graph_version: int


async def test_a_new_chat_review_loads_on_the_previous_deploy_too(user_id):
    session_id = f"chat-{uuid4()}"
    await get_or_create_human_review(
        user_id=user_id,
        node_exec_id=f"copilot-node-blk:{uuid4().hex[:8]}",
        chat_session_id=session_id,
        input_data={},
        message="Create Folder",
        editable=True,
    )
    [review] = await get_pending_reviews_for_chat_session(session_id, user_id)
    row = await PendingHumanReview.prisma().find_unique(
        where={"nodeExecId": review.node_exec_id}
    )
    assert row

    _PreviousDeployReview(
        node_exec_id=row.nodeExecId,
        graph_exec_id=row.graphExecId,
        graph_id=row.graphId,
        graph_version=row.graphVersion,
    )
    assert review.session_id == session_id and review.graph_exec_id is None


@pytest.mark.parametrize(
    "scope",
    [
        {},
        {
            "graphExecId": "ge-1",
            "graphId": "g-1",
            "graphVersion": 1,
            "chatSessionId": "s",
        },
    ],
    ids=["neither", "both"],
)
async def test_the_database_refuses_a_review_without_exactly_one_scope(user_id, scope):
    with pytest.raises(PrismaError, match="graph_or_chat_session_check"):
        await PendingHumanReview.prisma().create(
            data={
                "nodeExecId": f"bad-{uuid4()}",
                "userId": user_id,
                "payload": SafeJson({}),
                **scope,
            }
        )
