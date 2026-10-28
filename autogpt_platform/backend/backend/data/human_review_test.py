import datetime
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_mock
from prisma.enums import ReviewStatus

from backend.api.features.experts.spend_approval import spend_review_id
from backend.copilot.constants import step_card_id
from backend.data.human_review import (
    get_or_create_human_review,
    get_pending_reviews_for_execution,
    get_pending_reviews_for_user,
    get_reviews,
    has_pending_reviews_for_graph_exec,
    process_all_reviews_for_execution,
)


@pytest.fixture
def sample_db_review():
    """Create a sample database review object"""
    mock_review = Mock()
    mock_review.nodeExecId = "test_node_123"
    mock_review.userId = "test-user-123"
    mock_review.graphExecId = "test_graph_exec_456"
    mock_review.chatSessionId = None
    mock_review.graphId = "test_graph_789"
    mock_review.graphVersion = 1
    mock_review.payload = {"data": "test payload"}
    mock_review.instructions = "Please review"
    mock_review.editable = True
    mock_review.status = ReviewStatus.WAITING
    mock_review.reviewMessage = None
    mock_review.wasEdited = False
    mock_review.processed = False
    mock_review.createdAt = datetime.datetime.now(datetime.timezone.utc)
    mock_review.updatedAt = None
    mock_review.reviewedAt = None
    return mock_review


@pytest.mark.asyncio(loop_scope="function")
async def test_get_or_create_human_review_new(
    mocker: pytest_mock.MockFixture,
    sample_db_review,
):
    """Test creating a new human review"""
    # Mock the upsert to return a new review (created_at == updated_at)
    sample_db_review.status = ReviewStatus.WAITING
    sample_db_review.processed = False

    mock_prisma = mocker.patch("backend.data.human_review.PendingHumanReview.prisma")
    mock_prisma.return_value.upsert = AsyncMock(return_value=sample_db_review)

    result = await get_or_create_human_review(
        user_id="test-user-123",
        node_exec_id="test_node_123",
        graph_exec_id="test_graph_exec_456",
        graph_id="test_graph_789",
        graph_version=1,
        input_data={"data": "test payload"},
        message="Please review",
        editable=True,
    )

    # Should return None for pending reviews (waiting for human input)
    assert result is None


@pytest.mark.asyncio(loop_scope="function")
async def test_get_or_create_human_review_approved(
    mocker: pytest_mock.MockFixture,
    sample_db_review,
):
    """Test retrieving an already approved review"""
    # Set up review as already approved
    sample_db_review.status = ReviewStatus.APPROVED
    sample_db_review.processed = False
    sample_db_review.reviewMessage = "Looks good"

    mock_prisma = mocker.patch("backend.data.human_review.PendingHumanReview.prisma")
    mock_prisma.return_value.upsert = AsyncMock(return_value=sample_db_review)

    result = await get_or_create_human_review(
        user_id="test-user-123",
        node_exec_id="test_node_123",
        graph_exec_id="test_graph_exec_456",
        graph_id="test_graph_789",
        graph_version=1,
        input_data={"data": "test payload"},
        message="Please review",
        editable=True,
    )

    # Should return the approved result
    assert result is not None
    assert result.status == ReviewStatus.APPROVED
    assert result.data == {"data": "test payload"}
    assert result.message == "Looks good"


@pytest.mark.asyncio(loop_scope="function")
async def test_has_pending_reviews_for_graph_exec_true(
    mocker: pytest_mock.MockFixture,
):
    """Test when there are pending reviews"""
    mock_count = mocker.patch("backend.data.human_review.PendingHumanReview.prisma")
    mock_count.return_value.count = AsyncMock(return_value=2)

    result = await has_pending_reviews_for_graph_exec("test_graph_exec")

    assert result is True


@pytest.mark.asyncio(loop_scope="function")
async def test_has_pending_reviews_for_graph_exec_false(
    mocker: pytest_mock.MockFixture,
):
    """Test when there are no pending reviews"""
    mock_count = mocker.patch("backend.data.human_review.PendingHumanReview.prisma")
    mock_count.return_value.count = AsyncMock(return_value=0)

    result = await has_pending_reviews_for_graph_exec("test_graph_exec")

    assert result is False


@pytest.mark.asyncio(loop_scope="function")
async def test_get_pending_reviews_for_user(
    mocker: pytest_mock.MockFixture,
    sample_db_review,
):
    """Test getting pending reviews for a user with pagination"""
    mock_prisma = mocker.patch("backend.data.human_review.PendingHumanReview.prisma")
    mock_prisma.return_value.count = AsyncMock(return_value=11)
    # The chat cards answering steps (none), then the page.
    mock_prisma.return_value.find_many = AsyncMock(side_effect=[[], [sample_db_review]])

    # Mock the batched node_id resolution (AgentNodeExecution.prisma().find_many)
    mock_node_exec = Mock()
    mock_node_exec.id = "test_node_123"
    mock_node_exec.agentNodeId = "test_node_def_789"
    mock_node_exec_prisma = mocker.patch(
        "backend.data.human_review.AgentNodeExecution.prisma"
    )
    mock_node_exec_prisma.return_value.find_many = AsyncMock(
        return_value=[mock_node_exec]
    )

    # Mock the batched graph-execution lookup empty, so the enrichment step's
    # dependent LibraryAgent/ChatSession queries are skipped entirely.
    mock_graph_exec_prisma = mocker.patch(
        "backend.data.human_review.AgentGraphExecution.prisma"
    )
    mock_graph_exec_prisma.return_value.find_many = AsyncMock(return_value=[])

    result = await get_pending_reviews_for_user("test_user", page=2, page_size=10)

    assert len(result) == 1
    assert result[0].node_exec_id == "test_node_123"
    assert result[0].node_id == "test_node_def_789"

    # Verify pagination parameters
    call_args = mock_prisma.return_value.find_many.call_args_list[1]
    assert call_args.kwargs["skip"] == 10  # (page-1) * page_size = (2-1) * 10
    assert call_args.kwargs["take"] == 10


@pytest.mark.asyncio(loop_scope="function")
async def test_an_experts_waiting_spend_approvals_group_as_one_node(
    mocker: pytest_mock.MockFixture,
):
    # One node per expert gives the chat one Approve for all her paused runs.
    def spend_row(graph_exec_id: str) -> Mock:
        row = Mock()
        row.nodeExecId = spend_review_id("e-1", graph_exec_id)
        row.userId = "user-1"
        row.graphExecId = graph_exec_id
        row.chatSessionId = None
        row.graphId = "graph-1"
        row.graphVersion = 1
        row.payload = {"expert": "Maria"}
        row.instructions = "Maria needs your approval to keep spending"
        row.editable = False
        row.status = ReviewStatus.WAITING
        row.reviewMessage = None
        row.wasEdited = False
        row.processed = False
        row.createdAt = datetime.datetime.now(datetime.UTC)
        row.updatedAt = None
        row.reviewedAt = None
        return row

    prisma = mocker.patch("backend.data.human_review.PendingHumanReview.prisma")
    prisma.return_value.count = AsyncMock(return_value=2)
    prisma.return_value.find_many = AsyncMock(
        side_effect=[[], [spend_row("exec-1"), spend_row("exec-2")]]
    )
    node_execs = mocker.patch("backend.data.human_review.AgentNodeExecution.prisma")
    node_execs.return_value.find_many = AsyncMock(return_value=[])
    graph_execs = mocker.patch("backend.data.human_review.AgentGraphExecution.prisma")
    graph_execs.return_value.find_many = AsyncMock(return_value=[])

    reviews = await get_pending_reviews_for_user("user-1")

    assert [r.node_id for r in reviews] == ["expert-spend:e-1", "expert-spend:e-1"]
    assert {r.graph_exec_id for r in reviews} == {"exec-1", "exec-2"}


@pytest.mark.parametrize("named", ["its own step", "another step"])
@pytest.mark.asyncio(loop_scope="function")
async def test_a_step_card_hides_only_the_step_it_was_opened_for(
    mocker: pytest_mock.MockFixture, named
):
    card = Mock()
    card.nodeExecId = step_card_id("user-1", "step-1")
    card.payload = {
        "workflow": {"node_exec_id": "step-1" if named == "its own step" else "step-2"}
    }
    prisma = mocker.patch("backend.data.human_review.PendingHumanReview.prisma")
    prisma.return_value.count = AsyncMock(return_value=0)
    prisma.return_value.find_many = AsyncMock(side_effect=[[card], []])

    await get_pending_reviews_for_user("user-1")

    page = prisma.return_value.find_many.call_args_list[1].kwargs["where"]
    hidden = {"not_in": ["step-1"]} if named == "its own step" else None
    assert page.get("nodeExecId") == hidden


@pytest.mark.asyncio(loop_scope="function")
async def test_the_external_api_lists_a_step_whose_card_waits_in_a_chat(
    mocker: pytest_mock.MockFixture,
):
    # Its listing leaves chat cards out, so the step's own row is the one an API
    # client can see and answer; answering it closes the card.
    prisma = mocker.patch("backend.data.human_review.PendingHumanReview.prisma")
    prisma.return_value.count = AsyncMock(return_value=0)
    prisma.return_value.find_many = AsyncMock(return_value=[])

    await get_reviews("user-1", status=ReviewStatus.WAITING, graph_runs_only=True)

    (page,) = prisma.return_value.find_many.call_args_list
    assert "nodeExecId" not in page.kwargs["where"]


@pytest.mark.asyncio(loop_scope="function")
async def test_get_pending_reviews_for_execution(
    mocker: pytest_mock.MockFixture,
    sample_db_review,
):
    """Test getting pending reviews for specific execution"""
    mock_prisma = mocker.patch("backend.data.human_review.PendingHumanReview.prisma")
    mock_prisma.return_value.count = AsyncMock(return_value=1)
    mock_prisma.return_value.find_many = AsyncMock(return_value=[sample_db_review])

    # Mock the batched node_id resolution (AgentNodeExecution.prisma().find_many)
    mock_node_exec = Mock()
    mock_node_exec.id = "test_node_123"
    mock_node_exec.agentNodeId = "test_node_def_789"
    mock_node_exec_prisma = mocker.patch(
        "backend.data.human_review.AgentNodeExecution.prisma"
    )
    mock_node_exec_prisma.return_value.find_many = AsyncMock(
        return_value=[mock_node_exec]
    )
    mock_graph_exec_prisma = mocker.patch(
        "backend.data.human_review.AgentGraphExecution.prisma"
    )
    mock_graph_exec_prisma.return_value.find_many = AsyncMock(return_value=[])

    result = await get_pending_reviews_for_execution(
        "test_graph_exec_456", "test-user-123"
    )

    assert len(result) == 1
    assert result[0].graph_exec_id == "test_graph_exec_456"
    assert result[0].node_id == "test_node_def_789"

    # Verify it filters by execution and user
    call_args = mock_prisma.return_value.find_many.call_args
    where_clause = call_args.kwargs["where"]
    assert where_clause["userId"] == "test-user-123"
    assert where_clause["graphExecId"] == "test_graph_exec_456"
    assert where_clause["status"] == ReviewStatus.WAITING


@pytest.mark.asyncio(loop_scope="function")
async def test_process_all_reviews_for_execution_success(
    mocker: pytest_mock.MockFixture,
    sample_db_review,
):
    """Test successful processing of reviews for an execution"""
    # Mock finding reviews
    mock_prisma = mocker.patch("backend.data.human_review.PendingHumanReview.prisma")
    mock_prisma.return_value.find_many = AsyncMock(return_value=[sample_db_review])

    # Mock updating reviews
    updated_review = Mock()
    updated_review.nodeExecId = "test_node_123"
    updated_review.userId = "test-user-123"
    updated_review.graphExecId = "test_graph_exec_456"
    updated_review.chatSessionId = None
    updated_review.graphId = "test_graph_789"
    updated_review.graphVersion = 1
    updated_review.payload = {"data": "modified"}
    updated_review.instructions = "Please review"
    updated_review.editable = True
    updated_review.status = ReviewStatus.APPROVED
    updated_review.reviewMessage = "Approved"
    updated_review.wasEdited = True
    updated_review.processed = False
    updated_review.createdAt = datetime.datetime.now(datetime.timezone.utc)
    updated_review.updatedAt = datetime.datetime.now(datetime.timezone.utc)
    updated_review.reviewedAt = datetime.datetime.now(datetime.timezone.utc)
    mock_prisma.return_value.update = AsyncMock(return_value=updated_review)

    # Mock gather to simulate parallel updates
    mocker.patch(
        "backend.data.human_review.asyncio.gather",
        new=AsyncMock(return_value=[updated_review]),
    )

    # Mock get_node_execution to return node with node_id (async function)
    mock_node_exec = Mock()
    mock_node_exec.node_id = "test_node_def_789"
    mocker.patch(
        "backend.data.execution.get_node_execution",
        new=AsyncMock(return_value=mock_node_exec),
    )

    result = await process_all_reviews_for_execution(
        user_id="test-user-123",
        review_decisions={
            "test_node_123": (ReviewStatus.APPROVED, {"data": "modified"}, "Approved")
        },
    )

    assert len(result) == 1
    assert "test_node_123" in result
    assert result["test_node_123"].status == ReviewStatus.APPROVED
    assert result["test_node_123"].node_id == "test_node_def_789"


@pytest.mark.asyncio(loop_scope="function")
async def test_process_all_reviews_for_execution_validation_errors(
    mocker: pytest_mock.MockFixture,
):
    """Test validation errors in process_all_reviews_for_execution"""
    # Mock finding fewer reviews than requested (some not found)
    mock_find_many = mocker.patch("backend.data.human_review.PendingHumanReview.prisma")
    mock_find_many.return_value.find_many = AsyncMock(
        return_value=[]
    )  # No reviews found

    with pytest.raises(ValueError, match="Reviews not found"):
        await process_all_reviews_for_execution(
            user_id="test-user-123",
            review_decisions={
                "nonexistent_node": (ReviewStatus.APPROVED, {"data": "test"}, "message")
            },
        )


@pytest.mark.asyncio(loop_scope="function")
async def test_process_all_reviews_edit_permission_error(
    mocker: pytest_mock.MockFixture,
    sample_db_review,
):
    """Test editing non-editable review"""
    # Set review as non-editable
    sample_db_review.editable = False

    # Mock finding reviews
    mock_find_many = mocker.patch("backend.data.human_review.PendingHumanReview.prisma")
    mock_find_many.return_value.find_many = AsyncMock(return_value=[sample_db_review])

    with pytest.raises(ValueError, match="not editable"):
        await process_all_reviews_for_execution(
            user_id="test-user-123",
            review_decisions={
                "test_node_123": (
                    ReviewStatus.APPROVED,
                    {"data": "modified"},
                    "message",
                )
            },
        )


@pytest.mark.asyncio(loop_scope="function")
async def test_process_all_reviews_mixed_approval_rejection(
    mocker: pytest_mock.MockFixture,
    sample_db_review,
):
    """Test processing mixed approval and rejection decisions"""
    # Create second review for rejection
    second_review = Mock()
    second_review.nodeExecId = "test_node_456"
    second_review.userId = "test-user-123"
    second_review.graphExecId = "test_graph_exec_456"
    second_review.chatSessionId = None
    second_review.graphId = "test_graph_789"
    second_review.graphVersion = 1
    second_review.payload = {"data": "original"}
    second_review.instructions = "Second review"
    second_review.editable = True
    second_review.status = ReviewStatus.WAITING
    second_review.reviewMessage = None
    second_review.wasEdited = False
    second_review.processed = False
    second_review.createdAt = datetime.datetime.now(datetime.timezone.utc)
    second_review.updatedAt = None
    second_review.reviewedAt = None

    # Mock finding reviews
    mock_find_many = mocker.patch("backend.data.human_review.PendingHumanReview.prisma")
    mock_find_many.return_value.find_many = AsyncMock(
        return_value=[sample_db_review, second_review]
    )

    # Mock updating reviews
    approved_review = Mock()
    approved_review.nodeExecId = "test_node_123"
    approved_review.userId = "test-user-123"
    approved_review.graphExecId = "test_graph_exec_456"
    approved_review.chatSessionId = None
    approved_review.graphId = "test_graph_789"
    approved_review.graphVersion = 1
    approved_review.payload = {"data": "modified"}
    approved_review.instructions = "Please review"
    approved_review.editable = True
    approved_review.status = ReviewStatus.APPROVED
    approved_review.reviewMessage = "Approved"
    approved_review.wasEdited = True
    approved_review.processed = False
    approved_review.createdAt = datetime.datetime.now(datetime.timezone.utc)
    approved_review.updatedAt = datetime.datetime.now(datetime.timezone.utc)
    approved_review.reviewedAt = datetime.datetime.now(datetime.timezone.utc)

    rejected_review = Mock()
    rejected_review.nodeExecId = "test_node_456"
    rejected_review.userId = "test-user-123"
    rejected_review.graphExecId = "test_graph_exec_456"
    rejected_review.chatSessionId = None
    rejected_review.graphId = "test_graph_789"
    rejected_review.graphVersion = 1
    rejected_review.payload = {"data": "original"}
    rejected_review.instructions = "Please review"
    rejected_review.editable = True
    rejected_review.status = ReviewStatus.REJECTED
    rejected_review.reviewMessage = "Rejected"
    rejected_review.wasEdited = False
    rejected_review.processed = False
    rejected_review.createdAt = datetime.datetime.now(datetime.timezone.utc)
    rejected_review.updatedAt = datetime.datetime.now(datetime.timezone.utc)
    rejected_review.reviewedAt = datetime.datetime.now(datetime.timezone.utc)

    mocker.patch(
        "backend.data.human_review.asyncio.gather",
        new=AsyncMock(return_value=[approved_review, rejected_review]),
    )

    # Mock get_node_execution to return node with node_id (async function)
    mock_node_exec = Mock()
    mock_node_exec.node_id = "test_node_def_789"
    mocker.patch(
        "backend.data.execution.get_node_execution",
        new=AsyncMock(return_value=mock_node_exec),
    )

    result = await process_all_reviews_for_execution(
        user_id="test-user-123",
        review_decisions={
            "test_node_123": (ReviewStatus.APPROVED, {"data": "modified"}, "Approved"),
            "test_node_456": (ReviewStatus.REJECTED, None, "Rejected"),
        },
    )

    assert len(result) == 2
    assert "test_node_123" in result
    assert "test_node_456" in result
    assert result["test_node_123"].node_id == "test_node_def_789"
    assert result["test_node_456"].node_id == "test_node_def_789"


@pytest.mark.asyncio(loop_scope="function")
async def test_a_chat_with_legacy_and_new_rows_syncs_one_alert(
    mocker: pytest_mock.MockFixture,
):
    def row(node_exec_id, graph_exec_id, graph_id, session_id):
        r = Mock()
        r.nodeExecId = node_exec_id
        r.userId = "u1"
        r.graphExecId = graph_exec_id
        r.graphId = graph_id
        r.graphVersion = 1 if graph_id else None
        r.chatSessionId = session_id
        r.payload = {}
        r.instructions = "Review"
        r.editable = True
        r.status = ReviewStatus.WAITING
        r.reviewMessage = None
        r.wasEdited = False
        r.processed = False
        r.createdAt = r.updatedAt = r.reviewedAt = datetime.datetime.now(
            datetime.timezone.utc
        )
        return r

    rows = [
        row("copilot-node-a:1", "copilot-session-s1", "copilot-session-s1", None),
        row("copilot-node-b:2", None, None, "s1"),
        row("copilot-node-c:3", "copilot-session-s1", "copilot-session-s1", "s1"),
        row("node-exec-g", "ge-1", "g1", None),
    ]
    mock_prisma = mocker.patch("backend.data.human_review.PendingHumanReview.prisma")
    mock_prisma.return_value.find_many = AsyncMock(return_value=rows)
    mock_prisma.return_value.update = AsyncMock()
    mocker.patch(
        "backend.data.human_review.asyncio.gather", new=AsyncMock(return_value=[])
    )
    sync = mocker.patch(
        "backend.data.human_review._sync_awaiting_review_safely", new=AsyncMock()
    )

    await process_all_reviews_for_execution(
        user_id="u1",
        review_decisions={
            r.nodeExecId: (ReviewStatus.APPROVED, None, None) for r in rows
        },
    )

    calls = [c.args for c in sync.await_args_list]
    assert len(calls) == 2
    assert set(calls) == {("u1", None, "s1"), ("u1", "g1", None)}
