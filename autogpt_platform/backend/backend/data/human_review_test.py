import datetime
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_mock
from prisma.enums import ReviewStatus

from backend.data.human_review import (
    get_or_create_human_review,
    get_pending_reviews_for_execution,
    get_pending_reviews_for_user,
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
    mock_find_many = mocker.patch("backend.data.human_review.PendingHumanReview.prisma")
    mock_find_many.return_value.find_many = AsyncMock(return_value=[sample_db_review])

    # Mock get_node_execution to return node with node_id (async function)
    mock_node_exec = Mock()
    mock_node_exec.node_id = "test_node_def_789"
    mocker.patch(
        "backend.data.execution.get_node_execution",
        new=AsyncMock(return_value=mock_node_exec),
    )

    result = await get_pending_reviews_for_user("test_user", page=2, page_size=10)

    assert len(result) == 1
    assert result[0].node_exec_id == "test_node_123"
    assert result[0].node_id == "test_node_def_789"

    # Verify pagination parameters
    call_args = mock_find_many.return_value.find_many.call_args
    assert call_args.kwargs["skip"] == 10  # (page-1) * page_size = (2-1) * 10
    assert call_args.kwargs["take"] == 10


@pytest.mark.asyncio(loop_scope="function")
async def test_get_pending_reviews_for_execution(
    mocker: pytest_mock.MockFixture,
    sample_db_review,
):
    """Test getting pending reviews for specific execution"""
    mock_find_many = mocker.patch("backend.data.human_review.PendingHumanReview.prisma")
    mock_find_many.return_value.find_many = AsyncMock(return_value=[sample_db_review])

    # Mock get_node_execution to return node with node_id (async function)
    mock_node_exec = Mock()
    mock_node_exec.node_id = "test_node_def_789"
    mocker.patch(
        "backend.data.execution.get_node_execution",
        new=AsyncMock(return_value=mock_node_exec),
    )

    result = await get_pending_reviews_for_execution(
        "test_graph_exec_456", "test-user-123"
    )

    assert len(result) == 1
    assert result[0].graph_exec_id == "test_graph_exec_456"
    assert result[0].node_id == "test_node_def_789"

    # Verify it filters by execution and user
    call_args = mock_find_many.return_value.find_many.call_args
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
