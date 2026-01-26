import datetime
from typing import AsyncGenerator

import httpx
import pytest
import pytest_asyncio
import pytest_mock
from prisma.enums import ReviewStatus
from pytest_snapshot.plugin import Snapshot

from backend.api.rest_api import app
from backend.data.execution import (
    ExecutionContext,
    ExecutionStatus,
    NodeExecutionResult,
)
from backend.data.graph import GraphSettings

from .model import PendingHumanReviewModel

# Using a fixed timestamp for reproducible tests
FIXED_NOW = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)


@pytest_asyncio.fixture(loop_scope="session")
async def client(server, mock_jwt_user) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create async HTTP client with auth overrides"""
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    # Override get_jwt_payload dependency to return our test user
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as http_client:
        yield http_client

    # Clean up overrides
    app.dependency_overrides.pop(get_jwt_payload, None)


@pytest.fixture
def sample_pending_review(test_user_id: str) -> PendingHumanReviewModel:
    """Create a sample pending review for testing"""
    return PendingHumanReviewModel(
        node_exec_id="test_node_123",
        node_id="test_node_def_456",
        user_id=test_user_id,
        graph_exec_id="test_graph_exec_456",
        graph_id="test_graph_789",
        graph_version=1,
        payload={"data": "test payload", "value": 42},
        instructions="Please review this data",
        editable=True,
        status=ReviewStatus.WAITING,
        review_message=None,
        was_edited=None,
        processed=False,
        created_at=FIXED_NOW,
        updated_at=None,
        reviewed_at=None,
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_get_pending_reviews_empty(
    client: httpx.AsyncClient,
    mocker: pytest_mock.MockerFixture,
    snapshot: Snapshot,
    test_user_id: str,
) -> None:
    """Test getting pending reviews when none exist"""
    mock_get_reviews = mocker.patch(
        "backend.api.features.executions.review.routes.get_pending_reviews_for_user"
    )
    mock_get_reviews.return_value = []

    response = await client.get("/api/review/pending")

    assert response.status_code == 200
    assert response.json() == []
    mock_get_reviews.assert_called_once_with(test_user_id, 1, 25)


@pytest.mark.asyncio(loop_scope="session")
async def test_get_pending_reviews_with_data(
    client: httpx.AsyncClient,
    mocker: pytest_mock.MockerFixture,
    sample_pending_review: PendingHumanReviewModel,
    snapshot: Snapshot,
    test_user_id: str,
) -> None:
    """Test getting pending reviews with data"""
    mock_get_reviews = mocker.patch(
        "backend.api.features.executions.review.routes.get_pending_reviews_for_user"
    )
    mock_get_reviews.return_value = [sample_pending_review]

    response = await client.get("/api/review/pending?page=2&page_size=10")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["node_exec_id"] == "test_node_123"
    assert data[0]["status"] == "WAITING"
    mock_get_reviews.assert_called_once_with(test_user_id, 2, 10)


@pytest.mark.asyncio(loop_scope="session")
async def test_get_pending_reviews_for_execution_success(
    client: httpx.AsyncClient,
    mocker: pytest_mock.MockerFixture,
    sample_pending_review: PendingHumanReviewModel,
    snapshot: Snapshot,
    test_user_id: str,
) -> None:
    """Test getting pending reviews for specific execution"""
    mock_get_graph_execution = mocker.patch(
        "backend.api.features.executions.review.routes.get_graph_execution_meta"
    )
    mock_get_graph_execution.return_value = {
        "id": "test_graph_exec_456",
        "user_id": test_user_id,
    }

    mock_get_reviews = mocker.patch(
        "backend.api.features.executions.review.routes.get_pending_reviews_for_execution"
    )
    mock_get_reviews.return_value = [sample_pending_review]

    response = await client.get("/api/review/execution/test_graph_exec_456")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["graph_exec_id"] == "test_graph_exec_456"


@pytest.mark.asyncio(loop_scope="session")
async def test_get_pending_reviews_for_execution_not_available(
    client: httpx.AsyncClient,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Test access denied when user doesn't own the execution"""
    mock_get_graph_execution = mocker.patch(
        "backend.api.features.executions.review.routes.get_graph_execution_meta"
    )
    mock_get_graph_execution.return_value = None

    response = await client.get("/api/review/execution/test_graph_exec_456")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


@pytest.mark.asyncio(loop_scope="session")
async def test_process_review_action_approve_success(
    client: httpx.AsyncClient,
    mocker: pytest_mock.MockerFixture,
    sample_pending_review: PendingHumanReviewModel,
    test_user_id: str,
) -> None:
    """Test successful review approval"""
    # Mock the route functions

    # Mock get_reviews_by_node_exec_ids (called to find the graph_exec_id)
    mock_get_reviews_for_user = mocker.patch(
        "backend.api.features.executions.review.routes.get_reviews_by_node_exec_ids"
    )
    mock_get_reviews_for_user.return_value = {"test_node_123": sample_pending_review}

    mock_get_reviews_for_execution = mocker.patch(
        "backend.api.features.executions.review.routes.get_pending_reviews_for_execution"
    )
    mock_get_reviews_for_execution.return_value = [sample_pending_review]

    mock_process_all_reviews = mocker.patch(
        "backend.api.features.executions.review.routes.process_all_reviews_for_execution"
    )
    # Create approved review for return
    approved_review = PendingHumanReviewModel(
        node_exec_id="test_node_123",
        user_id=test_user_id,
        graph_exec_id="test_graph_exec_456",
        graph_id="test_graph_789",
        graph_version=1,
        payload={"data": "modified payload", "value": 50},
        instructions="Please review this data",
        editable=True,
        status=ReviewStatus.APPROVED,
        review_message="Looks good",
        was_edited=True,
        processed=False,
        created_at=FIXED_NOW,
        updated_at=FIXED_NOW,
        reviewed_at=FIXED_NOW,
    )
    mock_process_all_reviews.return_value = {"test_node_123": approved_review}

    # Mock get_graph_execution_meta to return execution in REVIEW status
    mock_get_graph_exec = mocker.patch(
        "backend.api.features.executions.review.routes.get_graph_execution_meta"
    )
    mock_graph_exec_meta = mocker.Mock()
    mock_graph_exec_meta.status = ExecutionStatus.REVIEW
    mock_get_graph_exec.return_value = mock_graph_exec_meta

    mock_has_pending = mocker.patch(
        "backend.api.features.executions.review.routes.has_pending_reviews_for_graph_exec"
    )
    mock_has_pending.return_value = False

    mocker.patch("backend.api.features.executions.review.routes.add_graph_execution")

    request_data = {
        "reviews": [
            {
                "node_exec_id": "test_node_123",
                "approved": True,
                "message": "Looks good",
                "reviewed_data": {"data": "modified payload", "value": 50},
            }
        ]
    }

    response = await client.post("/api/review/action", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert data["approved_count"] == 1
    assert data["rejected_count"] == 0
    assert data["failed_count"] == 0
    assert data["error"] is None


@pytest.mark.asyncio(loop_scope="session")
async def test_process_review_action_reject_success(
    client: httpx.AsyncClient,
    mocker: pytest_mock.MockerFixture,
    sample_pending_review: PendingHumanReviewModel,
    test_user_id: str,
) -> None:
    """Test successful review rejection"""
    # Mock the route functions

    # Mock get_reviews_by_node_exec_ids (called to find the graph_exec_id)
    mock_get_reviews_for_user = mocker.patch(
        "backend.api.features.executions.review.routes.get_reviews_by_node_exec_ids"
    )
    mock_get_reviews_for_user.return_value = {"test_node_123": sample_pending_review}

    # Mock get_graph_execution_meta to return execution in REVIEW status
    mock_get_graph_exec = mocker.patch(
        "backend.api.features.executions.review.routes.get_graph_execution_meta"
    )
    mock_graph_exec_meta = mocker.Mock()
    mock_graph_exec_meta.status = ExecutionStatus.REVIEW
    mock_get_graph_exec.return_value = mock_graph_exec_meta

    mock_get_reviews_for_execution = mocker.patch(
        "backend.api.features.executions.review.routes.get_pending_reviews_for_execution"
    )
    mock_get_reviews_for_execution.return_value = [sample_pending_review]

    mock_process_all_reviews = mocker.patch(
        "backend.api.features.executions.review.routes.process_all_reviews_for_execution"
    )
    rejected_review = PendingHumanReviewModel(
        node_exec_id="test_node_123",
        user_id=test_user_id,
        graph_exec_id="test_graph_exec_456",
        graph_id="test_graph_789",
        graph_version=1,
        payload={"data": "test payload"},
        instructions="Please review",
        editable=True,
        status=ReviewStatus.REJECTED,
        review_message="Rejected by user",
        was_edited=False,
        processed=False,
        created_at=FIXED_NOW,
        updated_at=None,
        reviewed_at=FIXED_NOW,
    )
    mock_process_all_reviews.return_value = {"test_node_123": rejected_review}

    mock_has_pending = mocker.patch(
        "backend.api.features.executions.review.routes.has_pending_reviews_for_graph_exec"
    )
    mock_has_pending.return_value = False

    request_data = {
        "reviews": [
            {
                "node_exec_id": "test_node_123",
                "approved": False,
                "message": None,
            }
        ]
    }

    response = await client.post("/api/review/action", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert data["approved_count"] == 0
    assert data["rejected_count"] == 1
    assert data["failed_count"] == 0
    assert data["error"] is None


@pytest.mark.asyncio(loop_scope="session")
async def test_process_review_action_mixed_success(
    client: httpx.AsyncClient,
    mocker: pytest_mock.MockerFixture,
    sample_pending_review: PendingHumanReviewModel,
    test_user_id: str,
) -> None:
    """Test mixed approve/reject operations"""
    # Create a second review
    second_review = PendingHumanReviewModel(
        node_exec_id="test_node_456",
        user_id=test_user_id,
        graph_exec_id="test_graph_exec_456",
        graph_id="test_graph_789",
        graph_version=1,
        payload={"data": "second payload"},
        instructions="Second review",
        editable=False,
        status=ReviewStatus.WAITING,
        review_message=None,
        was_edited=None,
        processed=False,
        created_at=FIXED_NOW,
        updated_at=None,
        reviewed_at=None,
    )

    # Mock the route functions

    # Mock get_reviews_by_node_exec_ids (called to find the graph_exec_id)
    mock_get_reviews_for_user = mocker.patch(
        "backend.api.features.executions.review.routes.get_reviews_by_node_exec_ids"
    )
    mock_get_reviews_for_user.return_value = {
        "test_node_123": sample_pending_review,
        "test_node_456": second_review,
    }

    mock_get_reviews_for_execution = mocker.patch(
        "backend.api.features.executions.review.routes.get_pending_reviews_for_execution"
    )
    mock_get_reviews_for_execution.return_value = [sample_pending_review, second_review]

    mock_process_all_reviews = mocker.patch(
        "backend.api.features.executions.review.routes.process_all_reviews_for_execution"
    )
    # Create approved version of first review
    approved_review = PendingHumanReviewModel(
        node_exec_id="test_node_123",
        user_id=test_user_id,
        graph_exec_id="test_graph_exec_456",
        graph_id="test_graph_789",
        graph_version=1,
        payload={"data": "modified"},
        instructions="Please review",
        editable=True,
        status=ReviewStatus.APPROVED,
        review_message="Approved",
        was_edited=True,
        processed=False,
        created_at=FIXED_NOW,
        updated_at=None,
        reviewed_at=FIXED_NOW,
    )
    # Create rejected version of second review
    rejected_review = PendingHumanReviewModel(
        node_exec_id="test_node_456",
        user_id=test_user_id,
        graph_exec_id="test_graph_exec_456",
        graph_id="test_graph_789",
        graph_version=1,
        payload={"data": "second payload"},
        instructions="Second review",
        editable=False,
        status=ReviewStatus.REJECTED,
        review_message="Rejected by user",
        was_edited=False,
        processed=False,
        created_at=FIXED_NOW,
        updated_at=None,
        reviewed_at=FIXED_NOW,
    )
    mock_process_all_reviews.return_value = {
        "test_node_123": approved_review,
        "test_node_456": rejected_review,
    }

    # Mock get_graph_execution_meta to return execution in REVIEW status
    mock_get_graph_exec = mocker.patch(
        "backend.api.features.executions.review.routes.get_graph_execution_meta"
    )
    mock_graph_exec_meta = mocker.Mock()
    mock_graph_exec_meta.status = ExecutionStatus.REVIEW
    mock_get_graph_exec.return_value = mock_graph_exec_meta

    mock_has_pending = mocker.patch(
        "backend.api.features.executions.review.routes.has_pending_reviews_for_graph_exec"
    )
    mock_has_pending.return_value = False

    request_data = {
        "reviews": [
            {
                "node_exec_id": "test_node_123",
                "approved": True,
                "message": "Approved",
                "reviewed_data": {"data": "modified"},
            },
            {
                "node_exec_id": "test_node_456",
                "approved": False,
                "message": None,
            },
        ]
    }

    response = await client.post("/api/review/action", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert data["approved_count"] == 1
    assert data["rejected_count"] == 1
    assert data["failed_count"] == 0
    assert data["error"] is None


@pytest.mark.asyncio(loop_scope="session")
async def test_process_review_action_empty_request(
    client: httpx.AsyncClient,
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    """Test error when no reviews provided"""
    request_data = {"reviews": []}

    response = await client.post("/api/review/action", json=request_data)

    assert response.status_code == 422
    response_data = response.json()
    # Pydantic validation error format
    assert isinstance(response_data["detail"], list)
    assert len(response_data["detail"]) > 0
    assert "At least one review must be provided" in response_data["detail"][0]["msg"]


@pytest.mark.asyncio(loop_scope="session")
async def test_process_review_action_review_not_found(
    client: httpx.AsyncClient,
    mocker: pytest_mock.MockerFixture,
    sample_pending_review: PendingHumanReviewModel,
    test_user_id: str,
) -> None:
    """Test error when review is not found"""
    # Mock get_reviews_by_node_exec_ids (called to find the graph_exec_id)
    mock_get_reviews_for_user = mocker.patch(
        "backend.api.features.executions.review.routes.get_reviews_by_node_exec_ids"
    )
    # Return empty dict to simulate review not found
    mock_get_reviews_for_user.return_value = {}

    # Mock get_graph_execution_meta to return execution in REVIEW status
    mock_get_graph_exec = mocker.patch(
        "backend.api.features.executions.review.routes.get_graph_execution_meta"
    )
    mock_graph_exec_meta = mocker.Mock()
    mock_graph_exec_meta.status = ExecutionStatus.REVIEW
    mock_get_graph_exec.return_value = mock_graph_exec_meta

    # Mock the functions that extract graph execution ID from the request
    mock_get_reviews_for_execution = mocker.patch(
        "backend.api.features.executions.review.routes.get_pending_reviews_for_execution"
    )
    mock_get_reviews_for_execution.return_value = []  # No reviews found

    # Mock process_all_reviews to simulate not finding reviews
    mock_process_all_reviews = mocker.patch(
        "backend.api.features.executions.review.routes.process_all_reviews_for_execution"
    )
    # This should raise a ValueError with "Reviews not found" message based on the data/human_review.py logic
    mock_process_all_reviews.side_effect = ValueError(
        "Reviews not found or access denied for IDs: nonexistent_node"
    )

    request_data = {
        "reviews": [
            {
                "node_exec_id": "nonexistent_node",
                "approved": True,
                "message": "Test",
            }
        ]
    }

    response = await client.post("/api/review/action", json=request_data)

    assert response.status_code == 404
    assert "Review(s) not found" in response.json()["detail"]


@pytest.mark.asyncio(loop_scope="session")
async def test_process_review_action_partial_failure(
    client: httpx.AsyncClient,
    mocker: pytest_mock.MockerFixture,
    sample_pending_review: PendingHumanReviewModel,
    test_user_id: str,
) -> None:
    """Test handling of partial failures in review processing"""
    # Mock get_reviews_by_node_exec_ids (called to find the graph_exec_id)
    mock_get_reviews_for_user = mocker.patch(
        "backend.api.features.executions.review.routes.get_reviews_by_node_exec_ids"
    )
    mock_get_reviews_for_user.return_value = {"test_node_123": sample_pending_review}

    # Mock get_graph_execution_meta to return execution in REVIEW status
    mock_get_graph_exec = mocker.patch(
        "backend.api.features.executions.review.routes.get_graph_execution_meta"
    )
    mock_graph_exec_meta = mocker.Mock()
    mock_graph_exec_meta.status = ExecutionStatus.REVIEW
    mock_get_graph_exec.return_value = mock_graph_exec_meta

    # Mock the route functions
    mock_get_reviews_for_execution = mocker.patch(
        "backend.api.features.executions.review.routes.get_pending_reviews_for_execution"
    )
    mock_get_reviews_for_execution.return_value = [sample_pending_review]

    # Mock partial failure in processing
    mock_process_all_reviews = mocker.patch(
        "backend.api.features.executions.review.routes.process_all_reviews_for_execution"
    )
    mock_process_all_reviews.side_effect = ValueError("Some reviews failed validation")

    request_data = {
        "reviews": [
            {
                "node_exec_id": "test_node_123",
                "approved": True,
                "message": "Test",
            }
        ]
    }

    response = await client.post("/api/review/action", json=request_data)

    assert response.status_code == 400
    assert "Some reviews failed validation" in response.json()["detail"]


@pytest.mark.asyncio(loop_scope="session")
async def test_process_review_action_invalid_node_exec_id(
    client: httpx.AsyncClient,
    mocker: pytest_mock.MockerFixture,
    sample_pending_review: PendingHumanReviewModel,
    test_user_id: str,
) -> None:
    """Test failure when trying to process review with invalid node execution ID"""
    # Mock get_reviews_by_node_exec_ids (called to find the graph_exec_id)
    mock_get_reviews_for_user = mocker.patch(
        "backend.api.features.executions.review.routes.get_reviews_by_node_exec_ids"
    )
    # Return empty dict to simulate review not found
    mock_get_reviews_for_user.return_value = {}

    # Mock get_graph_execution_meta to return execution in REVIEW status
    mock_get_graph_exec = mocker.patch(
        "backend.api.features.executions.review.routes.get_graph_execution_meta"
    )
    mock_graph_exec_meta = mocker.Mock()
    mock_graph_exec_meta.status = ExecutionStatus.REVIEW
    mock_get_graph_exec.return_value = mock_graph_exec_meta

    request_data = {
        "reviews": [
            {
                "node_exec_id": "invalid-node-format",
                "approved": True,
                "message": "Test",
            }
        ]
    }

    response = await client.post("/api/review/action", json=request_data)

    # Returns 404 when review is not found
    assert response.status_code == 404
    assert "Review(s) not found" in response.json()["detail"]


@pytest.mark.asyncio(loop_scope="session")
async def test_process_review_action_auto_approve_creates_auto_approval_records(
    client: httpx.AsyncClient,
    mocker: pytest_mock.MockerFixture,
    sample_pending_review: PendingHumanReviewModel,
    test_user_id: str,
) -> None:
    """Test that auto_approve_future_actions flag creates auto-approval records"""
    # Mock get_reviews_by_node_exec_ids (called to find the graph_exec_id)
    mock_get_reviews_for_user = mocker.patch(
        "backend.api.features.executions.review.routes.get_reviews_by_node_exec_ids"
    )
    mock_get_reviews_for_user.return_value = {"test_node_123": sample_pending_review}

    # Mock process_all_reviews
    mock_process_all_reviews = mocker.patch(
        "backend.api.features.executions.review.routes.process_all_reviews_for_execution"
    )
    approved_review = PendingHumanReviewModel(
        node_exec_id="test_node_123",
        user_id=test_user_id,
        graph_exec_id="test_graph_exec_456",
        graph_id="test_graph_789",
        graph_version=1,
        payload={"data": "test payload"},
        instructions="Please review",
        editable=True,
        status=ReviewStatus.APPROVED,
        review_message="Approved",
        was_edited=False,
        processed=False,
        created_at=FIXED_NOW,
        updated_at=FIXED_NOW,
        reviewed_at=FIXED_NOW,
    )
    mock_process_all_reviews.return_value = {"test_node_123": approved_review}

    # Mock get_node_executions to return node_id mapping
    mock_get_node_executions = mocker.patch(
        "backend.data.execution.get_node_executions"
    )
    mock_node_exec = mocker.Mock(spec=NodeExecutionResult)
    mock_node_exec.node_exec_id = "test_node_123"
    mock_node_exec.node_id = "test_node_def_456"
    mock_get_node_executions.return_value = [mock_node_exec]

    # Mock create_auto_approval_record
    mock_create_auto_approval = mocker.patch(
        "backend.api.features.executions.review.routes.create_auto_approval_record"
    )

    # Mock get_graph_execution_meta to return execution in REVIEW status
    mock_get_graph_exec = mocker.patch(
        "backend.api.features.executions.review.routes.get_graph_execution_meta"
    )
    mock_graph_exec_meta = mocker.Mock()
    mock_graph_exec_meta.status = ExecutionStatus.REVIEW
    mock_get_graph_exec.return_value = mock_graph_exec_meta

    # Mock has_pending_reviews_for_graph_exec
    mock_has_pending = mocker.patch(
        "backend.api.features.executions.review.routes.has_pending_reviews_for_graph_exec"
    )
    mock_has_pending.return_value = False

    # Mock get_graph_settings to return custom settings
    mock_get_settings = mocker.patch(
        "backend.api.features.executions.review.routes.get_graph_settings"
    )
    mock_get_settings.return_value = GraphSettings(
        human_in_the_loop_safe_mode=True,
        sensitive_action_safe_mode=True,
    )

    # Mock get_user_by_id to prevent database access
    mock_get_user = mocker.patch(
        "backend.api.features.executions.review.routes.get_user_by_id"
    )
    mock_user = mocker.Mock()
    mock_user.timezone = "UTC"
    mock_get_user.return_value = mock_user

    # Mock add_graph_execution
    mock_add_execution = mocker.patch(
        "backend.api.features.executions.review.routes.add_graph_execution"
    )

    request_data = {
        "reviews": [
            {
                "node_exec_id": "test_node_123",
                "approved": True,
                "message": "Approved",
                "auto_approve_future": True,
            }
        ],
    }

    response = await client.post("/api/review/action", json=request_data)

    assert response.status_code == 200

    # Verify process_all_reviews_for_execution was called (without auto_approve param)
    mock_process_all_reviews.assert_called_once()

    # Verify create_auto_approval_record was called for the approved review
    mock_create_auto_approval.assert_called_once_with(
        user_id=test_user_id,
        graph_exec_id="test_graph_exec_456",
        graph_id="test_graph_789",
        graph_version=1,
        node_id="test_node_def_456",
        payload={"data": "test payload"},
    )

    # Verify get_graph_settings was called with correct parameters
    mock_get_settings.assert_called_once_with(
        user_id=test_user_id, graph_id="test_graph_789"
    )

    # Verify add_graph_execution was called with proper ExecutionContext
    mock_add_execution.assert_called_once()
    call_kwargs = mock_add_execution.call_args.kwargs
    execution_context = call_kwargs["execution_context"]

    assert isinstance(execution_context, ExecutionContext)
    assert execution_context.human_in_the_loop_safe_mode is True
    assert execution_context.sensitive_action_safe_mode is True


@pytest.mark.asyncio(loop_scope="session")
async def test_process_review_action_without_auto_approve_still_loads_settings(
    client: httpx.AsyncClient,
    mocker: pytest_mock.MockerFixture,
    sample_pending_review: PendingHumanReviewModel,
    test_user_id: str,
) -> None:
    """Test that execution context is created with settings even without auto-approve"""
    # Mock get_reviews_by_node_exec_ids (called to find the graph_exec_id)
    mock_get_reviews_for_user = mocker.patch(
        "backend.api.features.executions.review.routes.get_reviews_by_node_exec_ids"
    )
    mock_get_reviews_for_user.return_value = {"test_node_123": sample_pending_review}

    # Mock process_all_reviews
    mock_process_all_reviews = mocker.patch(
        "backend.api.features.executions.review.routes.process_all_reviews_for_execution"
    )
    approved_review = PendingHumanReviewModel(
        node_exec_id="test_node_123",
        user_id=test_user_id,
        graph_exec_id="test_graph_exec_456",
        graph_id="test_graph_789",
        graph_version=1,
        payload={"data": "test payload"},
        instructions="Please review",
        editable=True,
        status=ReviewStatus.APPROVED,
        review_message="Approved",
        was_edited=False,
        processed=False,
        created_at=FIXED_NOW,
        updated_at=FIXED_NOW,
        reviewed_at=FIXED_NOW,
    )
    mock_process_all_reviews.return_value = {"test_node_123": approved_review}

    # Mock create_auto_approval_record - should NOT be called when auto_approve is False
    mock_create_auto_approval = mocker.patch(
        "backend.api.features.executions.review.routes.create_auto_approval_record"
    )

    # Mock get_graph_execution_meta to return execution in REVIEW status
    mock_get_graph_exec = mocker.patch(
        "backend.api.features.executions.review.routes.get_graph_execution_meta"
    )
    mock_graph_exec_meta = mocker.Mock()
    mock_graph_exec_meta.status = ExecutionStatus.REVIEW
    mock_get_graph_exec.return_value = mock_graph_exec_meta

    # Mock has_pending_reviews_for_graph_exec
    mock_has_pending = mocker.patch(
        "backend.api.features.executions.review.routes.has_pending_reviews_for_graph_exec"
    )
    mock_has_pending.return_value = False

    # Mock get_graph_settings with sensitive_action_safe_mode enabled
    mock_get_settings = mocker.patch(
        "backend.api.features.executions.review.routes.get_graph_settings"
    )
    mock_get_settings.return_value = GraphSettings(
        human_in_the_loop_safe_mode=False,
        sensitive_action_safe_mode=True,
    )

    # Mock get_user_by_id to prevent database access
    mock_get_user = mocker.patch(
        "backend.api.features.executions.review.routes.get_user_by_id"
    )
    mock_user = mocker.Mock()
    mock_user.timezone = "UTC"
    mock_get_user.return_value = mock_user

    # Mock add_graph_execution
    mock_add_execution = mocker.patch(
        "backend.api.features.executions.review.routes.add_graph_execution"
    )

    # Request WITHOUT auto_approve_future (defaults to False)
    request_data = {
        "reviews": [
            {
                "node_exec_id": "test_node_123",
                "approved": True,
                "message": "Approved",
                # auto_approve_future defaults to False
            }
        ],
    }

    response = await client.post("/api/review/action", json=request_data)

    assert response.status_code == 200

    # Verify process_all_reviews_for_execution was called
    mock_process_all_reviews.assert_called_once()

    # Verify create_auto_approval_record was NOT called (auto_approve_future=False)
    mock_create_auto_approval.assert_not_called()

    # Verify settings were loaded
    mock_get_settings.assert_called_once()

    # Verify ExecutionContext has proper settings
    mock_add_execution.assert_called_once()
    call_kwargs = mock_add_execution.call_args.kwargs
    execution_context = call_kwargs["execution_context"]

    assert isinstance(execution_context, ExecutionContext)
    assert execution_context.human_in_the_loop_safe_mode is False
    assert execution_context.sensitive_action_safe_mode is True


@pytest.mark.asyncio(loop_scope="session")
async def test_process_review_action_auto_approve_only_applies_to_approved_reviews(
    client: httpx.AsyncClient,
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    """Test that auto_approve record is created only for approved reviews"""
    # Create two reviews - one approved, one rejected
    approved_review = PendingHumanReviewModel(
        node_exec_id="node_exec_approved",
        user_id=test_user_id,
        graph_exec_id="test_graph_exec_456",
        graph_id="test_graph_789",
        graph_version=1,
        payload={"data": "approved"},
        instructions="Review",
        editable=True,
        status=ReviewStatus.APPROVED,
        review_message=None,
        was_edited=False,
        processed=False,
        created_at=FIXED_NOW,
        updated_at=FIXED_NOW,
        reviewed_at=FIXED_NOW,
    )
    rejected_review = PendingHumanReviewModel(
        node_exec_id="node_exec_rejected",
        user_id=test_user_id,
        graph_exec_id="test_graph_exec_456",
        graph_id="test_graph_789",
        graph_version=1,
        payload={"data": "rejected"},
        instructions="Review",
        editable=True,
        status=ReviewStatus.REJECTED,
        review_message="Rejected",
        was_edited=False,
        processed=False,
        created_at=FIXED_NOW,
        updated_at=FIXED_NOW,
        reviewed_at=FIXED_NOW,
    )

    # Mock get_reviews_by_node_exec_ids (called to find the graph_exec_id)
    mock_get_reviews_for_user = mocker.patch(
        "backend.api.features.executions.review.routes.get_reviews_by_node_exec_ids"
    )
    # Need to return both reviews in WAITING state (before processing)
    approved_review_waiting = PendingHumanReviewModel(
        node_exec_id="node_exec_approved",
        user_id=test_user_id,
        graph_exec_id="test_graph_exec_456",
        graph_id="test_graph_789",
        graph_version=1,
        payload={"data": "approved"},
        instructions="Review",
        editable=True,
        status=ReviewStatus.WAITING,
        review_message=None,
        was_edited=False,
        processed=False,
        created_at=FIXED_NOW,
    )
    rejected_review_waiting = PendingHumanReviewModel(
        node_exec_id="node_exec_rejected",
        user_id=test_user_id,
        graph_exec_id="test_graph_exec_456",
        graph_id="test_graph_789",
        graph_version=1,
        payload={"data": "rejected"},
        instructions="Review",
        editable=True,
        status=ReviewStatus.WAITING,
        review_message=None,
        was_edited=False,
        processed=False,
        created_at=FIXED_NOW,
    )
    mock_get_reviews_for_user.return_value = {
        "node_exec_approved": approved_review_waiting,
        "node_exec_rejected": rejected_review_waiting,
    }

    # Mock process_all_reviews
    mock_process_all_reviews = mocker.patch(
        "backend.api.features.executions.review.routes.process_all_reviews_for_execution"
    )
    mock_process_all_reviews.return_value = {
        "node_exec_approved": approved_review,
        "node_exec_rejected": rejected_review,
    }

    # Mock get_node_executions to return node_id mapping
    mock_get_node_executions = mocker.patch(
        "backend.data.execution.get_node_executions"
    )
    mock_node_exec = mocker.Mock(spec=NodeExecutionResult)
    mock_node_exec.node_exec_id = "node_exec_approved"
    mock_node_exec.node_id = "test_node_def_approved"
    mock_get_node_executions.return_value = [mock_node_exec]

    # Mock create_auto_approval_record
    mock_create_auto_approval = mocker.patch(
        "backend.api.features.executions.review.routes.create_auto_approval_record"
    )

    # Mock get_graph_execution_meta to return execution in REVIEW status
    mock_get_graph_exec = mocker.patch(
        "backend.api.features.executions.review.routes.get_graph_execution_meta"
    )
    mock_graph_exec_meta = mocker.Mock()
    mock_graph_exec_meta.status = ExecutionStatus.REVIEW
    mock_get_graph_exec.return_value = mock_graph_exec_meta

    # Mock has_pending_reviews_for_graph_exec
    mock_has_pending = mocker.patch(
        "backend.api.features.executions.review.routes.has_pending_reviews_for_graph_exec"
    )
    mock_has_pending.return_value = False

    # Mock get_graph_settings
    mock_get_settings = mocker.patch(
        "backend.api.features.executions.review.routes.get_graph_settings"
    )
    mock_get_settings.return_value = GraphSettings()

    # Mock get_user_by_id to prevent database access
    mock_get_user = mocker.patch(
        "backend.api.features.executions.review.routes.get_user_by_id"
    )
    mock_user = mocker.Mock()
    mock_user.timezone = "UTC"
    mock_get_user.return_value = mock_user

    # Mock add_graph_execution
    mock_add_execution = mocker.patch(
        "backend.api.features.executions.review.routes.add_graph_execution"
    )

    request_data = {
        "reviews": [
            {
                "node_exec_id": "node_exec_approved",
                "approved": True,
                "auto_approve_future": True,
            },
            {
                "node_exec_id": "node_exec_rejected",
                "approved": False,
                "auto_approve_future": True,  # Should be ignored since rejected
            },
        ],
    }

    response = await client.post("/api/review/action", json=request_data)

    assert response.status_code == 200

    # Verify process_all_reviews_for_execution was called
    mock_process_all_reviews.assert_called_once()

    # Verify create_auto_approval_record was called ONLY for the approved review
    # (not for the rejected one)
    mock_create_auto_approval.assert_called_once_with(
        user_id=test_user_id,
        graph_exec_id="test_graph_exec_456",
        graph_id="test_graph_789",
        graph_version=1,
        node_id="test_node_def_approved",
        payload={"data": "approved"},
    )

    # Verify get_node_executions was called to batch-fetch node data
    mock_get_node_executions.assert_called_once()

    # Verify ExecutionContext was created (auto-approval is now DB-based)
    call_kwargs = mock_add_execution.call_args.kwargs
    execution_context = call_kwargs["execution_context"]
    assert isinstance(execution_context, ExecutionContext)


@pytest.mark.asyncio(loop_scope="session")
async def test_process_review_action_per_review_auto_approve_granularity(
    client: httpx.AsyncClient,
    mocker: pytest_mock.MockerFixture,
    sample_pending_review: PendingHumanReviewModel,
    test_user_id: str,
) -> None:
    """Test that auto-approval can be set per-review (granular control)"""
    # Mock get_reviews_by_node_exec_ids - return different reviews based on node_exec_id
    mock_get_reviews_for_user = mocker.patch(
        "backend.api.features.executions.review.routes.get_reviews_by_node_exec_ids"
    )

    # Create a mapping of node_exec_id to review
    review_map = {
        "node_1_auto": PendingHumanReviewModel(
            node_exec_id="node_1_auto",
            user_id=test_user_id,
            graph_exec_id="test_graph_exec",
            graph_id="test_graph",
            graph_version=1,
            payload={"data": "node1"},
            instructions="Review 1",
            editable=True,
            status=ReviewStatus.WAITING,
            review_message=None,
            was_edited=False,
            processed=False,
            created_at=FIXED_NOW,
        ),
        "node_2_manual": PendingHumanReviewModel(
            node_exec_id="node_2_manual",
            user_id=test_user_id,
            graph_exec_id="test_graph_exec",
            graph_id="test_graph",
            graph_version=1,
            payload={"data": "node2"},
            instructions="Review 2",
            editable=True,
            status=ReviewStatus.WAITING,
            review_message=None,
            was_edited=False,
            processed=False,
            created_at=FIXED_NOW,
        ),
        "node_3_auto": PendingHumanReviewModel(
            node_exec_id="node_3_auto",
            user_id=test_user_id,
            graph_exec_id="test_graph_exec",
            graph_id="test_graph",
            graph_version=1,
            payload={"data": "node3"},
            instructions="Review 3",
            editable=True,
            status=ReviewStatus.WAITING,
            review_message=None,
            was_edited=False,
            processed=False,
            created_at=FIXED_NOW,
        ),
    }

    # Return the review map dict (batch function returns all requested reviews)
    mock_get_reviews_for_user.return_value = review_map

    # Mock process_all_reviews - return 3 approved reviews
    mock_process_all_reviews = mocker.patch(
        "backend.api.features.executions.review.routes.process_all_reviews_for_execution"
    )
    mock_process_all_reviews.return_value = {
        "node_1_auto": PendingHumanReviewModel(
            node_exec_id="node_1_auto",
            user_id=test_user_id,
            graph_exec_id="test_graph_exec",
            graph_id="test_graph",
            graph_version=1,
            payload={"data": "node1"},
            instructions="Review 1",
            editable=True,
            status=ReviewStatus.APPROVED,
            review_message=None,
            was_edited=False,
            processed=False,
            created_at=FIXED_NOW,
            updated_at=FIXED_NOW,
            reviewed_at=FIXED_NOW,
        ),
        "node_2_manual": PendingHumanReviewModel(
            node_exec_id="node_2_manual",
            user_id=test_user_id,
            graph_exec_id="test_graph_exec",
            graph_id="test_graph",
            graph_version=1,
            payload={"data": "node2"},
            instructions="Review 2",
            editable=True,
            status=ReviewStatus.APPROVED,
            review_message=None,
            was_edited=False,
            processed=False,
            created_at=FIXED_NOW,
            updated_at=FIXED_NOW,
            reviewed_at=FIXED_NOW,
        ),
        "node_3_auto": PendingHumanReviewModel(
            node_exec_id="node_3_auto",
            user_id=test_user_id,
            graph_exec_id="test_graph_exec",
            graph_id="test_graph",
            graph_version=1,
            payload={"data": "node3"},
            instructions="Review 3",
            editable=True,
            status=ReviewStatus.APPROVED,
            review_message=None,
            was_edited=False,
            processed=False,
            created_at=FIXED_NOW,
            updated_at=FIXED_NOW,
            reviewed_at=FIXED_NOW,
        ),
    }

    # Mock get_node_executions to return batch node data
    mock_get_node_executions = mocker.patch(
        "backend.data.execution.get_node_executions"
    )
    # Create mock node executions for each review
    mock_node_execs = []
    for node_exec_id in ["node_1_auto", "node_2_manual", "node_3_auto"]:
        mock_node = mocker.Mock(spec=NodeExecutionResult)
        mock_node.node_exec_id = node_exec_id
        mock_node.node_id = f"node_def_{node_exec_id}"
        mock_node_execs.append(mock_node)
    mock_get_node_executions.return_value = mock_node_execs

    # Mock create_auto_approval_record
    mock_create_auto_approval = mocker.patch(
        "backend.api.features.executions.review.routes.create_auto_approval_record"
    )

    # Mock get_graph_execution_meta
    mock_get_graph_exec = mocker.patch(
        "backend.api.features.executions.review.routes.get_graph_execution_meta"
    )
    mock_graph_exec_meta = mocker.Mock()
    mock_graph_exec_meta.status = ExecutionStatus.REVIEW
    mock_get_graph_exec.return_value = mock_graph_exec_meta

    # Mock has_pending_reviews_for_graph_exec
    mock_has_pending = mocker.patch(
        "backend.api.features.executions.review.routes.has_pending_reviews_for_graph_exec"
    )
    mock_has_pending.return_value = False

    # Mock settings and execution
    mock_get_settings = mocker.patch(
        "backend.api.features.executions.review.routes.get_graph_settings"
    )
    mock_get_settings.return_value = GraphSettings(
        human_in_the_loop_safe_mode=False, sensitive_action_safe_mode=False
    )

    mocker.patch("backend.api.features.executions.review.routes.add_graph_execution")
    mocker.patch("backend.api.features.executions.review.routes.get_user_by_id")

    # Request with granular auto-approval:
    # - node_1_auto: auto_approve_future=True
    # - node_2_manual: auto_approve_future=False (explicit)
    # - node_3_auto: auto_approve_future=True
    request_data = {
        "reviews": [
            {
                "node_exec_id": "node_1_auto",
                "approved": True,
                "auto_approve_future": True,
            },
            {
                "node_exec_id": "node_2_manual",
                "approved": True,
                "auto_approve_future": False,  # Don't auto-approve this one
            },
            {
                "node_exec_id": "node_3_auto",
                "approved": True,
                "auto_approve_future": True,
            },
        ],
    }

    response = await client.post("/api/review/action", json=request_data)

    assert response.status_code == 200

    # Verify create_auto_approval_record was called ONLY for reviews with auto_approve_future=True
    assert mock_create_auto_approval.call_count == 2

    # Check that it was called for node_1 and node_3, but NOT node_2
    call_args_list = [call.kwargs for call in mock_create_auto_approval.call_args_list]
    node_ids_with_auto_approval = [args["node_id"] for args in call_args_list]

    assert "node_def_node_1_auto" in node_ids_with_auto_approval
    assert "node_def_node_3_auto" in node_ids_with_auto_approval
    assert "node_def_node_2_manual" not in node_ids_with_auto_approval
