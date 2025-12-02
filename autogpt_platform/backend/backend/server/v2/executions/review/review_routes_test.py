import datetime

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from prisma.enums import ReviewStatus
from pytest_snapshot.plugin import Snapshot

from backend.server.rest_api import handle_internal_http_error
from backend.server.v2.executions.review.model import PendingHumanReviewModel
from backend.server.v2.executions.review.routes import router

# Using a fixed timestamp for reproducible tests
FIXED_NOW = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)

app = fastapi.FastAPI()
app.include_router(router, prefix="/api/review")
app.add_exception_handler(ValueError, handle_internal_http_error(400))

client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    """Setup auth overrides for all tests in this module"""
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def sample_pending_review(test_user_id: str) -> PendingHumanReviewModel:
    """Create a sample pending review for testing"""
    return PendingHumanReviewModel(
        node_exec_id="test_node_123",
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


def test_get_pending_reviews_empty(
    mocker: pytest_mock.MockFixture,
    snapshot: Snapshot,
    test_user_id: str,
) -> None:
    """Test getting pending reviews when none exist"""
    mock_get_reviews = mocker.patch(
        "backend.server.v2.executions.review.routes.get_pending_reviews_for_user"
    )
    mock_get_reviews.return_value = []

    response = client.get("/api/review/pending")

    assert response.status_code == 200
    assert response.json() == []
    mock_get_reviews.assert_called_once_with(test_user_id, 1, 25)


def test_get_pending_reviews_with_data(
    mocker: pytest_mock.MockFixture,
    sample_pending_review: PendingHumanReviewModel,
    snapshot: Snapshot,
    test_user_id: str,
) -> None:
    """Test getting pending reviews with data"""
    mock_get_reviews = mocker.patch(
        "backend.server.v2.executions.review.routes.get_pending_reviews_for_user"
    )
    mock_get_reviews.return_value = [sample_pending_review]

    response = client.get("/api/review/pending?page=2&page_size=10")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["node_exec_id"] == "test_node_123"
    assert data[0]["status"] == "WAITING"
    mock_get_reviews.assert_called_once_with(test_user_id, 2, 10)


def test_get_pending_reviews_for_execution_success(
    mocker: pytest_mock.MockFixture,
    sample_pending_review: PendingHumanReviewModel,
    snapshot: Snapshot,
    test_user_id: str,
) -> None:
    """Test getting pending reviews for specific execution"""
    mock_get_graph_execution = mocker.patch(
        "backend.server.v2.executions.review.routes.get_graph_execution_meta"
    )
    mock_get_graph_execution.return_value = {
        "id": "test_graph_exec_456",
        "user_id": test_user_id,
    }

    mock_get_reviews = mocker.patch(
        "backend.server.v2.executions.review.routes.get_pending_reviews_for_execution"
    )
    mock_get_reviews.return_value = [sample_pending_review]

    response = client.get("/api/review/execution/test_graph_exec_456")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["graph_exec_id"] == "test_graph_exec_456"


def test_get_pending_reviews_for_execution_access_denied(
    mocker: pytest_mock.MockFixture,
    test_user_id: str,
) -> None:
    """Test access denied when user doesn't own the execution"""
    mock_get_graph_execution = mocker.patch(
        "backend.server.v2.executions.review.routes.get_graph_execution_meta"
    )
    mock_get_graph_execution.return_value = None

    response = client.get("/api/review/execution/test_graph_exec_456")

    assert response.status_code == 403
    assert "Access denied" in response.json()["detail"]


def test_process_review_action_approve_success(
    mocker: pytest_mock.MockFixture,
    sample_pending_review: PendingHumanReviewModel,
    test_user_id: str,
) -> None:
    """Test successful review approval"""
    # Mock the route functions

    mock_get_reviews_for_execution = mocker.patch(
        "backend.server.v2.executions.review.routes.get_pending_reviews_for_execution"
    )
    mock_get_reviews_for_execution.return_value = [sample_pending_review]

    mock_process_all_reviews = mocker.patch(
        "backend.server.v2.executions.review.routes.process_all_reviews_for_execution"
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

    mock_has_pending = mocker.patch(
        "backend.server.v2.executions.review.routes.has_pending_reviews_for_graph_exec"
    )
    mock_has_pending.return_value = False

    mocker.patch("backend.server.v2.executions.review.routes.add_graph_execution")

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

    response = client.post("/api/review/action", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert data["approved_count"] == 1
    assert data["rejected_count"] == 0
    assert data["failed_count"] == 0
    assert data["error"] is None


def test_process_review_action_reject_success(
    mocker: pytest_mock.MockFixture,
    sample_pending_review: PendingHumanReviewModel,
    test_user_id: str,
) -> None:
    """Test successful review rejection"""
    # Mock the route functions

    mock_get_reviews_for_execution = mocker.patch(
        "backend.server.v2.executions.review.routes.get_pending_reviews_for_execution"
    )
    mock_get_reviews_for_execution.return_value = [sample_pending_review]

    mock_process_all_reviews = mocker.patch(
        "backend.server.v2.executions.review.routes.process_all_reviews_for_execution"
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
        "backend.server.v2.executions.review.routes.has_pending_reviews_for_graph_exec"
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

    response = client.post("/api/review/action", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert data["approved_count"] == 0
    assert data["rejected_count"] == 1
    assert data["failed_count"] == 0
    assert data["error"] is None


def test_process_review_action_mixed_success(
    mocker: pytest_mock.MockFixture,
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

    mock_get_reviews_for_execution = mocker.patch(
        "backend.server.v2.executions.review.routes.get_pending_reviews_for_execution"
    )
    mock_get_reviews_for_execution.return_value = [sample_pending_review, second_review]

    mock_process_all_reviews = mocker.patch(
        "backend.server.v2.executions.review.routes.process_all_reviews_for_execution"
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

    mock_has_pending = mocker.patch(
        "backend.server.v2.executions.review.routes.has_pending_reviews_for_graph_exec"
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

    response = client.post("/api/review/action", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert data["approved_count"] == 1
    assert data["rejected_count"] == 1
    assert data["failed_count"] == 0
    assert data["error"] is None


def test_process_review_action_empty_request(
    mocker: pytest_mock.MockFixture,
    test_user_id: str,
) -> None:
    """Test error when no reviews provided"""
    request_data = {"reviews": []}

    response = client.post("/api/review/action", json=request_data)

    assert response.status_code == 422
    response_data = response.json()
    # Pydantic validation error format
    assert isinstance(response_data["detail"], list)
    assert len(response_data["detail"]) > 0
    assert "At least one review must be provided" in response_data["detail"][0]["msg"]


def test_process_review_action_review_not_found(
    mocker: pytest_mock.MockFixture,
    test_user_id: str,
) -> None:
    """Test error when review is not found"""
    # Mock the functions that extract graph execution ID from the request
    mock_get_reviews_for_execution = mocker.patch(
        "backend.server.v2.executions.review.routes.get_pending_reviews_for_execution"
    )
    mock_get_reviews_for_execution.return_value = []  # No reviews found

    # Mock process_all_reviews to simulate not finding reviews
    mock_process_all_reviews = mocker.patch(
        "backend.server.v2.executions.review.routes.process_all_reviews_for_execution"
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

    response = client.post("/api/review/action", json=request_data)

    assert response.status_code == 400
    assert "Reviews not found" in response.json()["detail"]


def test_process_review_action_partial_failure(
    mocker: pytest_mock.MockFixture,
    sample_pending_review: PendingHumanReviewModel,
    test_user_id: str,
) -> None:
    """Test handling of partial failures in review processing"""
    # Mock the route functions
    mock_get_reviews_for_execution = mocker.patch(
        "backend.server.v2.executions.review.routes.get_pending_reviews_for_execution"
    )
    mock_get_reviews_for_execution.return_value = [sample_pending_review]

    # Mock partial failure in processing
    mock_process_all_reviews = mocker.patch(
        "backend.server.v2.executions.review.routes.process_all_reviews_for_execution"
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

    response = client.post("/api/review/action", json=request_data)

    assert response.status_code == 400
    assert "Some reviews failed validation" in response.json()["detail"]


def test_process_review_action_invalid_node_exec_id(
    mocker: pytest_mock.MockFixture,
    sample_pending_review: PendingHumanReviewModel,
    test_user_id: str,
) -> None:
    """Test failure when trying to process review with invalid node execution ID"""
    # Mock the route functions
    mock_get_reviews_for_execution = mocker.patch(
        "backend.server.v2.executions.review.routes.get_pending_reviews_for_execution"
    )
    mock_get_reviews_for_execution.return_value = [sample_pending_review]

    # Mock validation failure - this should return 400, not 500
    mock_process_all_reviews = mocker.patch(
        "backend.server.v2.executions.review.routes.process_all_reviews_for_execution"
    )
    mock_process_all_reviews.side_effect = ValueError(
        "Invalid node execution ID format"
    )

    request_data = {
        "reviews": [
            {
                "node_exec_id": "invalid-node-format",
                "approved": True,
                "message": "Test",
            }
        ]
    }

    response = client.post("/api/review/action", json=request_data)

    # Should be a 400 Bad Request, not 500 Internal Server Error
    assert response.status_code == 400
    assert "Invalid node execution ID format" in response.json()["detail"]
