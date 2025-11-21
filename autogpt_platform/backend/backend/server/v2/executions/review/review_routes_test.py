import datetime

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from prisma.enums import ReviewStatus
from pytest_snapshot.plugin import Snapshot

from backend.server.v2.executions.review.model import PendingHumanReviewModel
from backend.server.v2.executions.review.routes import router

# Using a fixed timestamp for reproducible tests
FIXED_NOW = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)

app = fastapi.FastAPI()
app.include_router(router, prefix="/api/review")

client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    """Setup auth overrides for all tests in this module"""
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def sample_pending_review() -> PendingHumanReviewModel:
    """Create a sample pending review for testing"""
    return PendingHumanReviewModel(
        node_exec_id="test_node_123",
        user_id="test_user",
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
) -> None:
    """Test getting pending reviews when none exist"""
    mock_get_reviews = mocker.patch(
        "backend.server.v2.executions.review.routes.get_pending_reviews_for_user"
    )
    mock_get_reviews.return_value = []

    response = client.get("/api/review/pending")

    assert response.status_code == 200
    assert response.json() == []
    mock_get_reviews.assert_called_once_with("test_user", 1, 25)


def test_get_pending_reviews_with_data(
    mocker: pytest_mock.MockFixture,
    sample_pending_review: PendingHumanReviewModel,
    snapshot: Snapshot,
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
    mock_get_reviews.assert_called_once_with("test_user", 2, 10)


def test_get_pending_reviews_for_execution_success(
    mocker: pytest_mock.MockFixture,
    sample_pending_review: PendingHumanReviewModel,
    snapshot: Snapshot,
) -> None:
    """Test getting pending reviews for specific execution"""
    mock_get_graph_execution = mocker.patch(
        "backend.server.v2.executions.review.routes.get_graph_execution_meta"
    )
    mock_get_graph_execution.return_value = {
        "id": "test_graph_exec_456",
        "user_id": "test_user",
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
) -> None:
    """Test successful review approval"""
    # Mock the validation functions
    mock_get_pending_review = mocker.patch(
        "backend.data.human_review.get_pending_review_by_node_exec_id"
    )
    mock_get_pending_review.return_value = sample_pending_review

    mock_get_reviews_for_execution = mocker.patch(
        "backend.server.v2.executions.review.routes.get_pending_reviews_for_execution"
    )
    mock_get_reviews_for_execution.return_value = [sample_pending_review]

    mock_process_all_reviews = mocker.patch(
        "backend.server.v2.executions.review.routes.process_all_reviews_for_execution"
    )
    mock_process_all_reviews.return_value = {"test_node_123": sample_pending_review}

    mock_has_pending = mocker.patch(
        "backend.data.human_review.has_pending_reviews_for_graph_exec"
    )
    mock_has_pending.return_value = False

    mocker.patch("backend.executor.utils.add_graph_execution")

    request_data = {
        "approved_reviews": [
            {
                "node_exec_id": "test_node_123",
                "message": "Looks good",
                "reviewed_data": {"data": "modified payload", "value": 50},
            }
        ],
        "rejected_review_ids": [],
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
) -> None:
    """Test successful review rejection"""
    # Mock the validation functions
    mock_get_pending_review = mocker.patch(
        "backend.data.human_review.get_pending_review_by_node_exec_id"
    )
    mock_get_pending_review.return_value = sample_pending_review

    mock_get_reviews_for_execution = mocker.patch(
        "backend.server.v2.executions.review.routes.get_pending_reviews_for_execution"
    )
    mock_get_reviews_for_execution.return_value = [sample_pending_review]

    mock_process_all_reviews = mocker.patch(
        "backend.server.v2.executions.review.routes.process_all_reviews_for_execution"
    )
    rejected_review = PendingHumanReviewModel(
        node_exec_id="test_node_123",
        user_id="test_user",
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
        "backend.data.human_review.has_pending_reviews_for_graph_exec"
    )
    mock_has_pending.return_value = False

    request_data = {"approved_reviews": [], "rejected_review_ids": ["test_node_123"]}

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
) -> None:
    """Test mixed approve/reject operations"""
    # Create a second review
    second_review = PendingHumanReviewModel(
        node_exec_id="test_node_456",
        user_id="test_user",
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

    # Mock the validation functions
    mock_get_pending_review = mocker.patch(
        "backend.data.human_review.get_pending_review_by_node_exec_id"
    )
    mock_get_pending_review.side_effect = lambda node_id, user_id: (
        sample_pending_review if node_id == "test_node_123" else second_review
    )

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
        user_id="test_user",
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
        user_id="test_user",
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
        "backend.data.human_review.has_pending_reviews_for_graph_exec"
    )
    mock_has_pending.return_value = False

    request_data = {
        "approved_reviews": [
            {
                "node_exec_id": "test_node_123",
                "message": "Approved",
                "reviewed_data": {"data": "modified"},
            }
        ],
        "rejected_review_ids": ["test_node_456"],
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
) -> None:
    """Test error when no reviews provided"""
    request_data = {"approved_reviews": [], "rejected_review_ids": []}

    response = client.post("/api/review/action", json=request_data)

    assert response.status_code == 400
    assert "At least one review must be provided" in response.json()["detail"]


def test_process_review_action_review_not_found(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Test error when review is not found"""
    mock_get_pending_review = mocker.patch(
        "backend.data.human_review.get_pending_review_by_node_exec_id"
    )
    mock_get_pending_review.return_value = None

    request_data = {
        "approved_reviews": [
            {
                "node_exec_id": "nonexistent_node",
                "message": "Test",
            }
        ],
        "rejected_review_ids": [],
    }

    response = client.post("/api/review/action", json=request_data)

    assert response.status_code == 403
    assert "not found or access denied" in response.json()["detail"]


def test_process_review_action_partial_failure(
    mocker: pytest_mock.MockFixture,
    sample_pending_review: PendingHumanReviewModel,
) -> None:
    """Test handling of partial failures in review processing"""
    # Mock successful validation
    mock_get_pending_review = mocker.patch(
        "backend.data.human_review.get_pending_review_by_node_exec_id"
    )
    mock_get_pending_review.return_value = sample_pending_review

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
        "approved_reviews": [
            {
                "node_exec_id": "test_node_123",
                "message": "Test",
            }
        ],
        "rejected_review_ids": [],
    }

    response = client.post("/api/review/action", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert data["approved_count"] == 0
    assert data["rejected_count"] == 0
    assert data["failed_count"] == 1
    assert "Failed to process reviews" in data["error"]


def test_process_review_action_complete_failure(
    mocker: pytest_mock.MockFixture,
    sample_pending_review: PendingHumanReviewModel,
) -> None:
    """Test complete failure scenario"""
    # Mock successful validation
    mock_get_pending_review = mocker.patch(
        "backend.data.human_review.get_pending_review_by_node_exec_id"
    )
    mock_get_pending_review.return_value = sample_pending_review

    mock_get_reviews_for_execution = mocker.patch(
        "backend.server.v2.executions.review.routes.get_pending_reviews_for_execution"
    )
    mock_get_reviews_for_execution.return_value = [sample_pending_review]

    # Mock complete failure in processing
    mock_process_all_reviews = mocker.patch(
        "backend.server.v2.executions.review.routes.process_all_reviews_for_execution"
    )
    mock_process_all_reviews.side_effect = Exception("Database error")

    request_data = {
        "approved_reviews": [
            {
                "node_exec_id": "test_node_123",
                "message": "Test",
            }
        ],
        "rejected_review_ids": [],
    }

    response = client.post("/api/review/action", json=request_data)

    assert response.status_code == 500
    assert "error" in response.json()["detail"].lower()
