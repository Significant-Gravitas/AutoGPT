"""Tests for the experiment assignment API."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock

from backend.data.experiments import Experiment, ExperimentAssignment

from .experiments import router as experiments_router

app = fastapi.FastAPI()
app.include_router(experiments_router)

client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


def _assignment(variant: str = "yearly-pro") -> ExperimentAssignment:
    return ExperimentAssignment(
        experiment_key="subscription-pricing-page-initial-state",
        variant=variant,
        source="posthog",
        assigned_at=datetime(2026, 9, 2, tzinfo=UTC),
    )


def test_get_assignment_assigns_for_the_caller(
    mocker: pytest_mock.MockFixture, test_user_id: str
) -> None:
    assign = mocker.patch(
        "backend.api.features.experiments.experiments_db.assign",
        new_callable=AsyncMock,
        return_value=_assignment(),
    )

    response = client.get("/assignments/subscription-pricing-page-initial-state")

    assert response.status_code == 200, response.text
    assert response.json()["variant"] == "yearly-pro"
    assign.assert_awaited_once_with(test_user_id, Experiment.SUBSCRIPTION_PRICING)


def test_get_assignment_refuses_a_flag_that_is_not_a_declared_experiment(
    mocker: pytest_mock.MockFixture,
) -> None:
    assign = mocker.patch(
        "backend.api.features.experiments.experiments_db.assign", new_callable=AsyncMock
    )

    response = client.get("/assignments/copilot-model-routing")

    assert response.status_code == 422
    assign.assert_not_awaited()


def test_list_assignments(mocker: pytest_mock.MockFixture, test_user_id: str) -> None:
    listing = mocker.patch(
        "backend.api.features.experiments.experiments_db.list_assignments",
        new_callable=AsyncMock,
        return_value=[_assignment("control")],
    )

    response = client.get("/assignments")

    assert response.status_code == 200, response.text
    assert [row["variant"] for row in response.json()] == ["control"]
    listing.assert_awaited_once_with(test_user_id)
