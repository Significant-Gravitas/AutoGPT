import datetime
from unittest import mock

import autogpt_libs.auth.middleware
import fastapi
import fastapi.testclient
import prisma.enums
import prisma.models

import market.app

client = fastapi.testclient.TestClient(market.app.app)


async def override_auth_middleware(request: fastapi.Request):
    return {"sub": "3e53486c-cf57-477e-ba2a-cb02dc828e1a", "role": "admin"}


market.app.app.dependency_overrides[autogpt_libs.auth.middleware.auth_middleware] = (
    override_auth_middleware
)


def test_get_submissions():
    with mock.patch("market.db.get_agents") as mock_get_agents:
        mock_get_agents.return_value = {
            "agents": [],
            "total_count": 0,
            "page": 1,
            "page_size": 10,
            "total_pages": 0,
        }
        response = client.get(
            "/api/v1/market/admin/agent/submissions?page=1&page_size=10&description_threshold=60&sort_by=createdAt&sort_order=desc",
            headers={"Bearer": ""},
        )
        assert response.status_code == 200
        assert response.json() == {
            "agents": [],
            "total_count": 0,
            "page": 1,
            "page_size": 10,
            "total_pages": 0,
        }


def test_review_submission():
    with mock.patch("market.db.update_agent_entry") as mock_update_agent_entry:
        mock_update_agent_entry.return_value = prisma.models.Agents(
            id="aaa-bbb-ccc",
            version=1,
            createdAt=datetime.datetime.fromisoformat("2021-10-01T00:00:00+00:00"),
            updatedAt=datetime.datetime.fromisoformat("2021-10-01T00:00:00+00:00"),
            submissionStatus=prisma.enums.SubmissionStatus.APPROVED,
            submissionDate=datetime.datetime.fromisoformat("2021-10-01T00:00:00+00:00"),
            submissionReviewComments="Looks good",
            submissionReviewDate=datetime.datetime.fromisoformat(
                "2021-10-01T00:00:00+00:00"
            ),
            keywords=["test"],
            categories=["test"],
            graph='{"name": "test", "description": "test"}',  # type: ignore
        )
        response = client.post(
            "/api/v1/market/admin/agent/submissions",
            headers={
                "Authorization": "Bearer token"
            },  # Assuming you need an authorization token
            json={
                "agent_id": "aaa-bbb-ccc",
                "version": 1,
                "status": "APPROVED",
                "comments": "Looks good",
            },
        )
        assert response.status_code == 200
