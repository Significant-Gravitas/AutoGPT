import fastapi
import fastapi.testclient
import pytest

from . import skill_routes, skill_submission_db

app = fastapi.FastAPI()
app.include_router(skill_routes.router, prefix="/api/store/skills")
client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


def test_submissions_is_a_route_of_its_own_not_a_skill_slug(mocker):
    """`/{slug}` is declared after `/submissions`, so the placeholder cannot
    swallow it — FastAPI matches in declaration order."""
    listed = mocker.patch.object(
        skill_submission_db, "list_my_skill_submissions", return_value=[]
    )
    detail = mocker.patch.object(skill_routes.skill_db, "get_marketplace_skill")

    response = client.get("/api/store/skills/submissions")

    assert response.status_code == 200
    listed.assert_awaited_once()
    detail.assert_not_called()
