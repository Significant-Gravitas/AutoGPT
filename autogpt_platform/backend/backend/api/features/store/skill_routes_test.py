"""The skills-hub gate in front of the marketplace skill routes."""

import fastapi
import fastapi.testclient
import pytest

from . import skill_routes, skill_submission_db

app = fastapi.FastAPI()
app.include_router(skill_routes.router, prefix="/api/store/skills")
client = fastapi.testclient.TestClient(app)

# ``require_skills_hub_flag`` runs the real ``is_feature_enabled``, which
# consults this env override before LaunchDarkly — so the gate is driven here
# exactly the way it is driven locally, with no mock in the path.
FLAG_ENV_VAR = "FORCE_FLAG_SKILLS_HUB"


@pytest.fixture
def authenticated(mock_jwt_user):
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


@pytest.mark.parametrize(
    "call",
    [
        lambda: client.get("/api/store/skills"),
        lambda: client.get("/api/store/skills/any-slug"),
        lambda: client.post("/api/store/skills/any-slug/install"),
        lambda: client.post("/api/store/skills/submissions"),
        lambda: client.get("/api/store/skills/submissions"),
        lambda: client.put("/api/store/skills/submissions/any-version"),
    ],
    ids=["browse", "detail", "install", "submit", "my-submissions", "edit"],
)
def test_every_endpoint_is_404_when_the_flag_is_off(
    monkeypatch: pytest.MonkeyPatch, call
):
    monkeypatch.setenv(FLAG_ENV_VAR, "false")

    response = call()

    assert response.status_code == 404
    assert response.json()["detail"] == "Feature not available"


def test_install_still_needs_auth_when_the_flag_is_on(
    monkeypatch: pytest.MonkeyPatch,
):
    """The gate takes an optional user, so it must not stand in for auth."""
    monkeypatch.setenv(FLAG_ENV_VAR, "true")

    response = client.post("/api/store/skills/any-slug/install")

    assert response.status_code == 401


async def test_the_gate_opens_when_the_flag_is_on(monkeypatch: pytest.MonkeyPatch):
    """Without this the 404s above would also pass on a permanently shut gate.

    Called directly rather than through the router: past the gate the handler
    reaches Prisma, which the sync TestClient drives on the wrong event loop.
    """
    monkeypatch.setenv(FLAG_ENV_VAR, "true")

    assert await skill_routes.require_skills_hub_flag(user_id=None) is None


def test_submissions_is_a_route_of_its_own_not_a_skill_slug(
    monkeypatch: pytest.MonkeyPatch, mocker, authenticated
):
    """`/{slug}` is declared after `/submissions`, so the placeholder cannot
    swallow it — FastAPI matches in declaration order."""
    monkeypatch.setenv(FLAG_ENV_VAR, "true")
    listed = mocker.patch.object(
        skill_submission_db, "list_my_skill_submissions", return_value=[]
    )
    detail = mocker.patch.object(skill_routes.skill_db, "get_marketplace_skill")

    response = client.get("/api/store/skills/submissions")

    assert response.status_code == 200
    listed.assert_awaited_once()
    detail.assert_not_called()
