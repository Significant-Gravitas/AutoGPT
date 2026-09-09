"""The skills-hub gate in front of the marketplace skill routes."""

import fastapi
import fastapi.testclient
import pytest

from . import skill_routes

app = fastapi.FastAPI()
app.include_router(skill_routes.router, prefix="/api/store/skills")
client = fastapi.testclient.TestClient(app)

# ``require_skills_hub_flag`` runs the real ``is_feature_enabled``, which
# consults this env override before LaunchDarkly — so the gate is driven here
# exactly the way it is driven locally, with no mock in the path.
FLAG_ENV_VAR = "FORCE_FLAG_SKILLS_HUB"


@pytest.mark.parametrize(
    "call",
    [
        lambda: client.get("/api/store/skills"),
        lambda: client.get("/api/store/skills/any-slug"),
        lambda: client.post("/api/store/skills/any-slug/install"),
    ],
    ids=["browse", "detail", "install"],
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
