from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from autogpt_libs.auth.jwt_utils import get_jwt_payload

from backend.util.settings import AppEnvironment, BehaveAs

from .test_data_routes import router as test_data_router

app = fastapi.FastAPI()
app.include_router(test_data_router)

client = fastapi.testclient.TestClient(app)

_MODULE = "backend.api.features.admin.test_data_routes"
_ENDPOINT = "/admin/generate-test-data"


@pytest.fixture(autouse=True)
def setup_app_admin_auth(mock_jwt_admin):
    """Run every request as an authenticated admin user by default."""
    app.dependency_overrides[get_jwt_payload] = mock_jwt_admin["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def local_env(mocker: pytest_mock.MockerFixture):
    """Force the endpoint to consider itself running locally."""
    settings_mock = MagicMock()
    settings_mock.config.app_env = AppEnvironment.LOCAL
    settings_mock.config.behave_as = BehaveAs.LOCAL
    mocker.patch(f"{_MODULE}.settings", settings_mock)
    return settings_mock


@pytest.fixture
def non_local_env(mocker: pytest_mock.MockerFixture):
    """Force a shared/cloud environment where generation must be blocked."""
    settings_mock = MagicMock()
    settings_mock.config.app_env = AppEnvironment.DEVELOPMENT
    settings_mock.config.behave_as = BehaveAs.CLOUD
    mocker.patch(f"{_MODULE}.settings", settings_mock)
    return settings_mock


class _FakeE2ECreator:
    def __init__(self):
        # Distinct lengths so a field->attribute mapping bug is detectable.
        self.users = list(range(7))
        self.agent_graphs = list(range(5))
        self.library_agents = list(range(4))
        self.store_submissions = list(range(3))
        self.presets = list(range(2))
        self.api_keys = list(range(6))

    async def create_all_test_data(self):
        return None


@pytest.fixture
def fake_scripts(mocker: pytest_mock.MockerFixture):
    """Stub the file-path script loader with in-memory fake modules."""
    e2e_module = SimpleNamespace(TestDataCreator=_FakeE2ECreator)
    full_module = SimpleNamespace(main=AsyncMock(return_value=None))
    loader = mocker.patch(
        f"{_MODULE}._load_test_script",
        side_effect=lambda name: {
            "e2e_test_data": e2e_module,
            "test_data_creator": full_module,
        }[name],
    )
    prisma_mock = MagicMock()
    prisma_mock.is_connected.return_value = False
    prisma_mock.connect = AsyncMock()
    mocker.patch(f"{_MODULE}.prisma", prisma_mock)
    return SimpleNamespace(
        loader=loader, e2e=e2e_module, full=full_module, prisma=prisma_mock
    )


def test_e2e_generation_maps_all_detail_fields(local_env, fake_scripts):
    response = client.post(_ENDPOINT, json={"script_type": "e2e"})

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["message"] == "E2E test data generated successfully"
    assert body["details"] == {
        "users_created": 7,
        "graphs_created": 5,
        "library_agents_created": 4,
        "store_submissions_created": 3,
        "presets_created": 2,
        "api_keys_created": 6,
    }
    fake_scripts.prisma.connect.assert_awaited_once()


def test_defaults_to_e2e_when_script_type_omitted(local_env, fake_scripts):
    response = client.post(_ENDPOINT, json={})

    assert response.status_code == 200
    assert response.json()["message"] == "E2E test data generated successfully"


def test_full_generation_success(local_env, fake_scripts):
    response = client.post(_ENDPOINT, json={"script_type": "full"})

    assert response.status_code == 200
    assert response.json()["message"] == "Full test data generated successfully"
    fake_scripts.full.main.assert_awaited_once()


def test_generation_failure_returns_500(local_env, fake_scripts):
    fake_scripts.full.main.side_effect = RuntimeError("boom")

    response = client.post(_ENDPOINT, json={"script_type": "full"})

    assert response.status_code == 500
    assert "boom" not in response.json()["detail"]


def test_blocked_outside_local_returns_403_without_running_scripts(
    non_local_env, mocker: pytest_mock.MockerFixture
):
    loader = mocker.patch(f"{_MODULE}._load_test_script")
    alert = mocker.patch(f"{_MODULE}.discord_send_alert", new_callable=AsyncMock)

    response = client.post(_ENDPOINT, json={"script_type": "full"})

    assert response.status_code == 403
    assert "local" in response.json()["detail"].lower()
    alert.assert_awaited_once()
    loader.assert_not_called()


def test_block_still_returns_403_when_discord_alert_fails(
    non_local_env, mocker: pytest_mock.MockerFixture
):
    mocker.patch(
        f"{_MODULE}.discord_send_alert",
        new_callable=AsyncMock,
        side_effect=RuntimeError("discord down"),
    )

    response = client.post(_ENDPOINT, json={"script_type": "e2e"})

    assert response.status_code == 403


def test_non_admin_is_rejected(local_env, fake_scripts, mock_jwt_user):
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]

    response = client.post(_ENDPOINT, json={"script_type": "e2e"})

    assert response.status_code == 403
    fake_scripts.loader.assert_not_called()


def test_rejects_invalid_script_type(local_env):
    response = client.post(_ENDPOINT, json={"script_type": "not-a-real-script"})

    assert response.status_code == 422
