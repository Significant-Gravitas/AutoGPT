import sys
import types
from unittest.mock import AsyncMock, MagicMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from autogpt_libs.auth.jwt_utils import get_jwt_payload

from backend.util.settings import AppEnvironment

from .test_data_routes import router as test_data_router

app = fastapi.FastAPI()
app.include_router(test_data_router)

client = fastapi.testclient.TestClient(app)

_MODULE = "backend.api.features.admin.test_data_routes"
_ENDPOINT = "/admin/generate-test-data"


@pytest.fixture(autouse=True)
def setup_app_admin_auth(mock_jwt_admin):
    """Run every request as an authenticated admin user."""
    app.dependency_overrides[get_jwt_payload] = mock_jwt_admin["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def set_app_env(mocker: pytest_mock.MockerFixture):
    def _set(env: AppEnvironment):
        settings_mock = MagicMock()
        settings_mock.config.app_env = env
        mocker.patch(f"{_MODULE}.settings", settings_mock)

    return _set


class _FakeE2ECreator:
    def __init__(self):
        self.users = list(range(15))
        self.agent_graphs = list(range(15))
        self.library_agents = list(range(15))
        self.store_submissions = list(range(15))
        self.presets = list(range(15))
        self.api_keys = list(range(15))

    async def create_all_test_data(self):
        return None


@pytest.fixture
def fake_test_scripts():
    """Inject importable stand-ins for the dynamically imported test scripts."""
    e2e_module = types.ModuleType("e2e_test_data")
    e2e_module.TestDataCreator = _FakeE2ECreator  # type: ignore[attr-defined]
    sys.modules["e2e_test_data"] = e2e_module

    full_module = types.ModuleType("test_data_creator")
    full_module.main = AsyncMock(return_value=None)  # type: ignore[attr-defined]
    sys.modules["test_data_creator"] = full_module

    yield e2e_module, full_module

    sys.modules.pop("e2e_test_data", None)
    sys.modules.pop("test_data_creator", None)


@pytest.fixture
def mock_prisma(mocker: pytest_mock.MockerFixture):
    prisma_mock = MagicMock()
    prisma_mock.is_connected.return_value = False
    prisma_mock.connect = AsyncMock()
    mocker.patch("backend.data.db.prisma", prisma_mock)
    return prisma_mock


def test_blocks_generation_in_production(
    set_app_env, mocker: pytest_mock.MockerFixture
):
    set_app_env(AppEnvironment.PRODUCTION)
    alert = mocker.patch(f"{_MODULE}.discord_send_alert", new_callable=AsyncMock)

    response = client.post(_ENDPOINT, json={"script_type": "e2e"})

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert "production" in body["message"].lower()
    alert.assert_awaited_once()


def test_e2e_generation_success(set_app_env, fake_test_scripts, mock_prisma):
    set_app_env(AppEnvironment.LOCAL)

    response = client.post(_ENDPOINT, json={"script_type": "e2e"})

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["message"] == "E2E test data generated successfully"
    assert body["details"]["users_created"] == 15
    # is_connected() returned False, so the route should connect first.
    mock_prisma.connect.assert_awaited_once()


def test_defaults_to_e2e_when_script_type_omitted(
    set_app_env, fake_test_scripts, mock_prisma
):
    set_app_env(AppEnvironment.LOCAL)

    response = client.post(_ENDPOINT, json={})

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["message"] == "E2E test data generated successfully"


def test_full_generation_success(set_app_env, fake_test_scripts):
    set_app_env(AppEnvironment.LOCAL)
    _, full_module = fake_test_scripts

    response = client.post(_ENDPOINT, json={"script_type": "full"})

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["message"] == "Full test data generated successfully"
    full_module.main.assert_awaited_once()


def test_returns_failure_when_generation_raises(
    set_app_env, fake_test_scripts, mock_prisma
):
    set_app_env(AppEnvironment.LOCAL)
    e2e_module, _ = fake_test_scripts

    class _ExplodingCreator(_FakeE2ECreator):
        async def create_all_test_data(self):
            raise RuntimeError("boom")

    e2e_module.TestDataCreator = _ExplodingCreator

    response = client.post(_ENDPOINT, json={"script_type": "e2e"})

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert "Failed to generate test data" in body["message"]


def test_rejects_invalid_script_type(set_app_env):
    set_app_env(AppEnvironment.LOCAL)

    response = client.post(_ENDPOINT, json={"script_type": "not-a-real-script"})

    assert response.status_code == 422
