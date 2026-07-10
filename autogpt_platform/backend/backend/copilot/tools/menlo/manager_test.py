"""Tests for the Menlo connection manager.

The Menlo SDK is an optional extra, so these tests never import it — the lazy
``_new_client`` / ``_connect`` seams are patched, and SDK presence for
``menlo_available`` is simulated via ``sys.modules``.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, ValidationError

from backend.copilot.tools.menlo import manager


class _RequiresField(BaseModel):
    value: int


def _validation_error() -> ValidationError:
    """Produce a real ValidationError (missing required field)."""
    try:
        _RequiresField.model_validate({})
    except ValidationError as exc:
        return exc
    raise AssertionError("expected ValidationError")  # pragma: no cover


class FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value

    async def delete(self, key):
        self.store.pop(key, None)


@pytest.fixture(autouse=True)
def clear_live():
    manager._LIVE.clear()
    yield
    manager._LIVE.clear()


@pytest.fixture
def fake_redis():
    redis = FakeRedis()
    with patch.object(manager, "get_redis_async", AsyncMock(return_value=redis)):
        yield redis


def _fake_client():
    client = MagicMock()
    created = MagicMock()
    created.robot.id = "rb_123"
    client.robots.create = AsyncMock(return_value=created)
    client.robots.delete = AsyncMock()
    client.aclose = AsyncMock()
    return client


def _settings_with_key(key: str):
    settings = MagicMock()
    settings.secrets.menlo_api_key = key
    settings.config.menlo_rcs_url = "https://rcs.example/rcs"
    settings.config.menlo_robot_viewer_url = "https://viewer.example"
    return settings


class TestMenloAvailable:
    def test_no_key_is_unavailable(self):
        with patch.object(manager, "Settings", return_value=_settings_with_key("")):
            assert manager.menlo_available() is False

    def test_key_and_sdk_present_is_available(self):
        fake_mod = MagicMock()
        with patch.dict(sys.modules, {"menlo_robot_sdk": fake_mod}), patch.object(
            manager, "Settings", return_value=_settings_with_key("sk_live_x")
        ):
            assert manager.menlo_available() is True

    def test_key_but_sdk_missing_is_unavailable(self):
        # sys.modules[name] = None makes `import name` raise ImportError.
        with patch.dict(sys.modules, {"menlo_robot_sdk": None}), patch.object(
            manager, "Settings", return_value=_settings_with_key("sk_live_x")
        ):
            assert manager.menlo_available() is False


class TestConnectNewRobot:
    @pytest.mark.asyncio
    async def test_creates_stores_and_caches(self, fake_redis):
        client = _fake_client()
        session = MagicMock()
        with patch.object(manager, "_new_client", return_value=client), patch.object(
            manager, "_connect", AsyncMock(return_value=session)
        ):
            conn = await manager.connect_new_robot(
                "sess1", model="asimov-v0", name="bot"
            )

        assert conn.robot_id == "rb_123"
        assert conn.session is session
        client.robots.create.assert_awaited_once()
        assert fake_redis.store[manager._robot_key("sess1")] == "rb_123"
        assert manager._LIVE["sess1"] is conn

    @pytest.mark.asyncio
    async def test_closes_client_on_connect_failure(self, fake_redis):
        client = _fake_client()
        with patch.object(manager, "_new_client", return_value=client), patch.object(
            manager, "_connect", AsyncMock(side_effect=RuntimeError("boom"))
        ):
            with pytest.raises(RuntimeError):
                await manager.connect_new_robot("sess1", name="bot")

        client.aclose.assert_awaited_once()
        assert "sess1" not in manager._LIVE


class TestCreateRobotWorkaround:
    @pytest.mark.asyncio
    async def test_returns_id_on_normal_create(self):
        client = _fake_client()
        robot_id = await manager._create_robot(client, name="bot", model="asimov-v0")
        assert robot_id == "rb_123"

    @pytest.mark.asyncio
    async def test_recovers_id_via_list_on_validation_error(self):
        # SDK 0.2.2 raises ValidationError (missing pin_code) though the robot
        # was created; the manager recovers the id by name via list().
        client = MagicMock()
        client.robots.create = AsyncMock(side_effect=_validation_error())
        robot = MagicMock()
        robot.name = "copilot-x"
        robot.id = "rb_recovered"
        listing = MagicMock()
        listing.robots = [robot]
        client.robots.list = AsyncMock(return_value=listing)

        robot_id = await manager._create_robot(client, name="copilot-x", model=None)
        assert robot_id == "rb_recovered"

    @pytest.mark.asyncio
    async def test_reraises_when_robot_not_found(self):
        client = MagicMock()
        client.robots.create = AsyncMock(side_effect=_validation_error())
        listing = MagicMock()
        listing.robots = []
        client.robots.list = AsyncMock(return_value=listing)

        with pytest.raises(ValidationError):
            await manager._create_robot(client, name="copilot-x", model=None)


class TestResolveConnection:
    @pytest.mark.asyncio
    async def test_returns_cached_without_reconnect(self, fake_redis):
        conn = manager.LiveConnection("rb_1", _fake_client(), MagicMock())
        manager._LIVE["sess1"] = conn
        with patch.object(manager, "_connect", AsyncMock()) as connect_mock:
            resolved = await manager.resolve_connection("sess1")
        assert resolved is conn
        connect_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_reconnects_from_stored_robot_id(self, fake_redis):
        fake_redis.store[manager._robot_key("sess1")] = "rb_9"
        client = _fake_client()
        session = MagicMock()
        with patch.object(manager, "_new_client", return_value=client), patch.object(
            manager, "_connect", AsyncMock(return_value=session)
        ):
            conn = await manager.resolve_connection("sess1")
        assert conn.robot_id == "rb_9"
        assert manager._LIVE["sess1"] is conn

    @pytest.mark.asyncio
    async def test_raises_when_no_robot(self, fake_redis):
        with pytest.raises(manager.MenloNotConnectedError):
            await manager.resolve_connection("sess1")

    @pytest.mark.asyncio
    async def test_forgets_robot_when_reconnect_fails(self, fake_redis):
        fake_redis.store[manager._robot_key("sess1")] = "rb_gone"
        client = _fake_client()
        with patch.object(manager, "_new_client", return_value=client), patch.object(
            manager, "_connect", AsyncMock(side_effect=RuntimeError("404"))
        ):
            with pytest.raises(RuntimeError):
                await manager.resolve_connection("sess1")
        assert manager._robot_key("sess1") not in fake_redis.store


class TestDisconnectRobot:
    @pytest.mark.asyncio
    async def test_tears_down_live_connection(self, fake_redis):
        fake_redis.store[manager._robot_key("sess1")] = "rb_1"
        client = _fake_client()
        session = MagicMock()
        session.disconnect = AsyncMock()
        conn = manager.LiveConnection("rb_1", client, session)
        manager._LIVE["sess1"] = conn

        robot_id = await manager.disconnect_robot("sess1")

        assert robot_id == "rb_1"
        session.disconnect.assert_awaited_once()
        client.robots.delete.assert_awaited_once_with("rb_1")
        assert "sess1" not in manager._LIVE
        assert manager._robot_key("sess1") not in fake_redis.store

    @pytest.mark.asyncio
    async def test_deletes_robot_without_live_handle(self, fake_redis):
        fake_redis.store[manager._robot_key("sess1")] = "rb_2"
        client = _fake_client()
        with patch.object(manager, "_new_client", return_value=client):
            robot_id = await manager.disconnect_robot("sess1")
        assert robot_id == "rb_2"
        client.robots.delete.assert_awaited_once_with("rb_2")
        assert manager._robot_key("sess1") not in fake_redis.store
