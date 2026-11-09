"""The voice call start route: who may call, and the start token it mints."""

import time
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import fastapi
import pytest
from fakeredis import FakeServer
from fakeredis.aioredis import FakeRedis
from httpx import ASGITransport, AsyncClient
from pytest_mock import MockerFixture

from backend.api.features.onboarding_dump import routes, voice
from backend.util.settings import AppEnvironment
from backend.voice_relay import locks, tokens
from backend.voice_relay.config import CAPS

app = fastapi.FastAPI()
app.include_router(routes.router)

URL = "/onboarding/brain-dump/interview/voice"
RELAY = "wss://api.example.com/voice"


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    from autogpt_libs.auth.dependencies import get_request_context
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    app.dependency_overrides[get_request_context] = mock_jwt_user["get_request_context"]
    yield
    app.dependency_overrides.clear()


@pytest.fixture
async def redis():
    client = FakeRedis(server=FakeServer(), decode_responses=True)
    yield client
    await client.aclose()


@pytest.fixture(autouse=True)
def voice_on(mocker: MockerFixture, monkeypatch, redis) -> SimpleNamespace:
    monkeypatch.setattr(voice.settings.config, "voice_relay_public_url", RELAY)
    monkeypatch.setattr(voice.settings.config, "voice_interview_max_concurrent", 25)
    monkeypatch.setattr(voice.settings.config, "voice_interview_daily_budget_usd", 25.0)
    monkeypatch.setattr(tokens.settings.secrets, "voice_relay_signing_key", "k" * 48)
    return SimpleNamespace(
        flag=mocker.patch.object(
            voice, "evaluate_feature_flag", AsyncMock(return_value=(True, True))
        ),
        edge=mocker.patch.object(
            voice, "get_feature_flag_value", AsyncMock(return_value="websocket")
        ),
        redis=mocker.patch.object(
            voice, "get_redis_async", AsyncMock(return_value=redis)
        ),
    )


async def start():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.post(URL)


async def test_the_route_is_not_there_while_voice_is_off(voice_on):
    voice_on.flag.return_value = (False, True)
    response = await start()
    assert response.status_code == 404


async def test_a_flag_that_cannot_be_read_keeps_voice_off(voice_on):
    voice_on.flag.side_effect = RuntimeError("flag service down")
    assert (await start()).status_code == 404


async def test_a_start_names_its_call_socket_caps_and_a_token_the_relay_trusts(
    test_user_id,
):
    response = await start()
    assert response.status_code == 200
    body = response.json()
    assert body["relay_url"] == f"{RELAY}/call"
    assert body["leg"] == {"kind": "websocket"}
    assert body["caps"] == CAPS.as_message()
    assert body["expires_in"] == 60
    claims = tokens.verify_start(body["token"])
    assert claims.user_id == test_user_id
    assert claims.call_id == body["call_id"]
    assert claims.leg == "websocket"


async def test_a_second_start_while_a_call_is_live_is_refused_naming_it(
    redis, test_user_id
):
    await locks.take_user_lock(redis, test_user_id, "a" * 32)
    response = await start()
    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "call_in_progress",
        "call_id": "a" * 32,
    }


async def test_past_the_daily_budget_voice_waits_for_the_next_day(redis):
    await locks.add_daily_spend(redis, 25.0, datetime.now(UTC))
    response = await start()
    assert response.status_code == 503
    wait = int(response.headers["Retry-After"])
    assert 0 < wait <= 86_400
    assert response.json()["detail"]["retry_after_seconds"] == wait


async def test_at_the_platforms_live_call_ceiling_a_start_waits(redis, monkeypatch):
    monkeypatch.setattr(voice.settings.config, "voice_interview_max_concurrent", 2)
    for call_id in ("a" * 32, "b" * 32):
        await locks.register_live_call(redis, call_id, time.time())
    response = await start()
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "30"
    assert response.json()["detail"]["retry_after_seconds"] == 30


async def test_calls_whose_pods_died_do_not_hold_the_ceiling(redis, monkeypatch):
    monkeypatch.setattr(voice.settings.config, "voice_interview_max_concurrent", 1)
    await locks.register_live_call(redis, "a" * 32, time.time() - 600)
    assert (await start()).status_code == 200


@pytest.mark.parametrize(
    "edge,leg",
    [("websocket", "websocket"), ("livekit", "websocket"), ("pigeon", "websocket")],
    ids=["websocket", "livekit-not-served", "unknown"],
)
async def test_the_edge_flag_picks_only_a_leg_this_deployment_serves(
    voice_on, edge, leg
):
    voice_on.edge.return_value = edge
    body = (await start()).json()
    assert body["leg"] == {"kind": leg}
    assert tokens.verify_start(body["token"]).leg == leg


async def test_an_edge_flag_that_cannot_be_read_falls_back_to_the_socket(voice_on):
    voice_on.edge.side_effect = RuntimeError("flag service down")
    assert (await start()).json()["leg"] == {"kind": "websocket"}


@pytest.mark.parametrize(
    "url,key,app_env",
    [
        ("", "k" * 48, AppEnvironment.PRODUCTION),
        (RELAY, "", AppEnvironment.PRODUCTION),
        (RELAY, "short", AppEnvironment.PRODUCTION),
        ("ws://relay.internal/voice", "k" * 48, AppEnvironment.PRODUCTION),
    ],
    ids=["no-relay", "no-key", "short-key", "plain-ws-outside-local"],
)
async def test_voice_is_unavailable_until_a_relay_and_its_key_are_set(
    monkeypatch, url, key, app_env
):
    monkeypatch.setattr(voice.settings.config, "voice_relay_public_url", url)
    monkeypatch.setattr(voice.settings.config, "app_env", app_env)
    monkeypatch.setattr(tokens.settings.secrets, "voice_relay_signing_key", key)
    response = await start()
    assert response.status_code == 503
    assert response.json()["detail"]["retry_after_seconds"] == 300


async def test_a_local_install_may_reach_its_relay_over_plain_ws(monkeypatch):
    monkeypatch.setattr(
        voice.settings.config, "voice_relay_public_url", "ws://localhost:8015/voice/"
    )
    monkeypatch.setattr(voice.settings.config, "app_env", AppEnvironment.LOCAL)
    response = await start()
    assert response.json()["relay_url"] == "ws://localhost:8015/voice/call"


async def test_a_start_is_refused_while_its_limits_cannot_be_read(voice_on):
    voice_on.redis.side_effect = ConnectionError("redis is down")
    response = await start()
    assert response.status_code == 503
    assert response.json()["detail"]["retry_after_seconds"] == 30
