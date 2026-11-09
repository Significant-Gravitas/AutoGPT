"""The voice call routes: who may call and the start token it mints, then the
call's state, its finish and ending it, each for the person's own call only."""

import asyncio
import time
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import fastapi
import jwt
import posthog.client
import pytest
import urllib3.connectionpool
from fakeredis import FakeServer
from fakeredis.aioredis import FakeRedis
from httpx import ASGITransport, AsyncClient
from livekit import api
from posthog import Posthog
from prisma.enums import BrainDumpInputMode, BrainDumpStatus, SubscriptionTier
from pytest_mock import MockerFixture

import backend.util.feature_flag as ff
import backend.util.feature_flag.posthog as ph
from backend.api.features.onboarding_dump import (
    interview,
    interview_endings,
    quality,
    rate_limit,
    routes,
    voice,
    voice_handoff,
)
from backend.api.features.onboarding_dump.models import FinalizeResponse
from backend.copilot.model import (
    INTERVIEW_STAGE_CLOSED,
    INTERVIEW_STAGE_HANDED_OFF,
    INTERVIEW_STAGE_WRAP_UP_ISSUED,
    ONBOARDING_INTERVIEW_KIND,
    ChatSession,
)
from backend.data.experiments import Experiment
from backend.util.settings import AppEnvironment, BehaveAs
from backend.voice_relay import locks, tokens
from backend.voice_relay.config import CAPS

app = fastapi.FastAPI()
app.include_router(routes.router)

URL = "/onboarding/brain-dump/interview/voice"
RELAY = "wss://api.example.com/voice"
LIVEKIT_URL = "wss://agpt-test.livekit.cloud"
LIVEKIT_KEY = "APIroutes"
LIVEKIT_SECRET = "q" * 48  # pragma: allowlist secret
# The control socket's leg, which carries no room.
SOCKET = {"kind": "websocket", "url": None, "token": None}


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
        # Someone who has not opened the typed interview.
        interview=mocker.patch.object(
            voice, "find_onboarding_interview_session", AsyncMock(return_value=None)
        ),
        # Nor had a call take it.
        taken_by=mocker.patch.object(
            voice, "call_that_took_the_interview", AsyncMock(return_value=None)
        ),
        latest=mocker.patch.object(
            voice, "get_onboarding_voice_call_id", AsyncMock(return_value=None)
        ),
        # Not enrolled in the voice ending experiment.
        ending=mocker.patch.object(
            voice, "assigned_ending", AsyncMock(return_value=None)
        ),
    )


def _interview(user_id: str, stage: str | None) -> ChatSession:
    session = ChatSession.new(user_id, dry_run=False)
    session.metadata.kind = ONBOARDING_INTERVIEW_KIND
    session.metadata.onboarding_interview_stage = stage
    return session


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
    assert body["leg"] == SOCKET
    assert body["caps"] == CAPS.as_message()
    assert body["expires_in"] == 60
    claims = tokens.verify_start(body["token"])
    assert claims.user_id == test_user_id
    assert claims.call_id == body["call_id"]
    assert claims.leg == "websocket"
    assert claims.ending is None


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


@pytest.fixture
def livekit_on(monkeypatch) -> None:
    monkeypatch.setattr(voice.settings.config, "livekit_url", LIVEKIT_URL)
    monkeypatch.setattr(voice.settings.secrets, "livekit_api_key", LIVEKIT_KEY)
    monkeypatch.setattr(voice.settings.secrets, "livekit_api_secret", LIVEKIT_SECRET)


@pytest.mark.parametrize(
    "edge,leg",
    [("livekit", "livekit"), ("websocket", "websocket"), ("pigeon", "websocket")],
    ids=["livekit", "websocket", "unknown"],
)
async def test_the_edge_flag_picks_the_leg_the_start_token_names(
    voice_on, livekit_on, edge, leg
):
    voice_on.edge.return_value = edge
    body = (await start()).json()
    assert body["leg"]["kind"] == leg
    assert tokens.verify_start(body["token"]).leg == leg


async def test_the_livekit_leg_names_the_project_and_the_browsers_token_for_the_room(
    voice_on, livekit_on, test_user_id
):
    voice_on.edge.return_value = "livekit"
    body = (await start()).json()

    leg = body["leg"]
    assert leg["url"] == LIVEKIT_URL
    claims = api.TokenVerifier(LIVEKIT_KEY, LIVEKIT_SECRET).verify(leg["token"])
    assert claims.identity == f"user:{test_user_id}"
    assert claims.video is not None
    assert claims.video.room == f"voice-{body['call_id']}"
    assert claims.video.can_publish_sources == ["microphone"]
    assert claims.video.can_publish_data is False
    assert not claims.video.room_admin
    lifetime = jwt.decode(
        leg["token"], LIVEKIT_SECRET, algorithms=["HS256"], issuer=LIVEKIT_KEY
    )
    assert lifetime["exp"] - lifetime["nbf"] == 600


async def test_the_livekit_leg_falls_back_to_the_socket_where_there_is_no_project(
    voice_on, monkeypatch
):
    voice_on.edge.return_value = "livekit"
    monkeypatch.setattr(voice.settings.config, "livekit_url", "")
    body = (await start()).json()
    assert body["leg"] == SOCKET
    assert tokens.verify_start(body["token"]).leg == "websocket"


async def test_an_edge_flag_that_cannot_be_read_falls_back_to_the_socket(
    voice_on, livekit_on
):
    voice_on.edge.side_effect = RuntimeError("flag service down")
    assert (await start()).json()["leg"] == SOCKET


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


@pytest.mark.parametrize("stage", [INTERVIEW_STAGE_CLOSED, INTERVIEW_STAGE_HANDED_OFF])
async def test_a_person_whose_interview_is_over_cannot_call(
    voice_on, test_user_id, stage
):
    voice_on.interview.return_value = _interview(test_user_id, stage)
    response = await start()
    assert response.status_code == 409
    assert response.json()["detail"] == {"code": "onboarding_interview_closed"}
    voice_on.interview.assert_awaited_once_with(test_user_id)


@pytest.mark.parametrize("stage", [None, INTERVIEW_STAGE_WRAP_UP_ISSUED])
async def test_an_interview_still_under_way_can_move_to_a_call(
    voice_on, test_user_id, stage
):
    voice_on.interview.return_value = _interview(test_user_id, stage)
    assert (await start()).status_code == 200


async def test_an_interview_that_cannot_be_read_holds_the_call_back(voice_on):
    voice_on.interview.side_effect = ConnectionError("database is down")
    response = await start()
    assert response.status_code == 503
    assert response.json()["detail"]["retry_after_seconds"] == 30


@pytest.fixture
def paying_platform(mocker: MockerFixture, set_platform_payment, redis) -> AsyncMock:
    """Calls are counted: the platform takes payment and runs as the cloud."""
    mocker.patch.object(rate_limit.settings.config, "behave_as", BehaveAs.CLOUD)
    set_platform_payment(True)
    mocker.patch.object(rate_limit, "get_redis_async", AsyncMock(return_value=redis))
    return mocker.patch.object(
        rate_limit,
        "get_user_subscription_tier",
        AsyncMock(return_value=SubscriptionTier.PRO),
    )


async def test_past_their_call_allowance_a_person_waits_with_the_wait_in_the_body(
    paying_platform,
):
    for _ in range(rate_limit.VOICE_HOURLY_CALLS):
        assert (await start()).status_code == 200
    response = await start()
    assert response.status_code == 429
    wait = int(response.headers["Retry-After"])
    assert 0 < wait <= 3600
    assert response.json()["detail"]["retry_after_seconds"] == wait
    # Counted whatever the plan: the tier is never asked.
    paying_platform.assert_not_awaited()


async def test_a_refused_start_spends_none_of_the_allowance(
    paying_platform, redis, test_user_id
):
    await locks.take_user_lock(redis, test_user_id, "a" * 32)
    for _ in range(rate_limit.VOICE_HOURLY_CALLS + 1):
        assert (await start()).status_code == 409
    await locks.release_user_lock(redis, test_user_id, "a" * 32)
    assert (await start()).status_code == 200


async def test_a_person_whose_call_took_the_interview_cannot_call_again(voice_on):
    voice_on.taken_by.return_value = "a" * 32
    response = await start()
    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "onboarding_interview_closed",
        "call_id": "a" * 32,
    }


# ---------- the call after it starts ----------

CALL = "c" * 32
OTHER_USER = "someone-else"


def _take(code: str | None = None) -> FinalizeResponse:
    return FinalizeResponse(
        status=BrainDumpStatus.failed if code else BrainDumpStatus.transcribed,
        input_mode=BrainDumpInputMode.typed,
        error_code=code,
    )


async def _record(redis, uid: str, call_id: str = CALL, **fields: object) -> None:
    await locks.write_call_record(
        redis, call_id, {"uid": uid, "leg": "websocket", **fields}
    )


async def request(method: str, path: str, **kwargs):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.request(method, f"{URL}{path}", **kwargs)


@pytest.fixture
def hand_off(mocker: MockerFixture) -> AsyncMock:
    """The hand-off a finish runs: accepted, unless a test says otherwise."""

    async def accept(redis, record, background, **kwargs):
        # The person's finish makes an accepted take the interview's.
        closure = interview.InterviewClosure(ending=None, hand_off=_take())
        await locks.write_call_record(
            redis,
            record.call_id,
            {
                "status": locks.HANDED_OFF,
                "hand_off": _take().model_dump(mode="json"),
                "interview": locks.INTERVIEW_TAKEN,
            },
        )
        return closure

    return mocker.patch.object(
        voice, "hand_off_call", new=AsyncMock(side_effect=accept)
    )


async def test_a_calls_state_moves_as_its_record_does(redis, test_user_id):
    seen = []
    for fields in (
        {"status": "connecting"},
        {"status": "live"},
        {"status": "ended", "end_reason": "finished"},
        {"status": "transcribing"},
        {"status": "handed_off", "hand_off": _take().model_dump(mode="json")},
    ):
        await _record(redis, test_user_id, **fields)
        body = (await request("GET", f"/{CALL}")).json()
        seen.append((body["status"], body["end_reason"], body["hand_off"]))

    assert seen == [
        ("connecting", None, None),
        ("live", None, None),
        ("ended", "finished", None),
        ("transcribing", "finished", None),
        ("handed_off", "finished", _take().model_dump(mode="json")),
    ]
    body = (await request("GET", f"/{CALL}")).json()
    assert body["call_id"] == CALL
    assert body["leg"] == "websocket"


async def test_a_live_calls_state_names_the_socket_to_rejoin_it_on(
    voice_on, redis, test_user_id
):
    """A tab holding the call's resume ticket rejoins on the socket the
    server names, never one it kept itself; an ended call names none."""
    for status, relay_url in (
        ("connecting", f"{RELAY}/call"),
        ("live", f"{RELAY}/call"),
        ("ended", None),
        ("transcribing", None),
    ):
        await _record(redis, test_user_id, status=status)
        body = (await request("GET", f"/{CALL}")).json()
        assert (status, body["relay_url"]) == (status, relay_url)


async def test_someone_elses_call_is_not_found(redis):
    await _record(redis, OTHER_USER, status="live")
    assert (await request("GET", f"/{CALL}")).status_code == 404
    assert (await request("GET", f"/{'d' * 32}")).status_code == 404


async def test_a_call_id_that_is_not_one_is_refused(redis):
    assert (await request("GET", "/not-a-call")).status_code == 422


async def test_a_reloaded_step_finds_the_persons_latest_call(
    voice_on, redis, test_user_id
):
    assert (await request("GET", "")).status_code == 404

    voice_on.latest.return_value = CALL
    await _record(redis, test_user_id, status="transcribed")
    body = (await request("GET", "")).json()

    assert (body["call_id"], body["status"]) == (CALL, "transcribed")
    voice_on.latest.assert_awaited_with(test_user_id)


async def test_finishing_a_call_that_took_the_interview_answers_its_take(
    redis, test_user_id, hand_off: AsyncMock
):
    await _record(
        redis,
        test_user_id,
        status="handed_off",
        hand_off=_take().model_dump(mode="json"),
        transcript_of_record="I run a bakery.",
        interview=locks.INTERVIEW_TAKEN,
    )

    response = await request("POST", f"/{CALL}/finish")

    assert response.status_code == 200
    assert response.json()["status"] == "transcribed"
    hand_off.assert_not_awaited()


async def test_finishing_a_call_whose_words_were_turned_away_answers_the_verdict(
    redis, test_user_id, hand_off: AsyncMock
):
    """The person calls again or types instead; the same words would only be
    turned away again."""
    await _record(
        redis,
        test_user_id,
        status="failed",
        hand_off=_take(quality.INSUFFICIENT_CONTENT).model_dump(mode="json"),
        transcript_of_record="idk",
    )

    response = await request("POST", f"/{CALL}/finish")

    assert response.json()["error_code"] == quality.INSUFFICIENT_CONTENT
    hand_off.assert_not_awaited()


@pytest.mark.parametrize(
    ("status", "stored"),
    [
        ("transcribed", None),
        ("failed", _take(quality.QUALITY_CHECK_UNAVAILABLE)),
        ("failed", _take("rate_limited")),
    ],
    ids=["confirm", "judge-unavailable", "rate-limited"],
)
async def test_a_finish_hands_off_a_call_the_pipeline_never_ruled_on(
    redis,
    test_user_id,
    hand_off: AsyncMock,
    status: str,
    stored: FinalizeResponse | None,
):
    await _record(
        redis,
        test_user_id,
        status=status,
        hand_off=stored.model_dump(mode="json") if stored else None,
        transcript_of_record="I run a bakery.",
    )

    response = await request(
        "POST", f"/{CALL}/finish", json={"corrections": "Eight staff."}
    )

    assert response.status_code == 200
    assert response.json()["status"] == "transcribed"
    hand_off.assert_awaited_once()
    record = hand_off.await_args.args[1]
    assert (record.call_id, record.uid) == (CALL, test_user_id)
    kwargs = hand_off.await_args.kwargs
    # The person's own finish, under the finish's hold on the call.
    assert (kwargs["confirmed"], kwargs["by_person"], kwargs["corrections"]) == (
        True,
        True,
        "Eight staff.",
    )
    assert isinstance(kwargs["claim"], voice_handoff.HandOffClaim)
    assert kwargs["claim"].call_id == CALL
    # The finish gave the call back once it was done with it.
    assert await redis.get(locks.postcall_claim_key(CALL)) is None
    again = await request("POST", f"/{CALL}/finish")
    assert again.json()["status"] == "transcribed"
    hand_off.assert_awaited_once()


@pytest.mark.parametrize("status", ["live", "ended", "transcribing"])
async def test_a_call_whose_words_are_not_ready_cannot_be_finished(
    redis, test_user_id, hand_off: AsyncMock, status: str
):
    await _record(redis, test_user_id, status=status)

    response = await request("POST", f"/{CALL}/finish")

    assert response.status_code == 409
    assert response.json()["detail"] == {"code": "call_not_ready"}
    hand_off.assert_not_awaited()


async def test_a_finish_already_running_holds_the_next_one_back(
    redis, test_user_id, hand_off: AsyncMock
):
    await _record(redis, test_user_id, status="transcribed", transcript_of_record="Hi.")
    await locks.claim_postcall(redis, CALL, "finish:another-request")

    response = await request("POST", f"/{CALL}/finish")

    assert response.status_code == 409
    hand_off.assert_not_awaited()


async def test_a_finish_whose_claim_reply_was_lost_gives_the_claim_back(
    redis, test_user_id, hand_off: AsyncMock, mocker: MockerFixture
):
    """The claim was written and the wait for its reply gave up: left, the
    claim would hold every finish back until it lapsed."""
    await _record(redis, test_user_id, status="transcribed", transcript_of_record="Hi.")
    claim = locks.claim_postcall

    async def written_then_lost(redis, call_id: str, holder: str, **kwargs):
        await claim(redis, call_id, holder, **kwargs)
        await asyncio.sleep(10)

    mocker.patch.object(locks, "claim_postcall", new=written_then_lost)

    response = await request("POST", f"/{CALL}/finish")

    assert response.status_code == 503
    hand_off.assert_not_awaited()
    assert await redis.get(locks.postcall_claim_key(CALL)) is None


async def test_someone_else_cannot_finish_a_call(redis, hand_off: AsyncMock):
    await _record(
        redis, OTHER_USER, status="transcribed", transcript_of_record="I run a bakery."
    )

    response = await request("POST", f"/{CALL}/finish")

    assert response.status_code == 404
    hand_off.assert_not_awaited()


async def test_a_finish_the_pipeline_turns_away_for_now_carries_its_wait(
    redis, test_user_id, hand_off: AsyncMock
):
    hand_off.side_effect = fastapi.HTTPException(
        status_code=429, detail="Slow down", headers={"Retry-After": "120"}
    )
    await _record(redis, test_user_id, status="transcribed", transcript_of_record="Hi.")

    response = await request("POST", f"/{CALL}/finish")

    assert response.status_code == 429
    assert response.json()["detail"]["retry_after_seconds"] == 120
    assert await redis.get(locks.postcall_claim_key(CALL)) is None


async def test_ending_a_call_of_ones_own_asks_its_pod_to_end_it(redis, test_user_id):
    await _record(redis, test_user_id, status="live")

    response = await request("POST", f"/{CALL}/end")

    assert response.status_code == 202
    assert await locks.end_requested(redis, CALL)


async def test_someone_else_cannot_end_a_call(redis):
    await _record(redis, OTHER_USER, status="live")
    await locks.take_user_lock(redis, OTHER_USER, CALL)

    response = await request("POST", f"/{CALL}/end")

    assert response.status_code == 404
    assert not await locks.end_requested(redis, CALL)


async def test_a_live_call_whose_record_was_lost_can_still_be_ended(
    redis, test_user_id
):
    await locks.take_user_lock(redis, test_user_id, CALL)

    response = await request("POST", f"/{CALL}/end")

    assert response.status_code == 202
    assert await locks.end_requested(redis, CALL)


async def test_ending_a_call_that_is_not_there_is_not_found(redis):
    assert (await request("POST", f"/{CALL}/end")).status_code == 404
    assert not await locks.end_requested(redis, CALL)


# ---------- the arm a call carries ----------

VOICE_ENDING = Experiment.ONBOARDING_INTERVIEW_ENDING_VOICE.value


def _refuse_network(pool, method, url, *args, **kwargs):
    raise urllib3.exceptions.ProtocolError(f"no network in tests: {method} {url}")


@pytest.fixture
def voice_ending_flag(mocker: MockerFixture, monkeypatch) -> dict:
    """A real PostHog client holding the voice ending experiment, everyone on
    the profile card, with no network; and the experiment table in memory."""
    monkeypatch.setattr(
        urllib3.connectionpool.HTTPConnectionPool, "urlopen", _refuse_network
    )
    client = Posthog(
        "phc_test",
        host="http://127.0.0.1:9",
        secret_key="phx_test",
        enable_local_evaluation=False,
        send=False,
        sync_mode=True,
    )
    client.feature_flags = [
        {
            "id": 1,
            "key": VOICE_ENDING,
            "active": True,
            "filters": {
                "groups": [{"properties": [], "rollout_percentage": 100}],
                "multivariate": {
                    "variants": [{"key": "02-profile-card", "rollout_percentage": 100}]
                },
            },
        }
    ]
    mocker.patch.object(ph, "get_flag_client", return_value=client)
    mocker.patch.object(posthog.client, "flags", side_effect=AssertionError)
    mocker.patch.object(
        ff,
        "_fetch_user_context",
        AsyncMock(side_effect=lambda user_id: ff._anonymous_context(user_id)),
    )
    rows: dict = {}

    async def find_unique(where):
        key = where["userId_experimentKey"]
        return rows.get((key["userId"], key["experimentKey"]))

    async def create(data):
        row = SimpleNamespace(**data, createdAt=datetime.now(UTC))
        rows[(row.userId, row.experimentKey)] = row
        return row

    table = mocker.patch("prisma.models.ExperimentAssignment.prisma")
    table.return_value.find_unique = AsyncMock(side_effect=find_unique)
    table.return_value.create = AsyncMock(side_effect=create)
    return rows


async def test_a_start_reads_the_persons_ending_arm_and_its_token_carries_it(
    mocker: MockerFixture, voice_ending_flag: dict, test_user_id
):
    """The relay reads no flags, so the arm reaches the call this way."""
    mocker.patch.object(voice, "assigned_ending", new=interview_endings.assigned_ending)

    body = (await start()).json()

    assert tokens.verify_start(body["token"]).ending == "02-profile-card"
    # Stored on the first read: the arm the person saw stays theirs.
    assert voice_ending_flag[(test_user_id, VOICE_ENDING)].variant == (
        "02-profile-card"
    )


@pytest.mark.parametrize(
    "arm",
    [
        interview_endings.InterviewEnding.SUMMARY_CHIPS.value,
        interview_endings.InterviewEnding.COVERAGE_RING.value,
        "99-an-arm-from-a-later-release",
    ],
)
async def test_a_start_carries_no_arm_the_voice_mode_does_not_offer(voice_on, arm):
    """The call ends as an unenrolled one does, rather than on a screen the
    call step does not have."""
    voice_on.ending.return_value = arm

    body = (await start()).json()

    assert tokens.verify_start(body["token"]).ending is None
    voice_on.ending.assert_awaited_once()
    assert voice_on.ending.await_args.args[1] == "voice"


# ---------- a call that no longer runs ----------


async def test_a_call_whose_pod_died_does_not_hold_the_next_start_back(
    redis, test_user_id
):
    await locks.take_user_lock(redis, test_user_id, "a" * 32)
    await locks.register_live_call(
        redis, "a" * 32, time.time() - locks.LIVE_CALL_STALE_S - 1
    )
    await _record(redis, test_user_id, "a" * 32, status="live")

    assert (await start()).status_code == 200


async def test_ending_a_call_whose_pod_died_frees_the_persons_lock(redis, test_user_id):
    await locks.take_user_lock(redis, test_user_id, CALL)
    await locks.register_live_call(
        redis, CALL, time.time() - locks.LIVE_CALL_STALE_S - 1
    )
    await _record(redis, test_user_id, status="live")

    response = await request("POST", f"/{CALL}/end")

    assert response.status_code == 202
    assert await locks.live_call_of(redis, test_user_id) is None


async def test_ending_a_running_call_leaves_its_lock_to_its_pod(redis, test_user_id):
    await locks.take_user_lock(redis, test_user_id, CALL)
    await locks.register_live_call(redis, CALL, time.time())
    await _record(redis, test_user_id, status="live")

    response = await request("POST", f"/{CALL}/end")

    assert response.status_code == 202
    assert await locks.end_requested(redis, CALL)
    assert await locks.live_call_of(redis, test_user_id) == CALL


async def test_a_calls_state_says_who_owns_the_interview(redis, test_user_id):
    await _record(redis, test_user_id, status="transcribing", end_reason="finished")
    assert (await request("GET", f"/{CALL}")).json()["interview"] is None

    await locks.claim_interview(redis, CALL, locks.INTERVIEW_SUPERSEDED)

    assert (await request("GET", f"/{CALL}")).json()["interview"] == "superseded"


async def test_a_call_owed_its_hand_off_reads_handing_off_and_starts_it(
    redis, test_user_id, mocker: MockerFixture
):
    hand_off = mocker.patch.object(
        voice_handoff, "hand_off_owed", new=AsyncMock(return_value="handed_off")
    )
    await _record(
        redis,
        test_user_id,
        status="transcribed",
        transcript_of_record="I run a bakery.",
        end_reason="finished",
    )

    body = (await request("GET", f"/{CALL}")).json()

    assert body["status"] == "handing_off"
    for _ in range(50):
        if hand_off.await_count:
            break
        await asyncio.sleep(0.01)
    hand_off.assert_awaited_once_with(CALL, trigger="status")


async def test_a_confirm_arms_call_reads_transcribed_and_starts_nothing(
    redis, test_user_id, mocker: MockerFixture
):
    hand_off = mocker.patch.object(voice_handoff, "hand_off_owed", new=AsyncMock())
    await _record(
        redis,
        test_user_id,
        status="transcribed",
        transcript_of_record="I run a bakery.",
        ending="02-profile-card",
    )

    body = (await request("GET", f"/{CALL}")).json()

    assert body["status"] == "transcribed"
    await asyncio.sleep(0.05)
    hand_off.assert_not_awaited()
