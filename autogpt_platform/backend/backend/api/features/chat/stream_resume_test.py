"""``GET /sessions/{id}/stream?turn&after`` over the recorded turns, on real Redis.

Each recorded turn is stored back into Redis as a finished turn of a session
the test user owns; the route must serve exactly its committed frames.
"""

import json
import uuid

import fastapi
import pytest
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from httpx import ASGITransport, AsyncClient

from backend.api.features.chat import routes as chat_routes
from backend.copilot import stream_registry
from backend.copilot.stream_drift.recording import RecordedTurn, load_fixture
from backend.data import redis_client
from backend.util.testing import is_tcp_port_reachable

pytestmark = pytest.mark.skipif(
    not is_tcp_port_reachable(redis_client.HOST, redis_client.PORT),
    reason="no local Redis reachable; the stream registry needs one to run",
)

RECORDED_TURNS = ["dummy-text-turn", "baseline-tool-turn", "sdk-late-tool-result"]
DONE = "data: [DONE]\n\n"

app = fastapi.FastAPI()
app.include_router(chat_routes.router)


@pytest.fixture(autouse=True)
def auth(mock_jwt_user):
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


@pytest.fixture
async def stored(test_user_id):
    """Store a recorded turn as a finished turn; returns its session and turn."""
    keys: list[str] = []

    async def store(turn: RecordedTurn) -> tuple[str, str]:
        session_id, turn_id = f"resume-{uuid.uuid4().hex}", str(uuid.uuid4())
        await stream_registry.create_session(
            session_id, test_user_id, "", "", turn_id=turn_id
        )
        redis = await redis_client.get_redis_async()
        stream_key = stream_registry._get_turn_stream_key(turn_id)
        for entry in turn.entries:
            await redis.xadd(
                stream_key, {"data": json.dumps(entry["data"])}, id=entry["id"]
            )
        meta_key = stream_registry.get_session_meta_key(session_id)
        await redis.hset(meta_key, "status", "completed")
        keys.extend([meta_key, stream_key, stream_registry._get_turn_meta_key(turn_id)])
        return session_id, turn_id

    yield store
    redis = await redis_client.get_redis_async()
    for key in keys:
        await redis.delete(key)


@pytest.mark.parametrize("name", RECORDED_TURNS)
async def test_a_cursor_read_serves_exactly_the_recorded_frames_after_it(name, stored):
    turn = load_fixture(name)
    session_id, turn_id = await stored(turn)
    ids = [entry["id"] for entry in turn.entries]

    for position, after in enumerate(["0-0", *ids[:-1]]):
        response = await _get_stream(session_id, turn=turn_id, after=after)

        assert response.status_code == 200
        assert response.text == _frames(turn, turn_id, ids[position:]) + DONE


async def test_a_cursor_before_the_trim_is_answered_with_the_checkpoint(stored):
    turn = load_fixture("baseline-tool-turn")
    [checkpoint] = [e for e in turn.entries if e["data"]["type"] == "data-checkpoint"]
    session_id, turn_id = await stored(turn)
    redis = await redis_client.get_redis_async()
    stream_key = stream_registry._get_turn_stream_key(turn_id)
    await redis.xtrim(stream_key, minid=checkpoint["id"], approximate=False)

    trimmed = await _get_stream(session_id, turn=turn_id, after="3-0")
    resumed = await _get_stream(session_id, turn=turn_id, after=checkpoint["id"])

    assert trimmed.status_code == 409
    assert trimmed.json() == {
        "reason": "trimmed",
        "checkpoint": {
            "entry_id": checkpoint["id"],
            "rows": checkpoint["data"]["rows"],
            "sequence": checkpoint["data"]["sequence"],
        },
    }
    ids = [entry["id"] for entry in turn.entries]
    after = ids[ids.index(checkpoint["id"]) + 1 :]
    assert resumed.text == _frames(turn, turn_id, after) + DONE


async def test_a_stream_that_is_gone_is_answered_410(stored):
    session_id, _ = await stored(load_fixture("dummy-text-turn"))

    response = await _get_stream(session_id, turn=str(uuid.uuid4()), after="0-0")

    assert response.status_code == 410
    assert response.json() == {"reason": "expired", "checkpoint": None}


async def _get_stream(session_id: str, **params: str):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.get(f"/sessions/{session_id}/stream", params=params)


def _frames(turn: RecordedTurn, turn_id: str, ids: list[str]) -> str:
    wanted = set(ids)
    return "".join(
        frame["sse"].replace("id: drift-turn:", f"id: {turn_id}:")
        for frame in turn.frames
        if frame["id"] in wanted
    )
