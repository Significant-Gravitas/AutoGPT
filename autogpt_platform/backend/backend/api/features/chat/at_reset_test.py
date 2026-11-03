"""Route tests for /sessions/{session_id}/at-reset: setting, listing and
cancelling a resume or send at the budget reset.

The session checks, the budget reads, the scheduler and Redis (the chat's job
lock) are mocked at the boundary; ``backend/copilot/at_reset_test.py`` covers
when the turn fires.
"""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import fastapi
import fastapi.testclient
import httpx
import pytest
import pytest_mock
from autogpt_libs.auth.dependencies import get_request_context
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from redis.exceptions import ConnectionError as RedisConnectionError

from backend.api.features.chat import at_reset as at_reset_routes
from backend.copilot import at_reset_lock
from backend.copilot.at_reset import NoResetAhead
from backend.copilot.at_reset_lock_fixtures import FakeAtResetRedis
from backend.copilot.continuation import RESUMED_AT_RESET_NOTE, as_model_prompt
from backend.copilot.rate_limit import RateLimitUnavailable
from backend.executor.scheduler import CopilotTurnJobInfo
from backend.util.exceptions import NotFoundError

app = fastapi.FastAPI()
app.include_router(at_reset_routes.router)


@app.exception_handler(NotFoundError)
async def _not_found_handler(
    request: fastapi.Request, exc: NotFoundError
) -> fastapi.responses.JSONResponse:
    """Mirror the production NotFoundError → 404 mapping from the REST app."""
    return fastapi.responses.JSONResponse(status_code=404, content={"detail": str(exc)})


client = fastapi.testclient.TestClient(app)

SESSION = "sess-1"
RUN_AT = datetime(2026, 10, 15, 0, 9, 14, tzinfo=UTC)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    app.dependency_overrides[get_request_context] = mock_jwt_user["get_request_context"]
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def scheduler(mocker: pytest_mock.MockerFixture) -> MagicMock:
    scheduler_client = MagicMock(
        add_copilot_turn_schedule=AsyncMock(),
        get_execution_schedules=AsyncMock(return_value=[]),
        cancel_at_reset_turn=AsyncMock(return_value=True),
    )
    mocker.patch.object(
        at_reset_routes, "get_scheduler_client", return_value=scheduler_client
    )
    return scheduler_client


@pytest.fixture
def session(mocker: pytest_mock.MockerFixture) -> MagicMock:
    chat = MagicMock(
        session_id=SESSION, organization_id="org-1", team_id="team-1", expert_id=None
    )
    chat.metadata.llm_auth_provider = "platform"
    for check in ("_validate_and_get_writable_session", "_validate_and_get_session"):
        mocker.patch.object(at_reset_routes, check, AsyncMock(return_value=chat))
    return chat


@pytest.fixture
def run_at(mocker: pytest_mock.MockerFixture) -> AsyncMock:
    return mocker.patch.object(
        at_reset_routes, "next_reset_run_at", AsyncMock(return_value=RUN_AT)
    )


@pytest.fixture(autouse=True)
def job_locks(mocker: pytest_mock.MockerFixture) -> FakeAtResetRedis:
    """Redis, where each chat's at-reset job is locked across replicas."""
    redis = FakeAtResetRedis()
    mocker.patch.object(at_reset_lock, "get_redis_async", AsyncMock(return_value=redis))
    return redis


def _when(body: dict) -> datetime:
    return datetime.fromisoformat(body["run_at"])


@pytest.mark.parametrize(
    "method, path, body",
    [
        ("GET", f"/sessions/{SESSION}/at-reset", None),
        ("POST", f"/sessions/{SESSION}/at-reset", {"kind": "send", "message": "x"}),
        ("DELETE", f"/sessions/{SESSION}/at-reset/send", None),
    ],
)
def test_every_scheduler_call_is_repeated_if_lost_in_transit(
    method, path, body, scheduler, session, run_at, mocker: pytest_mock.MockerFixture
):
    # A chat lists these turns each time it opens. In e2e, a pooled connection
    # the scheduler had just closed failed that load with a 500.
    factory = mocker.patch.object(
        at_reset_routes, "get_scheduler_client", return_value=scheduler
    )

    response = client.request(method, path, json=body)

    assert response.status_code < 300
    assert factory.call_args_list
    assert all(c.kwargs == {"should_retry": True} for c in factory.call_args_list)


def test_resume_at_reset(scheduler, session, run_at, test_user_id):
    response = client.post(
        f"/sessions/{SESSION}/at-reset", json={"kind": "resume", "message": "x"}
    )

    assert response.status_code == 200
    body = response.json()
    assert (body["kind"], _when(body), body["message"]) == ("resume", RUN_AT, None)
    run_at.assert_awaited_once_with(test_user_id, SESSION, "resume")
    job = scheduler.add_copilot_turn_schedule.await_args.kwargs
    assert job["purpose"] == "resume"
    assert job["schedule_id"] == f"at-reset-resume-{SESSION}"
    assert job["session_id"] == SESSION
    assert job["run_at"] == RUN_AT
    # The listing shows a readable line; what fires is the note itself.
    assert job["message"] == as_model_prompt(RESUMED_AT_RESET_NOTE)
    assert (job["organization_id"], job["team_id"]) == ("org-1", "team-1")


def test_send_at_reset_keeps_the_users_message(scheduler, session, run_at):
    response = client.post(
        f"/sessions/{SESSION}/at-reset",
        json={"kind": "send", "message": "also check the invoices"},
    )

    assert response.status_code == 200
    assert response.json()["message"] == "also check the invoices"
    job = scheduler.add_copilot_turn_schedule.await_args.kwargs
    assert job["purpose"] == "send"
    assert job["schedule_id"] == f"at-reset-send-{SESSION}"
    assert job["message"] == "also check the invoices"


def test_a_second_send_joins_the_message_already_waiting(scheduler, session, run_at):
    # One send per chat: replacing the waiting one would lose what the user
    # wrote first.
    scheduler.get_execution_schedules.return_value = [
        _job("send", "draft the launch post")
    ]
    response = client.post(
        f"/sessions/{SESSION}/at-reset",
        json={"kind": "send", "message": "also the changelog"},
    )

    assert response.status_code == 200
    joined = "draft the launch post\n\nalso the changelog"
    assert response.json()["message"] == joined
    assert scheduler.add_copilot_turn_schedule.await_args.kwargs["message"] == joined


class _ChatJobs:
    """The scheduler's jobs for the chat, in memory. A listing reads them and
    then waits to be let go, the window two writers race in."""

    def __init__(self) -> None:
        self.jobs: dict[str, CopilotTurnJobInfo] = {}
        self.listed = asyncio.Event()
        self.let_go = asyncio.Event()

    async def get_execution_schedules(self, **_) -> list[CopilotTurnJobInfo]:
        jobs = list(self.jobs.values())
        self.listed.set()
        await self.let_go.wait()
        return jobs

    async def add_copilot_turn_schedule(
        self, *, purpose: str, message: str, schedule_id: str, **_
    ) -> None:
        self.jobs[schedule_id] = _job(purpose, message)


@pytest.mark.asyncio
async def test_two_sends_at_once_both_end_up_in_the_waiting_message(
    session, run_at, mocker: pytest_mock.MockerFixture
):
    # Two tabs, or two API replicas: each read the waiting message before the
    # other set its own, so the second set replaced the first.
    jobs = _ChatJobs()
    mocker.patch.object(at_reset_routes, "get_scheduler_client", return_value=jobs)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as http:

        async def send(message: str) -> httpx.Response:
            return await http.post(
                f"/sessions/{SESSION}/at-reset",
                json={"kind": "send", "message": message},
            )

        first = asyncio.create_task(send("draft the launch post"))
        await jobs.listed.wait()
        second = asyncio.create_task(send("also the changelog"))
        await asyncio.sleep(0.2)
        jobs.let_go.set()
        responses = await asyncio.gather(first, second)

    assert [r.status_code for r in responses] == [200, 200]
    (waiting,) = jobs.jobs.values()
    assert waiting.message == "draft the launch post\n\nalso the changelog"


def test_a_joined_send_longer_than_a_message_is_413(scheduler, session, run_at):
    scheduler.get_execution_schedules.return_value = [_job("send", "x" * 60_000)]
    response = client.post(
        f"/sessions/{SESSION}/at-reset",
        json={"kind": "send", "message": "y" * 5_000},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "message_too_long"
    scheduler.add_copilot_turn_schedule.assert_not_awaited()


def test_a_resume_set_again_replaces_it(scheduler, session, run_at):
    scheduler.get_execution_schedules.return_value = [
        _job("resume", "Resumed after the budget reset.")
    ]
    response = client.post(f"/sessions/{SESSION}/at-reset", json={"kind": "resume"})

    assert response.status_code == 200
    scheduler.get_execution_schedules.assert_not_awaited()


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "send"},
        {"kind": "send", "message": "   "},
        {"kind": "send", "message": "x" * 64_001},
        {"kind": "later"},
    ],
)
def test_a_send_needs_a_message_and_a_known_kind(scheduler, session, run_at, payload):
    response = client.post(f"/sessions/{SESSION}/at-reset", json=payload)

    assert response.status_code == 422
    scheduler.add_copilot_turn_schedule.assert_not_awaited()


def test_an_unknown_session_is_404(
    scheduler, run_at, mocker: pytest_mock.MockerFixture
):
    mocker.patch.object(
        at_reset_routes,
        "_validate_and_get_writable_session",
        AsyncMock(side_effect=NotFoundError(f"Session {SESSION} not found.")),
    )

    response = client.post(f"/sessions/{SESSION}/at-reset", json={"kind": "resume"})

    assert response.status_code == 404
    run_at.assert_not_awaited()
    scheduler.add_copilot_turn_schedule.assert_not_awaited()


@pytest.mark.parametrize("reason", ["not_out_of_budget", "no_reset"])
def test_nothing_to_wait_for_is_409(scheduler, session, run_at, reason):
    run_at.side_effect = NoResetAhead(reason)

    response = client.post(f"/sessions/{SESSION}/at-reset", json={"kind": "resume"})

    assert response.status_code == 409
    assert response.json()["detail"] == reason
    scheduler.add_copilot_turn_schedule.assert_not_awaited()


def test_a_chat_off_the_platform_route_has_nothing_to_wait_for(
    scheduler, session, run_at
):
    session.metadata.llm_auth_provider = "codex"

    response = client.post(
        f"/sessions/{SESSION}/at-reset", json={"kind": "send", "message": "hi"}
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "not_out_of_budget"
    run_at.assert_not_awaited()


def test_unreadable_usage_is_503(scheduler, session, run_at):
    run_at.side_effect = RateLimitUnavailable()

    response = client.post(f"/sessions/{SESSION}/at-reset", json={"kind": "resume"})

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "30"


@pytest.mark.parametrize(
    "method, path, body",
    [
        ("POST", f"/sessions/{SESSION}/at-reset", {"kind": "send", "message": "x"}),
        ("POST", f"/sessions/{SESSION}/at-reset", {"kind": "resume"}),
        ("DELETE", f"/sessions/{SESSION}/at-reset/send", None),
    ],
)
def test_a_job_it_cannot_lock_is_503_and_left_alone(
    method, path, body, scheduler, session, run_at, mocker: pytest_mock.MockerFixture
):
    # Going on without the lock could lose a send another tab just wrote.
    mocker.patch.object(
        at_reset_lock,
        "get_redis_async",
        AsyncMock(side_effect=RedisConnectionError("down")),
    )

    response = client.request(method, path, json=body)

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "30"
    scheduler.get_execution_schedules.assert_not_awaited()
    scheduler.add_copilot_turn_schedule.assert_not_awaited()
    scheduler.cancel_at_reset_turn.assert_not_awaited()


def _job(purpose: str, message: str) -> CopilotTurnJobInfo:
    return CopilotTurnJobInfo.model_validate(
        {
            "schedule_id": f"{purpose}-job",
            "user_id": "user-1",
            "session_id": SESSION,
            "message": message,
            "run_at": RUN_AT,
            "purpose": purpose,
            "id": f"{purpose}-job",
            "name": purpose,
            "next_run_time": RUN_AT.isoformat(),
        }
    )


def test_list_shows_the_chats_resume_and_send_only(scheduler, session, test_user_id):
    scheduler.get_execution_schedules.return_value = [
        _job("followup", "check CI"),
        _job("resume", "Resumed after the budget reset."),
        _job("send", "also check the invoices"),
    ]

    response = client.get(f"/sessions/{SESSION}/at-reset")

    assert response.status_code == 200
    jobs = response.json()["jobs"]
    assert [(j["kind"], j["message"]) for j in jobs] == [
        ("resume", None),
        ("send", "also check the invoices"),
    ]
    assert all(_when(j) == RUN_AT for j in jobs)
    scheduler.get_execution_schedules.assert_awaited_once_with(
        user_id=test_user_id, session_id=SESSION, kind="copilot_turn"
    )


@pytest.mark.parametrize("removed", [True, False])
def test_cancel_is_204_whether_or_not_one_was_set(
    scheduler, session, test_user_id, removed
):
    scheduler.cancel_at_reset_turn.return_value = removed

    response = client.delete(f"/sessions/{SESSION}/at-reset/send")

    assert response.status_code == 204
    scheduler.cancel_at_reset_turn.assert_awaited_once_with(
        user_id=test_user_id, session_id=SESSION, kind="send"
    )


def test_a_cancel_holds_the_jobs_lock_while_it_cancels(scheduler, session, job_locks):
    # A cancel landing between a put-back's check and its re-add finds no job
    # to remove, and the re-add then brings the cancelled job back.
    key = f"copilot:at_reset_lock:at-reset-send-{SESSION}"
    held_while_cancelling: list[bool] = []

    def cancel(**_) -> bool:
        held_while_cancelling.append(job_locks.is_locked(key))
        return True

    scheduler.cancel_at_reset_turn.side_effect = cancel

    response = client.delete(f"/sessions/{SESSION}/at-reset/send")

    assert response.status_code == 204
    assert held_while_cancelling == [True]
    assert not job_locks.is_locked(key)


def test_cancel_an_unknown_kind_is_422(scheduler, session):
    response = client.delete(f"/sessions/{SESSION}/at-reset/later")

    assert response.status_code == 422
    scheduler.cancel_at_reset_turn.assert_not_awaited()


def test_cancel_on_an_unknown_session_is_404(
    scheduler, mocker: pytest_mock.MockerFixture
):
    mocker.patch.object(
        at_reset_routes,
        "_validate_and_get_session",
        AsyncMock(side_effect=NotFoundError(f"Session {SESSION} not found.")),
    )

    response = client.delete(f"/sessions/{SESSION}/at-reset/resume")

    assert response.status_code == 404
    scheduler.cancel_at_reset_turn.assert_not_awaited()
