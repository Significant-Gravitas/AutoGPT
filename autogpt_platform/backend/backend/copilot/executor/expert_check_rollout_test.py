"""A new executor's expert check against a database manager mid-rollout.

The executor checks an expert's turn with ``check_private_expert``, a route a
database manager from the release before doesn't have. The RPC client waits
up to five minutes (``rpc_client_missing_route_wait_seconds``) for such a
manager to be replaced, so the check makes one attempt, and an older manager
is asked the same through the separate reads its release makes.

Both releases' expert routes are served here, in this process, by the real
app builder (``AppService.build_fastapi_app``), and the executor reaches them
through the real ``DatabaseManagerAsyncClient``. The retries run on a fake
clock that the tests read, so a wait shows up without being waited out.
"""

import asyncio
import logging
import secrets
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import ExitStack, asynccontextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from backend.api.features.experts.models import (
    PROTECTED_SOUL_RULES,
    CheckedExpert,
    Expert,
    ExpertSetupStatus,
)
from backend.copilot.executor.processor import (
    CoPilotProcessor,
    _normalize_private_expert_session_tenancy,
    _reauthorize_private_expert_after_setup_wait,
    _wait_for_expert_setup,
)
from backend.copilot.executor.utils import CoPilotExecutionEntry, CoPilotLogMetadata
from backend.copilot.expert_check import checked_expert, sharing_expert_checks
from backend.copilot.expert_context import (
    EXPERT_SESSION_MISSING_MESSAGE,
    EXPERT_SESSION_TEMPORARY_MESSAGE,
    ExpertSessionUnavailableError,
    build_expert_identity_suffix,
)
from backend.copilot.model import ChatSession
from backend.copilot.response_model import StreamTextDelta
from backend.data import db_accessors
from backend.data.db_manager import DatabaseManager, DatabaseManagerAsyncClient
from backend.util import clients
from backend.util import retry as retry_module
from backend.util import service as service_module
from backend.util.exceptions import (
    DatabaseError,
    ExpertNotFoundError,
    ExpertPrivateTenancyNotFoundError,
)
from backend.util.service import AppService, expose, get_service_client
from backend.util.service_auth import internal_auth_headers

_PROCESSOR = "backend.copilot.executor.processor"
CHECK = "/check_private_expert"
GET_EXPERT = "/get_expert"
TENANCY = "/resolve_private_expert_tenancy"
SETUP_STATUS = "/expert_setup_status"
# What a manager from the release before answers for an expert's check.
READ_SEPARATELY = [(CHECK, 404), (GET_EXPERT, 200), (TENANCY, 200)]


def _expert(name: str = "Maria", setup_status: ExpertSetupStatus = "ready") -> Expert:
    return Expert(
        id="expert-1",
        name=name,
        avatar_url=None,
        role="SEO Specialist",
        tagline=None,
        bio=None,
        skills=[],
        identity=f"You are {name}.",
        voice_preferences="Direct.",
        boundaries="Ask first.",
        protected_soul_rules=list(PROTECTED_SOUL_RULES),
        is_template=False,
        source_template_id=None,
        is_archived=False,
        workflows=[],
        setup_status=setup_status,
    )


def _session() -> ChatSession:
    session = ChatSession.new(
        "user-1",
        dry_run=False,
        organization_id="org-1",
        team_id="team-1",
        expert_id="expert-1",
    )
    session.session_id = "sess-1"
    return session


class _Roster:
    """The rows the expert routes read: one owner's experts, the owner's
    personal organization and team, and what each setup poll finds. A read
    of an expert applies the rules of ``experts_db``: the owner's own,
    PRIVATE, not a template, not archived."""

    def __init__(self) -> None:
        self.experts = {"expert-1": _expert()}
        self.owners = {"expert-1": "user-1"}
        self.private = {"expert-1"}
        self.personal_scope: dict[str, tuple[str, str | None]] = {
            "user-1": ("org-1", "team-1")
        }
        self.setup_statuses: list[ExpertSetupStatus] = []
        self.while_setting_up: Callable[[], None] | None = None
        self.check_fails_once: Exception | None = None
        self.check_hangs = False
        self.check_entered = asyncio.Event()
        self.calls: list[str] = []

    def active(self, user_id: str, expert_id: str) -> Expert:
        expert = self.experts.get(expert_id)
        if (
            expert is None
            or self.owners.get(expert_id) != user_id
            or expert_id not in self.private
            or expert.is_template
            or expert.is_archived
        ):
            raise ExpertNotFoundError(expert_id)
        return expert

    def scope(self, user_id: str, expert_id: str) -> tuple[str, str | None]:
        scope = self.personal_scope.get(user_id)
        if scope is None:
            raise ExpertPrivateTenancyNotFoundError(expert_id)
        return scope

    def change(self, **fields: Any) -> None:
        self.experts["expert-1"] = self.experts["expert-1"].model_copy(update=fields)


class _PreviousReleaseManager(AppService):
    """A database manager's expert routes before ``check_private_expert``."""

    def __init__(self, roster: _Roster) -> None:
        super().__init__()
        self.roster = roster

    @classmethod
    def get_port(cls) -> int:
        return DatabaseManager.get_port()

    @expose
    async def get_expert(
        self,
        user_id: str,
        expert_id: str,
        include_workflows: bool = True,
        include_archived: bool = False,
        include_credentials: bool = False,
    ) -> Expert | None:
        self.roster.calls.append("get_expert")
        try:
            return self.roster.active(user_id, expert_id)
        except ExpertNotFoundError:
            return None

    @expose
    async def resolve_private_expert_tenancy(
        self, user_id: str, expert_id: str
    ) -> tuple[str, str | None]:
        self.roster.calls.append("resolve_private_expert_tenancy")
        self.roster.active(user_id, expert_id)
        return self.roster.scope(user_id, expert_id)

    @expose
    async def expert_setup_status(
        self, user_id: str, expert_id: str
    ) -> ExpertSetupStatus:
        self.roster.calls.append("expert_setup_status")
        if self.roster.while_setting_up is not None:
            self.roster.while_setting_up()
            self.roster.while_setting_up = None
        return (
            self.roster.setup_statuses.pop(0) if self.roster.setup_statuses else "ready"
        )


class _NewReleaseManager(_PreviousReleaseManager):
    """The same routes, and ``check_private_expert``."""

    # Only a class's own exposed methods are served.
    get_expert = _PreviousReleaseManager.get_expert
    resolve_private_expert_tenancy = (
        _PreviousReleaseManager.resolve_private_expert_tenancy
    )
    expert_setup_status = _PreviousReleaseManager.expert_setup_status

    @expose
    async def check_private_expert(self, user_id: str, expert_id: str) -> CheckedExpert:
        roster = self.roster
        roster.calls.append("check_private_expert")
        roster.check_entered.set()
        if roster.check_hangs:
            await asyncio.Event().wait()
        if roster.check_fails_once is not None:
            error, roster.check_fails_once = roster.check_fails_once, None
            raise error
        expert = roster.active(user_id, expert_id)
        organization_id, team_id = roster.scope(user_id, expert_id)
        return CheckedExpert(
            expert=expert, organization_id=organization_id, team_id=team_id
        )


class _FakeClock:
    """The retries' clock and sleep, where a sleep only moves the clock on."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    async def async_sleep(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    clock = _FakeClock()
    monkeypatch.setattr(service_module, "monotonic", clock.monotonic)
    monkeypatch.setattr(retry_module, "_interruptible_sleep", clock.sleep)
    monkeypatch.setattr(retry_module, "_interruptible_async_sleep", clock.async_sleep)
    return clock


@asynccontextmanager
async def _serving(
    release: type[_PreviousReleaseManager],
    roster: _Roster,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[list[tuple[str, int]]]:
    """Answer the executor's database-manager calls, from both the retrying
    client and the one-attempt one, with *release*'s routes over *roster*.
    Yields each answered request's route and status, in order."""
    monkeypatch.setenv("INTERNAL_API_KEY", secrets.token_urlsafe(32))
    monkeypatch.setattr(service_module, "instrument_fastapi", lambda *_a, **_k: None)
    monkeypatch.setattr(db_accessors, "db", SimpleNamespace(is_connected=lambda: False))
    # A server answers an error it didn't map with a 500, as uvicorn does,
    # instead of raising it into the client.
    transport = httpx.ASGITransport(
        app=release(roster).build_fastapi_app(), raise_app_exceptions=False
    )
    answered: list[tuple[str, int]] = []

    async def record(response: httpx.Response) -> None:
        answered.append((response.request.url.path, response.status_code))

    loop = asyncio.get_running_loop()
    by_retry: dict[bool, DatabaseManagerAsyncClient] = {}
    for should_retry in (True, False):
        client = get_service_client(
            DatabaseManagerAsyncClient, request_retry=should_retry
        )
        inner = cast(Any, client)
        inner._async_clients[loop] = httpx.AsyncClient(
            transport=transport,
            base_url=inner.base_url,
            headers=internal_auth_headers(),
            event_hooks={"response": [record]},
        )
        by_retry[should_retry] = client
    monkeypatch.setattr(
        clients,
        "get_database_manager_async_client",
        lambda should_retry=True: by_retry[should_retry],
    )
    try:
        yield answered
    finally:
        for client in by_retry.values():
            await cast(Any, client).aclose()


async def _check(session: ChatSession) -> CheckedExpert | None:
    """Check *session*'s expert as a turn does; what the turn remembers."""
    await _normalize_private_expert_session_tenancy(session)
    return checked_expert("user-1", "expert-1")


@pytest.mark.asyncio
async def test_a_previous_release_manager_is_asked_once_and_read_separately(
    monkeypatch: pytest.MonkeyPatch, fake_clock: _FakeClock
):
    roster = _Roster()
    async with _serving(_PreviousReleaseManager, roster, monkeypatch) as answered:
        with sharing_expert_checks():
            check = await _check(_session())

    assert check is not None
    assert (check.organization_id, check.team_id) == ("org-1", "team-1")
    assert check.expert.name == "Maria"
    assert check.expert.setup_status == "ready"
    assert answered == READ_SEPARATELY
    assert fake_clock.now == 0


@pytest.mark.asyncio
async def test_a_new_release_manager_answers_the_check_in_one_call(
    monkeypatch: pytest.MonkeyPatch, fake_clock: _FakeClock
):
    roster = _Roster()
    async with _serving(_NewReleaseManager, roster, monkeypatch) as answered:
        with sharing_expert_checks():
            check = await _check(_session())

    assert check is not None
    assert (check.organization_id, check.team_id) == ("org-1", "team-1")
    assert check.expert.name == "Maria"
    assert answered == [(CHECK, 200)]
    assert roster.calls == ["check_private_expert"]
    assert fake_clock.now == 0


def _not_owned(roster: _Roster) -> None:
    roster.owners["expert-1"] = "user-2"


def _not_private(roster: _Roster) -> None:
    roster.private.clear()


def _a_template(roster: _Roster) -> None:
    roster.change(is_template=True)


def _archived(roster: _Roster) -> None:
    roster.change(is_archived=True)


def _no_personal_organization(roster: _Roster) -> None:
    roster.personal_scope.clear()


# Each refusal: what the turn is told, the status of the new route's answer,
# and the answers a manager from the release before gives.
REFUSALS = [
    pytest.param(
        _not_owned,
        EXPERT_SESSION_MISSING_MESSAGE,
        404,
        READ_SEPARATELY[:2],
        id="not-owned",
    ),
    pytest.param(
        _not_private,
        EXPERT_SESSION_MISSING_MESSAGE,
        404,
        READ_SEPARATELY[:2],
        id="not-private",
    ),
    pytest.param(
        _a_template,
        EXPERT_SESSION_MISSING_MESSAGE,
        404,
        READ_SEPARATELY[:2],
        id="template",
    ),
    pytest.param(
        _archived,
        EXPERT_SESSION_MISSING_MESSAGE,
        404,
        READ_SEPARATELY[:2],
        id="archived",
    ),
    pytest.param(
        _no_personal_organization,
        EXPERT_SESSION_TEMPORARY_MESSAGE,
        503,
        [*READ_SEPARATELY[:2], (TENANCY, 503)],
        id="no-personal-organization",
    ),
]
REFUSAL_FIELDS = ("refuse", "message", "status", "previous_release_answers")


@pytest.mark.parametrize(REFUSAL_FIELDS, REFUSALS)
@pytest.mark.asyncio
async def test_a_previous_release_manager_still_refuses_what_it_should(
    monkeypatch: pytest.MonkeyPatch,
    fake_clock: _FakeClock,
    refuse: Callable[[_Roster], None],
    message: str,
    status: int,
    previous_release_answers: list[tuple[str, int]],
):
    roster = _Roster()
    refuse(roster)
    async with _serving(_PreviousReleaseManager, roster, monkeypatch) as answered:
        with sharing_expert_checks():
            with pytest.raises(ExpertSessionUnavailableError) as refused:
                await _check(_session())
            assert checked_expert("user-1", "expert-1") is None

    assert str(refused.value) == message
    assert answered == previous_release_answers
    assert fake_clock.now == 0


@pytest.mark.parametrize(REFUSAL_FIELDS, REFUSALS)
@pytest.mark.asyncio
async def test_a_new_release_managers_refusal_is_final(
    monkeypatch: pytest.MonkeyPatch,
    fake_clock: _FakeClock,
    refuse: Callable[[_Roster], None],
    message: str,
    status: int,
    previous_release_answers: list[tuple[str, int]],
):
    """A refusal is an answer: the check is not read again another way."""
    roster = _Roster()
    refuse(roster)
    async with _serving(_NewReleaseManager, roster, monkeypatch) as answered:
        with sharing_expert_checks():
            with pytest.raises(ExpertSessionUnavailableError) as refused:
                await _check(_session())
            assert checked_expert("user-1", "expert-1") is None

    assert str(refused.value) == message
    assert answered == [(CHECK, status)]
    assert roster.calls == ["check_private_expert"]
    assert fake_clock.now == 0


@pytest.mark.asyncio
async def test_a_check_that_fails_in_transit_is_read_separately(
    monkeypatch: pytest.MonkeyPatch, fake_clock: _FakeClock
):
    roster = _Roster()
    roster.check_fails_once = DatabaseError("connection reset")
    async with _serving(_NewReleaseManager, roster, monkeypatch) as answered:
        with sharing_expert_checks():
            check = await _check(_session())

    assert check is not None and check.expert.name == "Maria"
    assert answered == [(CHECK, 500), (GET_EXPERT, 200), (TENANCY, 200)]
    assert fake_clock.now == 0


@pytest.mark.asyncio
async def test_a_turn_cancelled_during_its_check_reads_nothing_more(
    monkeypatch: pytest.MonkeyPatch, fake_clock: _FakeClock
):
    roster = _Roster()
    roster.check_hangs = True
    async with _serving(_NewReleaseManager, roster, monkeypatch) as answered:
        with sharing_expert_checks():
            check = asyncio.create_task(_check(_session()))
            await asyncio.wait_for(roster.check_entered.wait(), timeout=30)
            check.cancel()
            with pytest.raises(asyncio.CancelledError):
                await check
            assert checked_expert("user-1", "expert-1") is None

    assert roster.calls == ["check_private_expert"]
    assert answered == []
    assert fake_clock.now == 0


def _renamed(roster: _Roster) -> None:
    roster.change(name="Maria after setup", setup_status="ready")


def _renamed_and_archived(roster: _Roster) -> None:
    roster.change(name="Maria after setup", setup_status="ready", is_archived=True)


def _renamed_and_moved(roster: _Roster) -> None:
    _renamed(roster)
    roster.personal_scope["user-1"] = ("org-2", "team-2")


@pytest.mark.parametrize(
    ("while_setting_up", "message", "after_the_wait"),
    [
        pytest.param(_renamed, None, READ_SEPARATELY, id="unchanged"),
        pytest.param(
            _renamed_and_archived,
            EXPERT_SESSION_MISSING_MESSAGE,
            READ_SEPARATELY[:2],
            id="archived",
        ),
        pytest.param(
            _renamed_and_moved,
            EXPERT_SESSION_TEMPORARY_MESSAGE,
            READ_SEPARATELY,
            id="moved",
        ),
    ],
)
@pytest.mark.asyncio
async def test_a_held_turn_reads_its_expert_again_from_a_previous_release_manager(
    monkeypatch: pytest.MonkeyPatch,
    fake_clock: _FakeClock,
    while_setting_up: Callable[[_Roster], None],
    message: str | None,
    after_the_wait: list[tuple[str, int]],
):
    roster = _Roster()
    roster.change(setup_status="installing")
    roster.setup_statuses = ["installing", "ready"]
    roster.while_setting_up = lambda: while_setting_up(roster)
    monkeypatch.setattr(f"{_PROCESSOR}.EXPERT_SETUP_POLL_SECONDS", 0)
    async with _serving(_PreviousReleaseManager, roster, monkeypatch) as answered:
        with sharing_expert_checks():
            session = _session()
            before = await _check(session)
            assert before is not None and before.expert.setup_status == "installing"
            assert await _wait_for_expert_setup(session) is True
            if message is None:
                await _reauthorize_private_expert_after_setup_wait(session)
                after = checked_expert("user-1", "expert-1")
                assert after is not None and after.expert.name == "Maria after setup"
            else:
                with pytest.raises(ExpertSessionUnavailableError) as refused:
                    await _reauthorize_private_expert_after_setup_wait(session)
                assert str(refused.value) == message
                assert checked_expert("user-1", "expert-1") is None

    assert answered == [
        *READ_SEPARATELY,
        (SETUP_STATUS, 200),
        (SETUP_STATUS, 200),
        *after_the_wait,
    ]
    assert fake_clock.now == 0


async def _run_turn(
    release: type[_PreviousReleaseManager],
    roster: _Roster,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[str], list[tuple[str, int]]]:
    """Run an expert's turn as the executor runs it, its database manager
    answered by *release*. Returns the identities the engine was given and
    the database manager's answers."""
    identities: list[str] = []

    async def engine(*, session: ChatSession, **_kwargs):
        identities.append(
            await build_expert_identity_suffix(
                session.user_id,
                session.expert_id,
                organization_id=session.organization_id,
                team_id=session.team_id,
            )
        )
        yield StreamTextDelta(id="t", delta="hi")

    usage = MagicMock()
    usage.get_usage_activation_state = AsyncMock(
        side_effect=lambda user_id: SimpleNamespace(
            user_id=user_id, generation=None, trial_id=None, ready=True
        )
    )
    entry = CoPilotExecutionEntry(
        session_id="sess-1",
        turn_id="turn-1",
        user_id="user-1",
        message="hi",
        is_user_message=True,
        request_arrival_at=0.0,
    )
    session = _session()
    async with _serving(release, roster, monkeypatch) as answered:
        with ExitStack() as stack:
            for target, patch_args in (
                (
                    f"{_PROCESSOR}.ChatConfig",
                    {"return_value": MagicMock(test_mode=True)},
                ),
                (f"{_PROCESSOR}.stream_chat_completion_dummy", {"new": engine}),
                (
                    f"{_PROCESSOR}.stream_registry.stream_and_publish",
                    {"side_effect": lambda **kwargs: kwargs["stream"]},
                ),
                (f"{_PROCESSOR}.stream_registry.publish_chunk", {"new": AsyncMock()}),
                (
                    f"{_PROCESSOR}.stream_registry.mark_session_completed",
                    {"new": AsyncMock()},
                ),
                (
                    f"{_PROCESSOR}.close_browser_daemon",
                    {"new": AsyncMock(return_value=False)},
                ),
                (
                    "backend.copilot.model.get_chat_session",
                    {"new": AsyncMock(return_value=session)},
                ),
                (f"{_PROCESSOR}.refusal_for", {"new": AsyncMock(return_value=None)}),
                (
                    "backend.copilot.usage_activation.pro_activation_db",
                    {"return_value": usage},
                ),
            ):
                stack.enter_context(patch(target, **patch_args))
            await CoPilotProcessor()._execute_async(
                entry,
                threading.Event(),
                MagicMock(),
                CoPilotLogMetadata(logger=logging.getLogger("test-copilot")),
            )
    return identities, answered


@pytest.mark.asyncio
async def test_an_expert_turn_on_a_previous_release_manager_starts_at_once(
    monkeypatch: pytest.MonkeyPatch, fake_clock: _FakeClock
):
    identities, answered = await _run_turn(
        _PreviousReleaseManager, _Roster(), monkeypatch
    )

    assert len(identities) == 1
    assert "For this session you are Maria" in identities[0]
    assert answered == READ_SEPARATELY
    assert fake_clock.now == 0


@pytest.mark.asyncio
async def test_an_archived_experts_turn_on_a_previous_release_manager_is_refused(
    monkeypatch: pytest.MonkeyPatch, fake_clock: _FakeClock
):
    roster = _Roster()
    _archived(roster)

    with pytest.raises(ExpertSessionUnavailableError) as refused:
        await _run_turn(_PreviousReleaseManager, roster, monkeypatch)

    assert str(refused.value) == EXPERT_SESSION_MISSING_MESSAGE
    assert roster.calls == ["get_expert"]
    assert fake_clock.now == 0
