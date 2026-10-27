"""Tests for Otto starting the user's first suggested tasks during onboarding.

The brain-dump row lives in memory and everything that starts a turn is
patched where ``kickoff`` imports it, so these assert what would be
started without an executor, a queue or a database.
"""

import asyncio
import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import BrainDumpInputMode, BrainDumpStatus
from prisma.models import OnboardingBrainDump
from pytest_mock import MockerFixture

from backend.api.features.onboarding_dump import kickoff
from backend.copilot.active_turns import ConcurrentTurnLimitError
from backend.copilot.model import ONBOARDING_DRAFT_KIND, ChatSessionInfo
from backend.copilot.turn_queue import TurnRefusal
from backend.util.exceptions import DatabaseError

USER_ID = "user-1"
RECORDING_ID = "rec-1"

STORED = [
    {"title": "Chase unpaid invoices", "prompt": "Chase my invoices.", "icon": "bell"},
    {"title": "Draft the newsletter", "prompt": "Draft it.", "icon": "megaphone"},
    {"title": "Watch competitors", "prompt": "Watch them.", "icon": "eye"},
]


def draft_session_id(task_id: str) -> str:
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL, f"onboarding-draft:{USER_ID}:{RECORDING_ID}:{task_id}"
        )
    )


class RowStore:
    """The one brain-dump row, with ``updatedAt`` moving on every write."""

    def __init__(self) -> None:
        self.row: OnboardingBrainDump | None = None

    def put(self, prompts: list[Any], **fields: Any) -> None:
        self.row = OnboardingBrainDump.model_construct(
            **{
                "userId": USER_ID,
                "recordingId": RECORDING_ID,
                "status": BrainDumpStatus.completed,
                "inputMode": BrainDumpInputMode.voice,
                "transcript": "I run a bakery.",
                "greeting": "Let's get your invoices handled.",
                "greetingSeen": False,
                "errorCode": None,
                "suggestedPrompts": prompts,
                "updatedAt": datetime(2026, 10, 1, tzinfo=UTC),
                **fields,
            }
        )

    async def get_dump(self, user_id: str) -> OnboardingBrainDump | None:
        return self.row

    async def ensure_dump(self, user_id: str) -> OnboardingBrainDump:
        assert self.row is not None
        return self.row

    async def replace_suggested_prompts(
        self, user_id: str, read_at: datetime, prompts: list[dict[str, Any]]
    ) -> bool:
        assert self.row is not None
        if self.row.updatedAt != read_at:
            return False
        self.row.suggestedPrompts = prompts
        self.row.updatedAt += timedelta(milliseconds=1)
        return True

    def stored(self) -> list[dict[str, Any]]:
        assert self.row is not None
        prompts = self.row.suggestedPrompts
        assert isinstance(prompts, list)
        return prompts


@pytest.fixture
def rows(mocker: MockerFixture) -> RowStore:
    store = RowStore()
    module = "backend.api.features.onboarding_dump.db"
    mocker.patch(f"{module}.get_dump", new=store.get_dump)
    mocker.patch(f"{module}.ensure_dump", new=store.ensure_dump)
    mocker.patch(
        f"{module}.replace_suggested_prompts", new=store.replace_suggested_prompts
    )
    return store


@pytest.fixture
def turns(mocker: MockerFixture) -> dict[str, MagicMock]:
    """Everything that starts a turn, patched where ``kickoff`` uses it."""
    mocks: dict[str, MagicMock] = {
        "resolve_default_chat_route": AsyncMock(return_value=("platform", None)),
        "resolve_default_tenancy": AsyncMock(return_value=("org-1", "team-1")),
        "turn_refusal": AsyncMock(return_value=None),
        "create_chat_session": AsyncMock(),
        "schedule_turn": AsyncMock(),
        "delete_chat_session": AsyncMock(return_value=True),
        "get_chat_session_metadata": AsyncMock(return_value=None),
    }
    for name, mock in mocks.items():
        mocker.patch.object(kickoff, name, new=mock)
    mocks["track"] = mocker.patch.object(
        kickoff.product_analytics, "track_onboarding_draft_started"
    )
    return mocks


@pytest.mark.asyncio
async def test_kickoff_starts_at_most_two_open_tasks(
    rows: RowStore, turns: dict[str, MagicMock]
):
    rows.put(STORED)

    response = await kickoff.kickoff_suggested_tasks(USER_ID)

    assert response.status == "started"
    assert response.started_task_ids == ["0", "1"]
    created = turns["create_chat_session"].await_args_list
    assert [call.kwargs["session_id"] for call in created] == [
        draft_session_id("0"),
        draft_session_id("1"),
    ]
    for call in created:
        assert call.args == (USER_ID,)
        assert call.kwargs["kind"] == ONBOARDING_DRAFT_KIND
        assert call.kwargs["origin"] == "interactive"
        assert call.kwargs["dry_run"] is False
        assert call.kwargs["organization_id"] == "org-1"
        assert call.kwargs["team_id"] == "team-1"
    scheduled = turns["schedule_turn"].await_args_list
    assert [call.kwargs["message"] for call in scheduled] == [
        "Chase my invoices.",
        "Draft it.",
    ]
    assert [call.kwargs["session_id"] for call in scheduled] == [
        draft_session_id("0"),
        draft_session_id("1"),
    ]
    assert all(call.kwargs["unattended"] is True for call in scheduled)
    # The prompt is Otto's, not a message the user sent.
    assert all(
        call.kwargs["message_metadata"] == {"kind": "onboarding_draft_kickoff"}
        for call in scheduled
    )
    stored = rows.stored()
    assert [
        (t.get("state"), t.get("kickoff"), t.get("session_id")) for t in stored
    ] == [
        ("sent", True, draft_session_id("0")),
        ("sent", True, draft_session_id("1")),
        ("open", False, None),
    ]


@pytest.mark.asyncio
async def test_kickoff_skips_rows_that_are_not_open(
    rows: RowStore, turns: dict[str, MagicMock]
):
    rows.put(
        [
            {**STORED[0], "state": "dismissed"},
            {**STORED[1], "state": "sent", "session_id": "typed-by-user"},
            STORED[2],
        ]
    )

    response = await kickoff.kickoff_suggested_tasks(USER_ID)

    assert response.started_task_ids == ["2"]
    assert turns["create_chat_session"].await_count == 1


@pytest.mark.asyncio
async def test_kickoff_is_idempotent(rows: RowStore, turns: dict[str, MagicMock]):
    rows.put(STORED)
    first = await kickoff.kickoff_suggested_tasks(USER_ID)

    second = await kickoff.kickoff_suggested_tasks(USER_ID)

    assert second.status == "started"
    assert second.started_task_ids == first.started_task_ids == ["0", "1"]
    assert turns["create_chat_session"].await_count == 2
    assert turns["schedule_turn"].await_count == 2


@pytest.mark.asyncio
async def test_kickoff_is_pending_until_prompts_are_stored(
    rows: RowStore, turns: dict[str, MagicMock]
):
    rows.put([], status=BrainDumpStatus.extracting)

    response = await kickoff.kickoff_suggested_tasks(USER_ID)

    assert response.status == "pending"
    assert response.started_task_ids == []
    turns["create_chat_session"].assert_not_awaited()


@pytest.mark.asyncio
async def test_kickoff_is_nothing_without_a_dump_or_when_skipped_or_failed(
    rows: RowStore, turns: dict[str, MagicMock]
):
    assert (await kickoff.kickoff_suggested_tasks(USER_ID)).status == "nothing"

    rows.put([], inputMode=BrainDumpInputMode.skipped)
    assert (await kickoff.kickoff_suggested_tasks(USER_ID)).status == "nothing"

    rows.put(STORED, status=BrainDumpStatus.failed, errorCode="no_usable_speech")
    assert (await kickoff.kickoff_suggested_tasks(USER_ID)).status == "nothing"

    # A finished take whose greeting stored no prompts will never get any.
    rows.put([])
    assert (await kickoff.kickoff_suggested_tasks(USER_ID)).status == "nothing"

    turns["create_chat_session"].assert_not_awaited()


@pytest.mark.asyncio
async def test_kickoff_is_refused_when_turn_may_not_start(
    rows: RowStore, turns: dict[str, MagicMock]
):
    rows.put(STORED)
    turns["turn_refusal"].return_value = TurnRefusal(reason="paywalled", kind="no_plan")

    response = await kickoff.kickoff_suggested_tasks(USER_ID)

    assert response.status == "refused"
    assert response.started_task_ids == []
    turns["create_chat_session"].assert_not_awaited()
    turns["schedule_turn"].assert_not_awaited()
    assert all(not t.get("kickoff") for t in rows.stored())


@pytest.mark.asyncio
async def test_kickoff_stops_at_the_concurrent_turn_limit(
    rows: RowStore, turns: dict[str, MagicMock]
):
    rows.put(STORED)
    turns["schedule_turn"].side_effect = [None, ConcurrentTurnLimitError()]

    response = await kickoff.kickoff_suggested_tasks(USER_ID)

    assert response.status == "started"
    assert response.started_task_ids == ["0"]
    assert [t.get("kickoff", False) for t in rows.stored()] == [True, False, False]
    # The chat that never got its turn would sit empty in the sidebar.
    turns["delete_chat_session"].assert_awaited_once_with(
        draft_session_id("1"), USER_ID
    )


@pytest.mark.asyncio
async def test_kickoff_never_raises(
    rows: RowStore,
    turns: dict[str, MagicMock],
    caplog: pytest.LogCaptureFixture,
):
    rows.put(STORED)
    turns["create_chat_session"].side_effect = RuntimeError("database down")

    response = await kickoff.kickoff_suggested_tasks(USER_ID)

    assert response.status == "nothing"
    assert response.started_task_ids == []
    turns["schedule_turn"].assert_not_awaited()
    assert "database down" in caplog.text


@pytest.mark.asyncio
async def test_kickoff_emits_onboarding_draft_started(
    rows: RowStore, turns: dict[str, MagicMock]
):
    rows.put(STORED)

    await kickoff.kickoff_suggested_tasks(USER_ID)

    assert [call.kwargs for call in turns["track"].call_args_list] == [
        {"user_id": USER_ID, "task_id": "0", "session_id": draft_session_id("0")},
        {"user_id": USER_ID, "task_id": "1", "session_id": draft_session_id("1")},
    ]


def existing_chats(turns: dict[str, MagicMock], session_ids: set[str]) -> None:
    """Make ``session_ids`` the user's chats that already exist."""

    async def session_meta(session_id: str, user_id: str | None = None):
        if session_id not in session_ids:
            return None
        return ChatSessionInfo.model_construct(session_id=session_id)

    turns["get_chat_session_metadata"].side_effect = session_meta


def kickoff_errors(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == kickoff.__name__ and record.levelno >= logging.ERROR
    ]


@pytest.mark.asyncio
async def test_kickoff_marks_a_row_whose_chat_already_exists(
    rows: RowStore,
    turns: dict[str, MagicMock],
    caplog: pytest.LogCaptureFixture,
):
    # An earlier call started task 0's chat and turn, then died before it
    # could mark the row.
    rows.put(STORED)
    existing_chats(turns, {draft_session_id("0")})

    response = await kickoff.kickoff_suggested_tasks(USER_ID)

    assert response.status == "started"
    assert response.started_task_ids == ["0", "1"]
    assert [
        call.kwargs["session_id"]
        for call in turns["create_chat_session"].await_args_list
    ] == [draft_session_id("1")]
    assert [
        call.kwargs["session_id"] for call in turns["schedule_turn"].await_args_list
    ] == [draft_session_id("1")]
    stored = rows.stored()
    assert (stored[0]["kickoff"], stored[0]["session_id"]) == (
        True,
        draft_session_id("0"),
    )
    assert kickoff_errors(caplog) == []


@pytest.mark.asyncio
async def test_kickoff_that_loses_the_create_race_reports_started(
    rows: RowStore,
    turns: dict[str, MagicMock],
    caplog: pytest.LogCaptureFixture,
):
    # The chats did not exist when checked, then another call created them.
    rows.put(STORED)
    created: set[str] = set()

    async def session_meta(session_id: str, user_id: str | None = None):
        if session_id not in created:
            created.add(session_id)
            return None
        return ChatSessionInfo.model_construct(session_id=session_id)

    turns["get_chat_session_metadata"].side_effect = session_meta
    turns["create_chat_session"].side_effect = DatabaseError("duplicate id")

    response = await kickoff.kickoff_suggested_tasks(USER_ID)

    assert response.status == "started"
    assert response.started_task_ids == ["0", "1"]
    turns["schedule_turn"].assert_not_awaited()
    assert all(task["kickoff"] for task in rows.stored()[:2])
    assert kickoff_errors(caplog) == []


@pytest.mark.asyncio
async def test_racing_kickoffs_both_report_started_and_start_each_task_once(
    rows: RowStore,
    turns: dict[str, MagicMock],
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
):
    """The background trigger and the copilot home's call read the row
    before either marks it; the second still answers ``started``."""
    rows.put(STORED)
    created: set[str] = set()
    existing_chats(turns, created)

    async def create(user_id: str, **kwargs: Any) -> None:
        if kwargs["session_id"] in created:
            raise DatabaseError("duplicate id")
        created.add(kwargs["session_id"])

    turns["create_chat_session"].side_effect = create
    both_read = asyncio.Barrier(2)
    first_reads = 2
    read_row = rows.get_dump

    async def get_dump(user_id: str):
        nonlocal first_reads
        row = await read_row(user_id)
        if first_reads:
            first_reads -= 1
            await asyncio.wait_for(both_read.wait(), timeout=5)
        return row

    mocker.patch("backend.api.features.onboarding_dump.db.get_dump", new=get_dump)

    first, second = await asyncio.gather(
        kickoff.kickoff_suggested_tasks(USER_ID),
        kickoff.kickoff_suggested_tasks(USER_ID),
    )

    assert (first.status, second.status) == ("started", "started")
    assert first.started_task_ids == second.started_task_ids == ["0", "1"]
    assert turns["schedule_turn"].await_count == 2
    assert kickoff_errors(caplog) == []
