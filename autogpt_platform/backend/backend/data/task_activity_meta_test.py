"""An activity line's account and skill, written into its data and read back."""

from datetime import UTC, datetime

import prisma.models
import pytest
from prisma.enums import TaskActivityKind, TaskActor

from backend.data.task_activity_meta import TaskActivityAccount, meta_data
from backend.data.task_record import TaskActivity

_GMAIL = TaskActivityAccount(service="google", name="Gmail", label="work@acme.com")


def _row(data: dict) -> prisma.models.TaskActivity:
    return prisma.models.TaskActivity.model_construct(
        id="line-1",
        createdAt=datetime(2026, 11, 15, tzinfo=UTC),
        taskId="task-1",
        actorType=TaskActor.EXPERT,
        actorExpertId="expert-1",
        chatSessionId="chat-1",
        kind=TaskActivityKind.TOOL,
        text="Used Gmail to send email",
        isMilestone=False,
        data=data,
    )


def test_a_line_reads_back_the_account_and_skill_it_was_written_with():
    data = {"tool": "run_block", **meta_data(_GMAIL, "Cold email")}
    line = TaskActivity.from_db(_row(data))
    assert line.account == _GMAIL
    assert line.skill == "Cold email"


def test_an_account_without_a_label_is_stored_without_one():
    data = meta_data(TaskActivityAccount(service="github", name="GitHub"), None)
    assert data == {"account": {"service": "github", "name": "GitHub"}}


def test_nothing_is_written_when_neither_is_known():
    assert meta_data(None, None) == {}


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"tool": "web_search", "count": 3},
        {"account": "google", "skill": 7},
        {"account": {"service": "google"}, "skill": "   "},
    ],
)
def test_an_older_or_malformed_line_has_neither(data):
    line = TaskActivity.from_db(_row(data))
    assert line.account is None and line.skill is None
