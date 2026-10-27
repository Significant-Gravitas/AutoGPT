"""Endpoint tests for the Home checklist of suggested first tasks.

The row lives in memory; the compare-and-set write it stands in for is
proven against Postgres in ``suggested_tasks_db_test.py``.
"""

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from prisma.enums import BrainDumpInputMode, BrainDumpStatus
from prisma.models import OnboardingBrainDump
from pytest_mock import MockerFixture

from backend.api.features.experts import experts_db
from backend.api.features.experts.models import Expert
from backend.api.features.onboarding_dump import (
    intro,
    role_tasks,
    routes,
    service,
    suggested_tasks,
)
from backend.api.features.onboarding_dump.models import SuggestedTasksUpdate
from backend.api.rest_api import app as rest_app

app = fastapi.FastAPI(exception_handlers=rest_app.exception_handlers)
app.include_router(routes.router)
client = fastapi.testclient.TestClient(app)

URL = "/onboarding/brain-dump/suggested-tasks"
INTRO_URL = "/onboarding/brain-dump/intro"

STORED = [
    {"title": "Chase unpaid invoices", "prompt": "Chase my invoices.", "icon": "bell"},
    {"title": "Draft the newsletter", "prompt": "Draft it.", "icon": "megaphone"},
    {"title": "Watch competitors", "prompt": "Watch them.", "icon": "not-an-icon"},
]


class RowStore:
    """The one brain-dump row, with ``updatedAt`` moving on every write."""

    def __init__(self) -> None:
        self.row: OnboardingBrainDump | None = None
        self.writes = 0
        self.conflicts_left = 0

    def put(self, prompts: list[Any], **fields: Any) -> None:
        self.row = OnboardingBrainDump.model_construct(
            **{
                "userId": "test-user-id",
                "recordingId": "rec-1",
                "status": BrainDumpStatus.completed,
                "inputMode": BrainDumpInputMode.voice,
                "transcript": "I run a bakery.",
                "greeting": "Let's get your invoices handled.",
                "greetingSeen": True,
                "suggestedPrompts": prompts,
                "updatedAt": datetime(2026, 10, 1, tzinfo=UTC),
                **fields,
            }
        )

    async def get_dump(self, user_id: str) -> OnboardingBrainDump | None:
        return self.row

    async def ensure_dump(self, user_id: str) -> OnboardingBrainDump:
        if self.row is None:
            self.put([], recordingId="greeting-only", greetingSeen=False)
        assert self.row is not None
        return self.row

    async def replace_suggested_prompts(
        self, user_id: str, read_at: datetime, prompts: list[dict[str, Any]]
    ) -> bool:
        assert self.row is not None
        if self.conflicts_left:
            # Another tab wrote between this request's read and its write.
            self.conflicts_left -= 1
            self.row.updatedAt += timedelta(milliseconds=1)
        if self.row.updatedAt != read_at:
            return False
        self.row.suggestedPrompts = prompts
        self.row.updatedAt += timedelta(milliseconds=1)
        self.writes += 1
        return True


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def rows(mocker: MockerFixture) -> RowStore:
    store = RowStore()
    module = "backend.api.features.onboarding_dump.db"
    mocker.patch(f"{module}.get_dump", new=store.get_dump)
    mocker.patch(f"{module}.ensure_dump", new=store.ensure_dump)
    mocker.patch(
        f"{module}.replace_suggested_prompts", new=store.replace_suggested_prompts
    )
    return store


@pytest.fixture(autouse=True)
def wizard(mocker: MockerFixture) -> dict[str, AsyncMock]:
    """No role picked and nobody hired, unless a test says otherwise.

    The reads are stubbed under the wrappers, so a read and its degrade
    both run: a read shows a list through ``service.wizard_answers`` and
    ``service.safe_templates``, a change stores one straight from
    ``service.read_wizard_answers`` and ``suggested_tasks.list_templates``.
    """
    mocks = {
        "read_wizard_answers": AsyncMock(return_value=(None, [])),
        "list_templates": AsyncMock(return_value=[]),
        "list_experts": AsyncMock(return_value=[]),
    }
    mocker.patch.object(
        service, "read_wizard_answers", new=mocks["read_wizard_answers"]
    )
    mocker.patch.object(service, "list_templates", new=mocks["list_templates"])
    mocker.patch.object(suggested_tasks, "list_templates", new=mocks["list_templates"])
    mocker.patch.object(experts_db, "list_experts", new=mocks["list_experts"])
    return mocks


def states() -> dict[str, str]:
    return {task["id"]: task["state"] for task in client.get(URL).json()["tasks"]}


def update(task_ids: list[str], state: str, **extra: Any):
    return client.post(URL, json={"task_ids": task_ids, "state": state, **extra})


def test_a_list_stored_before_states_existed_reads_open_by_index(rows: RowStore):
    rows.put(STORED)

    body = client.get(URL).json()

    assert body["stored"] is True
    assert [(t["id"], t["state"], t["title"]) for t in body["tasks"]] == [
        ("0", "open", "Chase unpaid invoices"),
        ("1", "open", "Draft the newsletter"),
        ("2", "open", "Watch competitors"),
    ]
    assert body["tasks"][2]["icon"] == intro.DEFAULT_PROMPT_ICON


def test_the_list_outlives_the_greeting(rows: RowStore):
    rows.put(STORED)

    assert client.get(INTRO_URL).json()["greeting_done"] is True
    assert len(client.get(URL).json()["tasks"]) == 3


def test_sending_one_leaves_the_others_open(rows: RowStore):
    rows.put(STORED)

    response = update(["1"], "sent", session_id="session-1")

    assert response.status_code == 200
    assert states() == {"0": "open", "1": "sent", "2": "open"}
    sent = client.get(URL).json()["tasks"][1]
    assert sent["session_id"] == "session-1"
    assert sent["updated_at"] is not None


def test_dismiss_then_undo_reopens_the_row(rows: RowStore):
    rows.put(STORED)

    update(["0"], "dismissed")
    assert states()["0"] == "dismissed"

    update(["0"], "open")
    assert states() == {"0": "open", "1": "open", "2": "open"}


def test_hide_all_and_its_undo_are_one_write_each(rows: RowStore):
    rows.put(STORED)

    update(["0", "1", "2"], "dismissed")
    assert set(states().values()) == {"dismissed"}
    update(["0", "1", "2"], "open")

    assert set(states().values()) == {"open"}
    assert rows.writes == 2


def test_a_sent_row_never_reopens(rows: RowStore):
    rows.put(STORED)
    update(["1"], "sent", session_id="session-1")

    response = update(["1"], "open")

    assert response.status_code == 200
    assert states()["1"] == "sent"
    assert rows.writes == 1


def test_undoing_the_hide_of_a_started_row_brings_it_back_in_progress(
    rows: RowStore,
):
    rows.put(STORED)
    update(["1"], "sent", session_id="session-1")
    update(["1"], "dismissed")
    assert states()["1"] == "dismissed"

    update(["0", "1", "2"], "open")

    tasks = client.get(URL).json()["tasks"]
    assert [t["state"] for t in tasks] == ["open", "sent", "open"]
    assert tasks[1]["session_id"] == "session-1"


def test_an_unknown_id_is_404_and_writes_nothing(rows: RowStore):
    rows.put(STORED)

    response = update(["0", "9"], "dismissed")

    assert response.status_code == 404
    assert rows.writes == 0
    assert states()["0"] == "open"


def test_a_dropped_malformed_row_does_not_renumber_its_neighbours(rows: RowStore):
    rows.put([STORED[0], {"title": 3}, STORED[1]])

    update(["2"], "dismissed")

    assert states() == {"0": "open", "2": "dismissed"}


def test_without_a_list_of_their_own_the_user_gets_the_unstored_generic_set(
    rows: RowStore,
):
    rows.put([])

    body = client.get(URL).json()

    assert body["stored"] is False
    assert [t["title"] for t in body["tasks"]] == [
        p.title for p in intro.fallback_prompts()
    ]


def test_acting_on_a_generic_row_stores_the_set(rows: RowStore):
    rows.put([])

    update(["0"], "sent", session_id="session-1")

    body = client.get(URL).json()
    assert body["stored"] is True
    assert [t["state"] for t in body["tasks"]] == ["sent"] + ["open"] * 4


def test_a_user_without_a_row_gets_one_on_their_first_change(rows: RowStore):
    update(["3"], "dismissed")

    assert rows.row is not None
    assert states()["3"] == "dismissed"


def test_a_write_that_loses_a_race_rereads_instead_of_overwriting(rows: RowStore):
    rows.put(STORED)
    rows.conflicts_left = 2

    response = update(["0"], "dismissed")

    assert response.status_code == 200
    assert states()["0"] == "dismissed"


def test_a_write_that_keeps_losing_is_409(rows: RowStore):
    rows.put(STORED)
    rows.conflicts_left = 3

    assert update(["0"], "dismissed").status_code == 409


def test_an_empty_id_list_is_refused(rows: RowStore):
    rows.put(STORED)

    assert update([], "dismissed").status_code == 422


def test_a_user_who_picked_a_role_starts_from_that_roles_tasks(
    rows: RowStore, wizard: dict[str, AsyncMock]
):
    rows.put([])
    wizard["read_wizard_answers"].return_value = ("Marketing", [])

    body = client.get(URL).json()

    assert body["stored"] is False
    assert [t["title"] for t in body["tasks"]] == [
        t.title for t in role_tasks.ROLE_FALLBACK_TASKS["Marketing"]
    ]


OWNED = [
    {**STORED[0], "expert_template_id": "tpl-max", "why": "Cash comes first."},
    {**STORED[1], "expert_template_id": "tpl-maria"},
    STORED[2],
]


def _hired(template_id: str, expert_id: str) -> Expert:
    return Expert.model_construct(
        id=expert_id, name=expert_id, source_template_id=template_id
    )


def test_each_owned_task_names_its_hired_expert(
    rows: RowStore, wizard: dict[str, AsyncMock]
):
    rows.put(OWNED)
    wizard["list_experts"].return_value = [_hired("tpl-max", "expert-max")]

    tasks = client.get(URL).json()["tasks"]

    assert [(t["expert_template_id"], t["expert_id"]) for t in tasks] == [
        ("tpl-max", "expert-max"),
        ("tpl-maria", None),
        (None, None),
    ]
    assert tasks[0]["why"] == "Cash comes first."


def test_a_change_answers_with_owners_but_never_stores_the_expert_id(
    rows: RowStore, wizard: dict[str, AsyncMock]
):
    rows.put(OWNED)
    wizard["list_experts"].return_value = [_hired("tpl-max", "expert-max")]

    body = update(["0"], "sent", session_id="session-1").json()

    assert body["tasks"][0]["expert_id"] == "expert-max"
    assert rows.row is not None
    assert all("expert_id" not in row for row in rows.row.suggestedPrompts)
    assert rows.row.suggestedPrompts[0]["expert_template_id"] == "tpl-max"


def test_an_unreadable_roster_leaves_every_task_with_otto(
    rows: RowStore, wizard: dict[str, AsyncMock]
):
    rows.put(OWNED)
    wizard["list_experts"].side_effect = RuntimeError("database down")

    response = client.get(URL)

    assert response.status_code == 200
    assert [t["expert_id"] for t in response.json()["tasks"]] == [None] * 3


MARKETING_TITLES = [t.title for t in role_tasks.ROLE_FALLBACK_TASKS["Marketing"]]


def test_a_failed_roster_read_still_shows_the_role_set_and_stores_nothing(
    rows: RowStore, wizard: dict[str, AsyncMock]
):
    # The titles and their order do not depend on the roster, only owners do.
    rows.put([])
    wizard["read_wizard_answers"].return_value = ("Marketing", [])
    wizard["list_templates"].side_effect = RuntimeError("database down")

    response = client.get(URL)

    assert response.status_code == 200
    body = response.json()
    assert body["stored"] is False
    assert [t["title"] for t in body["tasks"]] == MARKETING_TITLES
    assert [t["expert_template_id"] for t in body["tasks"]] == [None] * len(
        MARKETING_TITLES
    )
    assert rows.writes == 0


async def test_a_failed_role_read_hands_out_no_ids_rather_than_another_list(
    rows: RowStore, wizard: dict[str, AsyncMock]
):
    # Without the role the starters would be the generic set, while the next
    # change stores the role's set under the same ids: an x on one would
    # remove another. The read fails instead, and the client asks again.
    rows.put([])
    wizard["read_wizard_answers"].side_effect = RuntimeError("database down")

    with pytest.raises(RuntimeError):
        await suggested_tasks.get_suggested_tasks("test-user-id")

    assert rows.writes == 0


@pytest.mark.parametrize("failing", ["read_wizard_answers", "list_templates"])
async def test_a_change_to_the_starters_stores_nothing_when_a_read_fails(
    rows: RowStore, wizard: dict[str, AsyncMock], failing: str
):
    """A list stored from a failed read would stay degraded for good."""
    rows.put([])
    wizard["read_wizard_answers"].return_value = ("Marketing", [])
    wizard[failing].side_effect = RuntimeError("database down")

    with pytest.raises(RuntimeError):
        await suggested_tasks.update_suggested_tasks(
            "test-user-id", SuggestedTasksUpdate(task_ids=["0"], state="dismissed")
        )

    assert rows.writes == 0
    assert rows.row is not None
    assert rows.row.suggestedPrompts == []


async def test_the_roster_is_keyed_by_the_template_each_expert_was_hired_from(
    wizard: dict[str, AsyncMock],
):
    max_hire = _hired("tpl-max", "expert-max")
    own = Expert.model_construct(id="expert-own", name="Own", source_template_id=None)
    wizard["list_experts"].return_value = [max_hire, own]

    assert await suggested_tasks.roster_by_template("test-user-id") == {
        "tpl-max": max_hire
    }


async def test_an_unreadable_roster_reads_as_nobody_hired(
    wizard: dict[str, AsyncMock],
):
    wizard["list_experts"].side_effect = RuntimeError("database down")

    assert await suggested_tasks.roster_by_template("test-user-id") == {}


def _template(template_id: str, role: str) -> Expert:
    return Expert.model_construct(
        id=template_id,
        name=template_id,
        role=role,
        avatar_url=None,
        tagline=f"{template_id} handles {role.lower()}.",
        is_template=True,
        source_template_id=None,
        is_archived=False,
        workflows=[],
    )
