"""Endpoint tests for the Home checklist of suggested first tasks.

The row lives in memory; the compare-and-set write it stands in for is
proven against Postgres in ``suggested_tasks_db_test.py``.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

import fastapi
import fastapi.testclient
import pytest
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from prisma.enums import BrainDumpInputMode, BrainDumpStatus
from prisma.models import OnboardingBrainDump
from pytest_mock import MockerFixture

from backend.api.features.onboarding_dump import intro, routes
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
