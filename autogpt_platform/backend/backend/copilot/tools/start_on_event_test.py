"""start_task with on_event: the event read as a trigger block, the webhook set
up with the user's account, and one task that waits, watching; or the card to
connect the account, or a refusal Otto can act on."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import TaskCreator, TaskStatus

from backend.copilot import turn_end
from backend.copilot.tools.event_triggers import resolve_event_trigger
from backend.copilot.tools.models import ErrorResponse, SetupRequirementsResponse
from backend.copilot.tools.start_task import StartTaskTool
from backend.copilot.tools.task_models import TaskStartedResponse
from backend.data.model import CredentialsMetaInput
from backend.data.task import Task
from backend.data.task_schedule_meta import TaskRepeat, TaskWatch

from ._test_data import make_session

_USER = "test-user-on-event"
_NOW = datetime(2026, 11, 17, 7, 0, tzinfo=UTC)
_PR_TRIGGER = "6c60ec01-8128-419e-988f-96a063ee2fea"
_GENERIC_TRIGGER = "8fa8c167-2002-47ce-aba8-97572fc5d387"
_MODULE = "backend.copilot.tools.start_on_event"
_ON_PR = {
    "provider": "github",
    "event": "pull_request.opened",
    "config": {"repo": "acme/web"},
}


def _task(**overrides) -> Task:
    fields = {
        "id": "task-1",
        "user_id": _USER,
        "title": "Review new PRs",
        "status": TaskStatus.TODO,
        "created_by": TaskCreator.OTTO,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    return Task(**(fields | overrides))


def _watch(**overrides) -> TaskWatch:
    fields = {
        "webhook_id": "wh-1",
        "provider": "github",
        "trigger_block_id": _PR_TRIGGER,
        "trigger_config": {"repo": "acme/web", "events": {"opened": True}},
        "line": "PR opened on acme/web",
    }
    return TaskWatch(**(fields | overrides))


@pytest.fixture
def store():
    store = MagicMock()
    store.create_task = AsyncMock(return_value=_task())
    store.work_tasks = AsyncMock(return_value=[])
    store.find_open_task_by_title = AsyncMock(return_value=None)
    store.add_activity = AsyncMock()
    store.setup_task_watch = AsyncMock(return_value=_watch())
    store.release_watch_webhook = AsyncMock()
    store.set_task_start = AsyncMock(
        side_effect=lambda task_id, user_id, at, origin, repeat: _task(repeat=repeat)
    )
    with (
        patch("backend.copilot.tools.start_task.task_db", return_value=store),
        patch("backend.copilot.tools.start_task.task_record_db", return_value=store),
        patch(f"{_MODULE}.task_record_db", return_value=store),
        patch(f"{_MODULE}.task_schedule_db", return_value=store),
        patch(f"{_MODULE}.task_watch_db", return_value=store),
    ):
        yield store


@pytest.fixture
def session():
    session = make_session(_USER)
    yield session
    turn_end.clear(session.session_id)


def _connected(*fields: str):
    meta = CredentialsMetaInput(
        id="cred-1", provider="github", type="oauth2", title="My GitHub"
    )
    return patch(
        f"{_MODULE}.resolve_block_credentials",
        AsyncMock(return_value=({field: meta for field in fields}, [])),
    )


async def _start(session, on_event: dict):
    return await StartTaskTool()._execute(
        _USER,
        session,
        title="Review new PRs",
        brief="Review each new PR and tell me what matters.",
        on_event=on_event,
    )


async def test_a_watch_is_set_up_and_the_task_waits_without_a_start(store, session):
    with _connected("credentials"):
        result = await _start(session, _ON_PR)

    assert isinstance(result, TaskStartedResponse), result
    args = store.setup_task_watch.await_args.args
    assert args[0] == _USER and args[1] == _PR_TRIGGER
    assert args[2]["repo"] == "acme/web"
    assert args[2]["events"] == {"opened": True}
    assert args[2]["credentials"]["id"] == "cred-1"
    assert args[3] == "PR opened on acme/web"
    first_line = store.add_activity.await_args.args[2].text
    assert first_line == "Watching: PR opened on acme/web"
    task_id, user_id, at, origin = store.set_task_start.await_args.args
    assert (task_id, user_id, at, origin) == ("task-1", _USER, None, session.session_id)
    repeat = store.set_task_start.await_args.kwargs["repeat"]
    assert isinstance(repeat, TaskRepeat) and repeat.watch == _watch()
    assert result.task.repeat is not None and result.task.repeat.watch is not None
    assert "Nothing runs now" in result.message
    assert turn_end.refusal(session.session_id, "web_search") is not None


async def test_an_account_not_connected_gets_the_connect_card(store, session):
    missing = CredentialsMetaInput(
        id="credentials", provider="github", type="oauth2", title="GitHub"
    )
    with patch(
        f"{_MODULE}.resolve_block_credentials",
        AsyncMock(return_value=({}, [missing])),
    ):
        result = await _start(session, _ON_PR)

    assert isinstance(result, SetupRequirementsResponse)
    assert result.setup_info.user_readiness.missing_credentials
    store.setup_task_watch.assert_not_awaited()
    store.create_task.assert_not_awaited()


async def test_the_generic_webhook_needs_no_account_and_its_url_is_kept(store, session):
    url = (
        "https://example.invalid/api/integrations/generic_webhook/webhooks/wh-9/ingress"
    )
    store.setup_task_watch.return_value = _watch(
        provider="generic_webhook",
        trigger_block_id=_GENERIC_TRIGGER,
        trigger_config={},
        line="Calls to its webhook URL",
        url=url,
    )
    with _connected():
        result = await _start(session, {"provider": "generic_webhook"})

    assert isinstance(result, TaskStartedResponse), result
    assert store.setup_task_watch.await_args.args[1] == _GENERIC_TRIGGER
    assert result.task.repeat and result.task.repeat.watch
    assert result.task.repeat.watch.url == url
    assert "task's details" in result.message
    assert url not in result.message


async def test_a_failed_save_lets_go_of_the_webhook(store, session):
    store.set_task_start.side_effect = RuntimeError("database down")
    with _connected("credentials"):
        result = await _start(session, _ON_PR)

    assert isinstance(result, ErrorResponse)
    store.release_watch_webhook.assert_awaited_once_with(_USER, _watch())


@pytest.mark.parametrize(
    "on_event, says",
    [
        ({"provider": "slack", "event": "message"}, "repeating check"),
        (
            {"provider": "github", "event": "push", "config": {"repo": "a/b"}},
            "pull_request.<",
        ),
        ({"provider": "github", "event": "pull_request.opened"}, "repo ("),
    ],
)
async def test_an_event_it_cannot_watch_is_refused_in_words_otto_can_act_on(
    store, session, on_event, says
):
    result = await _start(session, on_event)

    assert isinstance(result, ErrorResponse)
    assert says in result.message
    store.create_task.assert_not_awaited()


async def test_a_part_of_a_task_cannot_watch(store, session):
    store.work_tasks.return_value = [_task(status=TaskStatus.IN_PROGRESS)]

    result = await _start(session, _ON_PR)

    assert isinstance(result, ErrorResponse)
    store.setup_task_watch.assert_not_awaited()


def test_the_event_is_read_as_the_trigger_block_with_its_filter():
    trigger = resolve_event_trigger(
        "GitHub", "issues.opened,reopened", {"repo": "acme/web", "secret_token": "x"}
    )

    assert trigger.trigger_config == {
        "repo": "acme/web",
        "events": {"opened": True, "reopened": True},
    }
    assert trigger.line == "Issue opened or reopened on acme/web"
    assert not trigger.manual
