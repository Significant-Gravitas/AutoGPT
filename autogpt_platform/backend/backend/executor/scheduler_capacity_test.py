"""The limits on how many schedules one user holds and how large each is."""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from backend.data.model import CredentialsMetaInput
from backend.data.schedule import MAX_SCHEDULE_NAME_LENGTH
from backend.executor import scheduler as scheduler_module
from backend.executor.schedule_index_test_helpers import (
    _graph_job_kwargs,
    _mock_job,
    _scheduler_with_index,
)
from backend.executor.scheduler import (
    CopilotTurnJobArgs,
    GraphExecutionJobArgs,
    Jobstores,
    Scheduler,
    _index_entry,
)
from backend.integrations.providers import ProviderName
from backend.util.json import dumps as json_dumps


@pytest.fixture
def full_scheduler(monkeypatch: pytest.MonkeyPatch) -> Scheduler:
    """u1 holds the limit of two schedules; ``held-*`` are its jobs."""
    monkeypatch.setattr(scheduler_module.config, "max_schedules_per_user", 2)
    scheduler = _scheduler_with_index()
    # As run_service leaves it, so the count goes through the index.
    scheduler._persistent_jobstores = {Jobstores.EXECUTION.value: MagicMock()}
    held = {
        job_id: _mock_job(_args(job_id, "u1").model_dump(mode="json"))
        for job_id in ("held-1", "held-2")
    }
    assert scheduler._schedule_index is not None
    scheduler._schedule_index.upsert_many(
        [_index_entry(job_id, _args(job_id, "u1")) for job_id in held]
    )
    scheduler.scheduler.get_job.side_effect = lambda job_id, jobstore=None: held.get(
        job_id
    )
    # What a scan with no usable index returns: every account's jobs.
    scheduler.scheduler.get_jobs.return_value = list(held.values())
    scheduler.scheduler.add_job.side_effect = lambda *_, id, **__: _mock_job(
        _graph_job_kwargs(id, "u1", "g1")
    )
    return scheduler


def test_a_user_at_the_limit_cannot_add_another_schedule(full_scheduler: Scheduler):
    with pytest.raises(ValueError, match="the most one account can have"):
        _persist(full_scheduler, _args("new", "u1"))
    full_scheduler.scheduler.add_job.assert_not_called()


def test_pausing_a_schedule_does_not_make_room_for_another(full_scheduler: Scheduler):
    # A pause is persisted as next_run_time = NULL; the schedule itself stays,
    # so the count has to keep it.
    for job in _held(full_scheduler):
        job.next_run_time = None

    with pytest.raises(ValueError, match="the most one account can have"):
        _persist(full_scheduler, _args("new", "u1"))
    full_scheduler.scheduler.add_job.assert_not_called()


def test_the_limit_stays_per_user_when_the_index_is_not_ready(
    full_scheduler: Scheduler,
):
    # Startup, or a failed index: the count falls back to a scan of every
    # account's jobs, which must still only count the owner's.
    full_scheduler._schedule_index_ready = False

    _persist(full_scheduler, _args("theirs", "u2"))
    full_scheduler.scheduler.add_job.assert_called_once()


def test_a_routine_given_new_times_is_not_refused_for_its_own_jobs(
    full_scheduler: Scheduler,
):
    # Its replacement jobs are created before the old ones are deleted, so the
    # owner would otherwise be at the limit against itself.
    _persist(full_scheduler, _args("new", "u1"), replaces=["held-1", "held-2"])
    full_scheduler.scheduler.add_job.assert_called_once()


def test_replacing_a_schedule_under_its_own_id_is_allowed_at_the_limit(
    full_scheduler: Scheduler,
):
    _persist(full_scheduler, _args("held-1", "u1"))
    full_scheduler.scheduler.add_job.assert_called_once()


def test_the_owners_briefing_job_does_not_count_when_the_index_is_not_ready(
    full_scheduler: Scheduler,
):
    # The index holds only listable schedules, so the scan must count the same.
    full_scheduler._schedule_index_ready = False
    _held(full_scheduler).pop()
    _held(full_scheduler).append(_mock_job({"user_id": "u1"}))

    _persist(full_scheduler, _args("new", "u1"))
    full_scheduler.scheduler.add_job.assert_called_once()


def test_another_users_schedules_do_not_count_against_the_limit(
    full_scheduler: Scheduler,
):
    _persist(full_scheduler, _args("theirs", "u2"))
    full_scheduler.scheduler.add_job.assert_called_once()


@pytest.mark.parametrize("kind", ["graph", "copilot_turn"])
@pytest.mark.parametrize("length, stored", [(3_000, True), (4_096, False)])
def test_a_schedule_is_stored_only_within_the_size_limit(
    full_scheduler: Scheduler,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    length: int,
    stored: bool,
):
    monkeypatch.setattr(scheduler_module.config, "max_schedule_payload_bytes", 4_096)
    text = "x" * length
    args = (
        GraphExecutionJobArgs(
            **_graph_job_kwargs("new", "u2", "g1") | {"input_data": {"topic": text}}
        )
        if kind == "graph"
        else CopilotTurnJobArgs(
            schedule_id="new",
            user_id="u2",
            message=text,
            run_at=datetime(2026, 12, 1, tzinfo=UTC),
        )
    )

    if stored:
        _persist(full_scheduler, args)
        full_scheduler.scheduler.add_job.assert_called_once()
    else:
        with pytest.raises(ValueError, match="too large to save"):
            _persist(full_scheduler, args)
        full_scheduler.scheduler.add_job.assert_not_called()


@pytest.mark.parametrize("items, stored", [(900, True), (1_000, False)])
def test_a_schedule_is_stored_only_within_the_value_limit(
    full_scheduler: Scheduler,
    monkeypatch: pytest.MonkeyPatch,
    items: int,
    stored: bool,
):
    monkeypatch.setattr(scheduler_module.config, "max_schedule_payload_values", 1_000)
    args = GraphExecutionJobArgs(
        **_graph_job_kwargs("new", "u2", "g1") | {"input_data": {"topic": [[]] * items}}
    )

    if stored:
        _persist(full_scheduler, args)
        full_scheduler.scheduler.add_job.assert_called_once()
    else:
        with pytest.raises(ValueError, match="too many items to save"):
            _persist(full_scheduler, args)
        full_scheduler.scheduler.add_job.assert_not_called()


@pytest.mark.parametrize(
    "setting, limit, topic, refusal",
    [
        ("max_schedule_payload_bytes", 4_096, "x" * 4_096, "too large to save"),
        ("max_schedule_payload_values", 1_000, [[]] * 1_000, "too many items to save"),
    ],
)
def test_editing_a_schedules_inputs_past_a_size_limit_is_refused(
    full_scheduler: Scheduler,
    monkeypatch: pytest.MonkeyPatch,
    setting: str,
    limit: int,
    topic: str | list,
    refusal: str,
):
    monkeypatch.setattr(scheduler_module.config, setting, limit)
    validate = MagicMock()
    monkeypatch.setattr(scheduler_module, "run_async", validate)

    with pytest.raises(ValueError, match=refusal):
        full_scheduler.update_graph_execution_schedule(
            schedule_id="held-1", user_id="u1", input_data={"topic": topic}
        )
    validate.assert_not_called()
    full_scheduler.scheduler.modify_job.assert_not_called()


@pytest.mark.parametrize("edit", ["rename", "credentials"])
def test_an_edit_is_stored_only_within_the_size_limit(
    full_scheduler: Scheduler, monkeypatch: pytest.MonkeyPatch, edit: str
):
    """The limit holds for the job an edit writes: one renamed, or one whose
    inputs another edit changed after this edit's were checked."""
    topic = "x" * 3_000
    stored = _args("held-1", "u1", topic=topic).model_dump(mode="json")
    limit = _stored_bytes(stored) + 100
    monkeypatch.setattr(scheduler_module.config, "max_schedule_payload_bytes", limit)
    held = full_scheduler.scheduler.get_job("held-1")
    full_scheduler.scheduler.modify_job.side_effect = (
        lambda job_id, jobstore, **changes: _rewritten(held, changes["kwargs"])
    )
    landed: list[bool] = []

    def an_input_edit_lands(*_: Any) -> None:
        if not landed:
            landed.append(True)
            full_scheduler.update_graph_execution_schedule(
                schedule_id="held-1", user_id="u1", input_data={"topic": topic}
            )

    monkeypatch.setattr(scheduler_module, "run_async", an_input_edit_lands)
    if edit == "rename":
        held.kwargs = stored
        change: dict[str, Any] = {"name": "n" * MAX_SCHEDULE_NAME_LENGTH}
    else:
        credentials = CredentialsMetaInput(
            id="cred-1",
            title="t" * 200,
            provider=ProviderName("github"),
            type="api_key",
        )
        change = {"input_credentials": {"credentials": credentials}}

    with pytest.raises(ValueError, match="too large to save"):
        full_scheduler.update_graph_execution_schedule(
            schedule_id="held-1", user_id="u1", **change
        )
    assert len(landed) == (edit == "credentials")
    assert full_scheduler.scheduler.modify_job.call_count == len(landed)
    assert _stored_bytes(held.kwargs) <= limit


def test_a_rename_is_held_to_the_storage_budget(
    full_scheduler: Scheduler, monkeypatch: pytest.MonkeyPatch
):
    _store(full_scheduler, "big", "u1", "", "x" * 100_000)
    stored = sum(
        scheduler_module._stored_size(job.kwargs) for job in _held(full_scheduler)
    )
    monkeypatch.setattr(
        scheduler_module.config, "max_schedule_storage_bytes", stored + 100
    )

    with pytest.raises(ValueError, match="bytes together"):
        full_scheduler.update_graph_execution_schedule(
            schedule_id="big", user_id="u1", name="n" * MAX_SCHEDULE_NAME_LENGTH
        )
    full_scheduler.scheduler.modify_job.assert_not_called()


def test_an_edit_is_measured_with_the_name_it_sets(
    full_scheduler: Scheduler, monkeypatch: pytest.MonkeyPatch
):
    """A shorter name makes room for longer inputs in the same edit."""
    monkeypatch.setattr(scheduler_module, "run_async", MagicMock())
    stored = _args("held-1", "u1", topic="x" * 3_000).model_dump(mode="json") | {
        "agent_name": "n" * MAX_SCHEDULE_NAME_LENGTH
    }
    held = full_scheduler.scheduler.get_job("held-1")
    held.kwargs = stored
    monkeypatch.setattr(
        scheduler_module.config,
        "max_schedule_payload_bytes",
        _stored_bytes(stored) + 50,
    )
    full_scheduler.scheduler.modify_job.side_effect = (
        lambda job_id, jobstore, **changes: _rewritten(held, changes["kwargs"])
    )

    full_scheduler.update_graph_execution_schedule(
        schedule_id="held-1",
        user_id="u1",
        name="Daily",
        input_data={"topic": "x" * 3_150},
    )
    assert held.kwargs["agent_name"] == "Daily"


def test_a_job_name_is_stored_no_longer_than_a_schedule_name(
    full_scheduler: Scheduler,
):
    # Copilot follow-ups and routines pass names no request model bounded.
    full_scheduler._persist_schedule(
        dispatch_func=MagicMock(),
        job_args=_args("new", "u2"),
        trigger=MagicMock(),
        name="n" * (MAX_SCHEDULE_NAME_LENGTH + 1),
    )
    stored = full_scheduler.scheduler.add_job.call_args.kwargs["name"]
    assert stored == "n" * MAX_SCHEDULE_NAME_LENGTH


@pytest.mark.parametrize(
    "teammate_org, topic, stored",
    [
        (None, "x" * 50_000, True),
        ("org-a", "x" * 50_000, False),
        ("org-b", "x" * 50_000, True),
        (None, "x" * 50_000 + "\U0001f600", False),
        (None, [[]] * 2_500, False),
    ],
    ids=["alone", "teammate", "other-org", "wide-text", "many-values"],
)
def test_a_schedule_is_stored_only_within_the_storage_budget(
    full_scheduler: Scheduler,
    monkeypatch: pytest.MonkeyPatch,
    teammate_org: str | None,
    topic: str | list,
    stored: bool,
):
    monkeypatch.setattr(scheduler_module.config, "max_schedules_per_user", 10)
    monkeypatch.setattr(scheduler_module.config, "max_schedule_storage_bytes", 150_000)
    if teammate_org:
        _store(full_scheduler, "theirs", "u3", teammate_org, "x" * 100_000)
    args = _args("new", "u1", organization_id="org-a", topic=topic)

    if stored:
        _persist(full_scheduler, args)
        full_scheduler.scheduler.add_job.assert_called_once()
    else:
        with pytest.raises(ValueError, match="can store 150,000 bytes together"):
            _persist(full_scheduler, args)
        full_scheduler.scheduler.add_job.assert_not_called()


def test_a_teammates_schedules_count_when_the_index_is_not_ready(
    full_scheduler: Scheduler, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(scheduler_module.config, "max_schedules_per_user", 10)
    monkeypatch.setattr(scheduler_module.config, "max_schedule_storage_bytes", 150_000)
    _store(full_scheduler, "theirs", "u3", "org-a", "x" * 100_000)
    full_scheduler._schedule_index_ready = False

    with pytest.raises(ValueError, match="can store 150,000 bytes together"):
        _persist(full_scheduler, _args("new", "u1", "org-a", "x" * 50_000))
    full_scheduler.scheduler.add_job.assert_not_called()


@pytest.mark.parametrize("length, stored", [(100_000, True), (160_000, False)])
def test_an_edit_is_held_to_the_storage_budget_without_the_copy_it_replaces(
    full_scheduler: Scheduler,
    monkeypatch: pytest.MonkeyPatch,
    length: int,
    stored: bool,
):
    monkeypatch.setattr(scheduler_module.config, "max_schedule_storage_bytes", 150_000)
    monkeypatch.setattr(scheduler_module, "run_async", MagicMock())
    _store(full_scheduler, "big", "u1", "", "x" * 100_000)
    full_scheduler.scheduler.modify_job.side_effect = lambda job_id, **changes: (
        _mock_job(changes["kwargs"])
    )

    def edit() -> None:
        full_scheduler.update_graph_execution_schedule(
            schedule_id="big", user_id="u1", input_data={"topic": "y" * length}
        )

    if stored:
        edit()
        full_scheduler.scheduler.modify_job.assert_called_once()
    else:
        with pytest.raises(ValueError, match="can store 150,000 bytes together"):
            edit()
        full_scheduler.scheduler.modify_job.assert_not_called()


def _args(
    schedule_id: str,
    user_id: str,
    organization_id: str = "",
    topic: str | list | None = None,
) -> GraphExecutionJobArgs:
    inputs = {} if topic is None else {"topic": topic}
    return GraphExecutionJobArgs(
        **_graph_job_kwargs(schedule_id, user_id, "g1")
        | {"organization_id": organization_id, "input_data": inputs}
    )


def _persist(
    scheduler: Scheduler,
    args: GraphExecutionJobArgs,
    replaces: list[str] | None = None,
) -> None:
    scheduler._persist_schedule(
        dispatch_func=MagicMock(),
        job_args=args,
        trigger=MagicMock(),
        name="n",
        replaces=replaces,
    )


def _rewritten(job: MagicMock, kwargs: dict[str, Any]) -> MagicMock:
    job.kwargs = kwargs
    return job


def _stored_bytes(kwargs: dict[str, Any]) -> int:
    return len(json_dumps(kwargs).encode())


def _held(scheduler: Scheduler) -> list[MagicMock]:
    """u1's jobs, as the fixture's stubbed full scan returns them."""
    return scheduler.scheduler.get_jobs.return_value


def _store(
    scheduler: Scheduler, job_id: str, user_id: str, organization_id: str, topic: str
) -> None:
    """Add a stored schedule to the jobstore stub and the index."""
    args = _args(job_id, user_id, organization_id, topic)
    _held(scheduler).append(_mock_job(args.model_dump(mode="json")))
    assert scheduler._schedule_index is not None
    scheduler._schedule_index.upsert(_index_entry(job_id, args))
    jobs = {job.id: job for job in _held(scheduler)}
    scheduler.scheduler.get_job.side_effect = lambda job_id, jobstore=None: jobs.get(
        job_id
    )
