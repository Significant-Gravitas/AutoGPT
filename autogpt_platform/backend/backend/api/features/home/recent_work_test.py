from datetime import datetime, timedelta, timezone

from backend.api.features.experts.models import Expert
from backend.data.activity_event import ActivityEvent, ActivityEventCategory
from backend.data.execution import ExecutionStatus, GraphExecutionMeta

from .helpers import AgentRef
from .recent_work import _MAX_ITEMS_PER_GROUP, _MAX_RUNS_PER_GROUP, compose_recent_work

NOW = datetime(2026, 8, 28, 9, 0, tzinfo=timezone.utc)
AGENTS = {
    "graph-notes": AgentRef(name="Release Note Generator", library_agent_id="lib-1")
}


def _expert(expert_id: str, name: str) -> Expert:
    return Expert(
        id=expert_id,
        name=name,
        avatar_url=None,
        role="Marketing",
        tagline=None,
        bio=None,
        skills=[],
        identity="",
        voice_preferences="",
        boundaries="",
        protected_soul_rules=[],
        is_template=False,
        source_template_id=None,
        is_archived=False,
        workflows=[],
        last_run_status=None,
        schedules_paused_at=None,
    )


def _execution(
    *,
    exec_id: str,
    graph_id: str = "graph-notes",
    status: ExecutionStatus = ExecutionStatus.COMPLETED,
    minutes_ago: int = 0,
    expert_id: str | None = None,
    activity_status: str | None = "I generated release notes.",
) -> GraphExecutionMeta:
    ended_at = NOW - timedelta(minutes=minutes_ago)
    return GraphExecutionMeta(
        id=exec_id,
        user_id="user-1",
        graph_id=graph_id,
        graph_version=1,
        inputs=None,
        credential_inputs=None,
        nodes_input_masks=None,
        preset_id=None,
        status=status,
        started_at=ended_at - timedelta(minutes=1),
        ended_at=ended_at,
        expert_id=expert_id,
        stats=GraphExecutionMeta.Stats(
            activity_status=activity_status, error=None, duration=60.0, cost=7
        ),
    )


def _event(
    *,
    event_id: str,
    category: ActivityEventCategory = "FILE",
    minutes_ago: int = 0,
    expert_id: str | None = None,
    session_id: str | None = None,
    graph_exec_id: str | None = None,
    title: str = "draft.md",
    provider: str | None = None,
    object_id: str | None = None,
) -> ActivityEvent:
    return ActivityEvent(
        id=event_id,
        user_id="user-1",
        created_at=NOW - timedelta(minutes=minutes_ago),
        category=category,
        event_type="file.created" if category == "FILE" else "integration.action",
        title=title,
        expert_id=expert_id,
        session_id=session_id,
        graph_exec_id=graph_exec_id,
        provider=provider,
        object_id=object_id,
    )


def _compose(
    executions: list[GraphExecutionMeta] | None = None,
    events: list[ActivityEvent] | None = None,
    experts: dict[str, Expert] | None = None,
    session_titles: dict[str, str | None] | None = None,
):
    return compose_recent_work(
        now=NOW,
        executions=executions or [],
        events=events or [],
        expert_by_id=experts or {},
        agent_by_graph=AGENTS,
        session_titles=session_titles or {},
    )


def test_groups_runs_and_deliverables_under_the_actor_that_did_them() -> None:
    maria = _expert("maria", "Maria")
    executions = [
        _execution(exec_id="run-maria", expert_id="maria", minutes_ago=30),
        _execution(exec_id="run-notes", minutes_ago=5),
    ]
    events = [
        _event(event_id="e1", expert_id="maria", session_id="s1", minutes_ago=1),
        _event(
            event_id="e2",
            category="INTEGRATION",
            title="Send Email",
            provider="google",
            graph_exec_id="run-notes",
            minutes_ago=6,
        ),
    ]

    work = _compose(executions, events, {"maria": maria}, {"s1": "Blog pipeline"})

    assert [group.actor.name for group in work.groups] == [
        "Maria",
        "Release Note Generator",
    ]
    maria_group, notes_group = work.groups
    assert maria_group.actor.kind == "expert"
    assert maria_group.actor.expert is not None
    assert maria_group.actor.link == "/copilot?expertId=maria"
    assert [run.id for run in maria_group.runs] == ["run-maria"]
    assert [item.id for item in maria_group.items] == ["e1"]
    assert maria_group.items[0].session_title == "Blog pipeline"
    assert maria_group.items[0].link == "/copilot?sessionId=s1"

    assert notes_group.actor.kind == "workflow"
    assert notes_group.actor.link == "/library/agents/lib-1"
    assert [run.id for run in notes_group.runs] == ["run-notes"]
    assert [item.id for item in notes_group.items] == ["e2"]
    assert notes_group.items[0].link is None


def test_an_executor_event_follows_its_run_to_the_expert() -> None:
    maria = _expert("maria", "Maria")
    executions = [_execution(exec_id="run-1", expert_id="maria")]
    events = [
        _event(
            event_id="e1",
            category="INTEGRATION",
            title="Post Message",
            provider="slack",
            graph_exec_id="run-1",
        )
    ]

    work = _compose(executions, events, {"maria": maria})

    assert len(work.groups) == 1
    assert work.groups[0].actor.kind == "expert"
    assert [item.id for item in work.groups[0].items] == ["e1"]


def test_thread_work_without_an_expert_is_autopilots() -> None:
    work = _compose(events=[_event(event_id="e1", session_id="s2")])

    group = work.groups[0]
    assert group.actor.kind == "autopilot"
    assert group.actor.name == "Autopilot"
    assert group.actor.link == "/copilot"
    assert group.items[0].link == "/copilot?sessionId=s2"


def test_llm_credential_use_is_not_delivered_work() -> None:
    events = [
        _event(
            event_id="e1",
            category="INTEGRATION",
            title="AITextGeneratorBlock",
            provider="openai",
            graph_exec_id="run-1",
        ),
        _event(
            event_id="e2",
            category="INTEGRATION",
            title="Send Email",
            provider="google",
            graph_exec_id="run-1",
        ),
        _event(event_id="run-event", category="RUN", session_id="s1"),
    ]

    work = _compose(events=events)

    assert work.total_count == 1
    assert [item.id for group in work.groups for item in group.items] == ["e2"]


def test_groups_are_capped_and_report_the_overflow() -> None:
    executions = [
        _execution(exec_id=f"run-{index}", minutes_ago=index)
        for index in range(_MAX_RUNS_PER_GROUP + 2)
    ]
    events = [
        _event(event_id=f"e{index}", graph_exec_id="run-0", minutes_ago=index)
        for index in range(_MAX_ITEMS_PER_GROUP + 3)
    ]

    work = _compose(executions, events)

    group = work.groups[0]
    assert len(group.runs) == _MAX_RUNS_PER_GROUP
    assert len(group.items) == _MAX_ITEMS_PER_GROUP
    assert group.more_count == 5
    assert work.total_count == _MAX_ITEMS_PER_GROUP + 3


def test_counts_the_weeks_runs_and_orders_groups_by_latest_activity() -> None:
    maria = _expert("maria", "Maria")
    executions = [
        _execution(exec_id="old", minutes_ago=8 * 24 * 60),
        _execution(exec_id="failed", status=ExecutionStatus.FAILED, minutes_ago=50),
        _execution(exec_id="fresh", expert_id="maria", minutes_ago=10),
        _execution(exec_id="done", minutes_ago=120),
    ]

    work = _compose(executions, experts={"maria": maria})

    assert work.window_started_at == NOW - timedelta(days=7)
    assert work.completed_count == 2
    assert work.failed_count == 1
    assert [group.actor.name for group in work.groups] == [
        "Maria",
        "Release Note Generator",
    ]
    # Failures lead within a group so they get looked at first.
    assert [run.id for run in work.groups[1].runs] == ["failed", "done"]


def test_a_removed_expert_keeps_its_attribution_kind() -> None:
    work = _compose(events=[_event(event_id="e1", expert_id="gone", session_id="s1")])

    group = work.groups[0]
    assert group.actor.kind == "expert"
    assert group.actor.name == "Expert"
    assert group.actor.link is None


def test_file_items_carry_file_id_and_mime_type() -> None:
    event = _event(event_id="e1", session_id="s1", object_id="file-9")
    event.data = {"mime_type": "text/markdown", "size_bytes": 10}

    work = _compose(events=[event])

    item = work.groups[0].items[0]
    assert item.file_id == "file-9"
    assert item.mime_type == "text/markdown"
