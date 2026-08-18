from datetime import datetime, timezone

from backend.api.features.experts.models import Expert, ExpertWorkflowRef
from backend.executor.scheduler import GraphExecutionJobInfo

from .activity import compose_upcoming_tasks
from .helpers import experts_by_schedule, next_runs_by_expert, parse_datetime

SHARED_GRAPH = "graph-shared"


def _expert(expert_id: str, *, schedule_id: str | None = None) -> Expert:
    return Expert(
        id=expert_id,
        name=f"Expert {expert_id}",
        avatar_url=None,
        role="Assistant",
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
        workflows=[
            ExpertWorkflowRef(
                id=f"workflow-{expert_id}",
                store_listing_version_id=None,
                library_agent_id=None,
                graph_id=SHARED_GRAPH,
                name="Inbox triage",
                description=None,
                schedule_id=schedule_id,
            )
        ],
    )


def _job(
    job_id: str, *, next_run: str, expert_id: str | None = None
) -> GraphExecutionJobInfo:
    return GraphExecutionJobInfo(
        id=job_id,
        schedule_id=job_id,
        name=job_id,
        next_run_time=next_run,
        user_id="user",
        graph_id=SHARED_GRAPH,
        graph_version=1,
        cron="0 9 * * *",
        input_data={},
        expert_id=expert_id,
    )


def test_jobs_sharing_a_graph_keep_their_own_owner() -> None:
    experts = [_expert("alice"), _expert("bob")]
    jobs = [
        _job("job-alice", next_run="2026-08-10T10:00:00Z", expert_id="alice"),
        _job("job-bob", next_run="2026-08-10T11:00:00Z", expert_id="bob"),
    ]

    owners = experts_by_schedule(experts, jobs)

    assert {job_id: expert.id for job_id, expert in owners.items()} == {
        "job-alice": "alice",
        "job-bob": "bob",
    }


def test_job_without_expert_stamp_falls_back_to_workflow_schedule() -> None:
    experts = [_expert("alice", schedule_id="job-legacy")]
    jobs = [_job("job-legacy", next_run="2026-08-10T10:00:00Z")]

    owners = experts_by_schedule(experts, jobs)

    assert owners["job-legacy"].id == "alice"


def test_unowned_job_is_left_unattributed() -> None:
    jobs = [_job("job-teammate", next_run="2026-08-10T10:00:00Z", expert_id="carol")]

    assert experts_by_schedule([_expert("alice")], jobs) == {}


def test_next_run_uses_earliest_of_an_experts_jobs() -> None:
    experts = [_expert("alice")]
    jobs = [
        _job("job-late", next_run="2026-08-10T18:00:00Z", expert_id="alice"),
        _job("job-early", next_run="2026-08-10T10:00:00Z", expert_id="alice"),
    ]

    next_runs = next_runs_by_expert(jobs, experts_by_schedule(experts, jobs))

    assert next_runs == {"alice": datetime(2026, 8, 10, 10, 0, tzinfo=timezone.utc)}


def test_upcoming_tasks_attribute_each_job_to_its_own_expert() -> None:
    experts = [_expert("alice"), _expert("bob")]
    jobs = [
        _job("job-alice", next_run="2026-08-10T10:00:00Z", expert_id="alice"),
        _job("job-bob", next_run="2026-08-10T11:00:00Z", expert_id="bob"),
    ]

    tasks = compose_upcoming_tasks(list(jobs), experts_by_schedule(experts, jobs))

    assert [(task.id, task.expert.id if task.expert else None) for task in tasks] == [
        ("job-alice", "alice"),
        ("job-bob", "bob"),
    ]


def test_parse_datetime_pins_naive_values_to_utc() -> None:
    assert parse_datetime("2026-08-10T09:00:00") == datetime(
        2026, 8, 10, 9, 0, tzinfo=timezone.utc
    )
    assert parse_datetime("2026-08-10T09:00:00Z") == datetime(
        2026, 8, 10, 9, 0, tzinfo=timezone.utc
    )


def test_parse_datetime_returns_none_for_garbage() -> None:
    assert parse_datetime("not-a-timestamp") is None
