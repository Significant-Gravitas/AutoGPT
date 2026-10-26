from datetime import datetime, timezone

import pytest

from backend.api.features.experts.models import Expert, ExpertWorkflowRef
from backend.api.features.library.model import LibraryAgentRef
from backend.executor.scheduler import GraphExecutionJobInfo

from .activity import compose_upcoming_tasks
from .helpers import (
    AgentRef,
    agent_refs_by_graph,
    experts_by_graph,
    experts_by_schedule,
    next_runs_by_expert,
    parse_datetime,
)

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


def test_expert_workflows_are_found_by_the_graph_they_run() -> None:
    owners = experts_by_graph([_expert("alice")])

    assert owners[SHARED_GRAPH].id == "alice"


def test_a_graph_two_experts_build_on_is_left_unattributed() -> None:
    assert experts_by_graph([_expert("alice"), _expert("bob")]) == {}


def test_an_expert_with_two_workflows_on_one_graph_still_owns_it() -> None:
    alice = _expert("alice")
    alice.workflows.append(
        alice.workflows[0].model_copy(update={"id": "workflow-alice-2"})
    )

    assert experts_by_graph([alice])[SHARED_GRAPH].id == "alice"


def test_an_experts_copy_of_a_workflow_keeps_the_library_picture() -> None:
    refs = [
        LibraryAgentRef(
            id="lib-live",
            graph_id=SHARED_GRAPH,
            name="Inbox triage",
            image_url="https://example.com/triage.png",
        )
    ]

    agents = agent_refs_by_graph([_expert("alice")], refs)

    assert agents[SHARED_GRAPH].image_url == "https://example.com/triage.png"


def test_parse_datetime_pins_naive_values_to_utc() -> None:
    assert parse_datetime("2026-08-10T09:00:00") == datetime(
        2026, 8, 10, 9, 0, tzinfo=timezone.utc
    )
    assert parse_datetime("2026-08-10T09:00:00Z") == datetime(
        2026, 8, 10, 9, 0, tzinfo=timezone.utc
    )


def test_parse_datetime_returns_none_for_garbage() -> None:
    assert parse_datetime("not-a-timestamp") is None


def test_a_removed_library_agent_still_names_its_runs_without_a_link() -> None:
    refs = [
        LibraryAgentRef(id="lib-live", graph_id="graph-live", name="Inbox triage"),
        LibraryAgentRef(
            id="lib-gone", graph_id="graph-gone", name="Sum Agent", is_deleted=True
        ),
    ]

    agents = agent_refs_by_graph([], refs)

    assert agents["graph-live"].library_agent_id == "lib-live"
    assert agents["graph-gone"].name == "Sum Agent"
    assert agents["graph-gone"].library_agent_id is None


def test_a_scheduled_workflow_missing_from_the_library_is_named_by_its_schedule() -> (
    None
):
    job = _job("job-1", next_run="2026-08-10T10:00:00Z").model_copy(
        update={"agent_name": "Smart Meeting Prep"}
    )
    unnamed_ref = LibraryAgentRef(id="lib-1", graph_id="graph-unnamed", name="")
    unnamed_job = job.model_copy(update={"graph_id": "graph-unnamed"})

    agents = agent_refs_by_graph([], [unnamed_ref], [job, unnamed_job])

    assert agents[SHARED_GRAPH] == AgentRef(
        name="Smart Meeting Prep", library_agent_id=None
    )
    assert agents["graph-unnamed"].name == "Smart Meeting Prep"
    assert agents["graph-unnamed"].library_agent_id == "lib-1"


def test_a_library_or_expert_name_outranks_the_schedules() -> None:
    job = _job("job-1", next_run="2026-08-10T10:00:00Z").model_copy(
        update={"agent_name": "Old schedule name"}
    )
    refs = [LibraryAgentRef(id="lib-1", graph_id=SHARED_GRAPH, name="Inbox triage")]

    assert agent_refs_by_graph([], refs, [job])[SHARED_GRAPH].name == "Inbox triage"
    by_expert = agent_refs_by_graph([_expert("alice")], [], [job])
    assert by_expert[SHARED_GRAPH].name == "Inbox triage"


@pytest.mark.parametrize(
    "names, expected",
    [
        (["Morning prep", "Evening prep"], None),
        (["Evening prep", "Morning prep"], None),
        (["Daily prep", "Daily prep"], "Daily prep"),
        (["Daily prep", None], None),
    ],
)
def test_a_graph_takes_a_schedule_name_only_when_its_schedules_agree(
    names: list[str | None], expected: str | None
) -> None:
    # Older runs carry no schedule id, so one schedule's name may not be theirs.
    jobs = [
        _job(f"job-{index}", next_run="2026-08-10T10:00:00Z").model_copy(
            update={"agent_name": name}
        )
        for index, name in enumerate(names)
    ]

    agent = agent_refs_by_graph([], [], jobs).get(SHARED_GRAPH)

    assert (agent.name if agent else None) == expected
