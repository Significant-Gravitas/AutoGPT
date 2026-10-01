from backend.executor.scheduler import CopilotTurnJobInfo, GraphExecutionJobInfo

Job = GraphExecutionJobInfo | CopilotTurnJobInfo


async def hidden_expert_ids(jobs: list[Job], user_id: str) -> set[str]:
    """Experts whose paused schedules must not be reachable.

    A schedule with no next run belonging to an expert the user no longer has
    is one ``detach_expert_triggers`` paused so re-hire can restore it. Deleting
    or resuming it from outside that flow loses the cadence permanently, so both
    the REST listing and the copilot tools have to hide the same rows — one
    batched query, because a request that previously did no expert work at all
    must not grow a lookup per archived expert.
    """
    # Imported here, not at module scope: experts_db reaches back into
    # copilot.tools, which imports this module, and that is a cycle.
    from backend.api.features.experts import experts_db

    paused = {j.expert_id for j in jobs if j.expert_id and not j.next_run_time}
    return paused - await experts_db.active_expert_ids(user_id, paused)


async def visible_graph_schedules(
    schedules: list[GraphExecutionJobInfo], user_id: str
) -> list[GraphExecutionJobInfo]:
    hidden = await hidden_expert_ids(list(schedules), user_id)
    return [s for s in schedules if is_visible_schedule(s, hidden)]


def is_visible_schedule(job: Job, hidden: set[str]) -> bool:
    """The one rule both readers apply, so neither can list what the other hides."""
    return bool(job.next_run_time) or job.expert_id not in hidden
