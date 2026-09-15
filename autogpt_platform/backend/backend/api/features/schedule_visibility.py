from backend.api.features.experts import experts_db
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
    paused = {j.expert_id for j in jobs if j.expert_id and not j.next_run_time}
    return paused - await experts_db.active_expert_ids(user_id, paused)


async def visible_graph_schedules(
    schedules: list[GraphExecutionJobInfo], user_id: str
) -> list[GraphExecutionJobInfo]:
    hidden = await hidden_expert_ids(list(schedules), user_id)
    return [
        schedule
        for schedule in schedules
        if schedule.next_run_time
        or (schedule.cron and schedule.expert_id not in hidden)
    ]
