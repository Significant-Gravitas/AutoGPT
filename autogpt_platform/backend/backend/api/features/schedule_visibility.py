from backend.api.features.experts import experts_db
from backend.executor.scheduler import GraphExecutionJobInfo


async def visible_graph_schedules(
    schedules: list[GraphExecutionJobInfo], user_id: str
) -> list[GraphExecutionJobInfo]:
    paused_expert_ids = {
        schedule.expert_id
        for schedule in schedules
        if schedule.expert_id and not schedule.next_run_time
    }
    active_expert_ids = {
        expert_id
        for expert_id in paused_expert_ids
        if await experts_db.owns_active_expert(user_id, expert_id)
    }
    return [
        schedule
        for schedule in schedules
        if schedule.next_run_time
        or (
            schedule.cron
            and (not schedule.expert_id or schedule.expert_id in active_expert_ids)
        )
    ]
