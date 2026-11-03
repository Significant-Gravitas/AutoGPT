"""Starts and resumes expert work on tasks (stub — replaced by dispatch work)."""


async def on_task_assigned(user_id: str, task_id: str) -> None:
    return None


async def resume_task(user_id: str, task_id: str, message: str) -> None:
    return None


async def on_work_turn_finished(session_id: str, status: str) -> None:
    return None
