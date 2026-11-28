"""One writer at a time on a chat's turn at the budget reset, on any replica.

A chat has at most one resume and one send at the reset, each a scheduler job
under the chat's own id (``at_reset_schedule_id``). Three things read that job
and then set it: the API joining a new send onto the one waiting, the
scheduler putting a job back after it ran, and the API cancelling one. API
servers and schedulers each run as several replicas, so each of those holds
this Redis lock, keyed on the job's id, from its read to its write.
"""

from contextlib import AbstractAsyncContextManager

from backend.copilot.scheduler_write_lock import scheduler_write_lock


def at_reset_job_lock(job_id: str) -> AbstractAsyncContextManager[bool]:
    return scheduler_write_lock(f"copilot:at_reset_lock:{job_id}")
