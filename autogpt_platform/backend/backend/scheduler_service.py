"""The scheduler service as it is deployed, with every feature's system jobs."""

from backend.copilot.dream.system_jobs import dream_system_jobs
from backend.executor.scheduler import Scheduler, scheduler_system_jobs

# The scheduler imports none of the features: a feature declares its jobs beside
# its code and is listed here.
SYSTEM_JOBS = (scheduler_system_jobs, dream_system_jobs)


def scheduler_service() -> Scheduler:
    return Scheduler(system_jobs=SYSTEM_JOBS)
