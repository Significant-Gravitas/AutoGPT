"""Team-capacity limits and the failures the expert DB layer raises.

Kept free of Prisma and of the DB module itself so callers that only need to
name a failure — the copilot hire/raise tools, for one — can import it without
pulling the query layer in behind it.

The four failures a copilot tool has to tell apart are defined in
``backend.util.exceptions`` and re-exported here: ``EXCEPTION_MAPPING`` is
built from that module alone, so an exception class defined anywhere else
reaches the Prisma-less copilot executor as a retried ``HTTPServerError``
instead of its own type.
"""

from backend.util.exceptions import (
    ExpertHireUnavailableError,
    ExpertLimitExceededError,
    ExpertTemplateNotFoundError,
    RaisedExpertLifetimeLimitExceededError,
)

__all__ = [
    "ACTIVE_EXPERT_LIMIT",
    "LIFETIME_RAISED_EXPERT_LIMIT",
    "ExpertHireUnavailableError",
    "ExpertLimitExceededError",
    "ExpertPodLimitReachedError",
    "ExpertPodNameTakenError",
    "ExpertPodNotFoundError",
    "ExpertTemplateNotFoundError",
    "RaisedExpertLifetimeLimitExceededError",
]

# The active cap bounds team-list fan-out. The lifetime raised-expert cap also
# bounds durable rows when users repeatedly raise and archive experts.
ACTIVE_EXPERT_LIMIT = 20
LIFETIME_RAISED_EXPERT_LIMIT = 100


class ExpertPodNotFoundError(Exception):
    def __init__(self, pod_id: str):
        super().__init__(f"Pod {pod_id} not found")
        self.pod_id = pod_id


class ExpertPodNameTakenError(Exception):
    def __init__(self, name: str):
        super().__init__(f"A pod named {name!r} already exists")
        self.name = name


class ExpertPodLimitReachedError(Exception):
    def __init__(self, limit: int):
        super().__init__(f"You can have at most {limit} pods")
        self.limit = limit
