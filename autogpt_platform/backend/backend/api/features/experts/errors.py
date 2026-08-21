"""Team-capacity limits and the failures the expert DB layer raises.

Kept free of Prisma and of the DB module itself so callers that only need to
name a failure — the copilot hire/raise tools, for one — can import it without
pulling the query layer in behind it.
"""

# The active cap bounds team-list fan-out. The lifetime raised-expert cap also
# bounds durable rows when users repeatedly raise and archive experts.
ACTIVE_EXPERT_LIMIT = 20
LIFETIME_RAISED_EXPERT_LIMIT = 100


class ExpertTemplateNotFoundError(Exception):
    def __init__(self, template_id: str):
        super().__init__(f"Expert template {template_id} not found")
        self.template_id = template_id


class ExpertHireUnavailableError(Exception):
    def __init__(self, expert_id: str):
        super().__init__(expert_id)
        self.expert_id = expert_id


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


class ExpertLimitExceededError(Exception):
    def __init__(self, limit: int):
        super().__init__(f"Active expert limit of {limit} reached")
        self.limit = limit


class RaisedExpertLifetimeLimitExceededError(Exception):
    def __init__(self, limit: int):
        super().__init__(f"Raised expert lifetime limit of {limit} reached")
        self.limit = limit
