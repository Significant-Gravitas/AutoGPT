import re
from datetime import datetime, timedelta
from enum import Enum

from pydantic import BaseModel

# How long a locked account waits before the full deletion may run, so a
# request can still be undone (`delete-account --unlock`).
UNDO_WINDOW = timedelta(days=14)

# Account IDs are UUIDs. Anything else is refused before a step runs
# (`preflight.invalid_id`) and never turned into a storage prefix
# (`storage.py`), so a malformed ID can never widen a sweep.
SAFE_ID = re.compile(r"[0-9A-Za-z-]{8,64}")


class StepStatus(str, Enum):
    DONE = "done"
    # Dry run: the counts are what an --apply would remove.
    WOULD_DO = "would_do"
    # Not configured on this stack (no bucket, no PostHog key, ...).
    SKIPPED = "skipped"
    FAILED = "failed"


class StepReport(BaseModel):
    """One step's outcome. Counts and short notes only: never an email, a
    name or a storage path, since the report is printed and logged."""

    name: str
    status: StepStatus
    counts: dict[str, int] = {}
    note: str = ""


class Blocker(BaseModel):
    """Something the routine will not decide on its own. Any blocker stops an
    --apply before it deletes anything."""

    name: str
    count: int
    resolution: str


class DeletionReport(BaseModel):
    user_id: str
    dry_run: bool
    # False when neither a platform User nor an auth identity was found.
    account_found: bool = True
    blockers: list[Blocker] = []
    steps: list[StepReport] = []

    @property
    def failed(self) -> bool:
        return any(step.status == StepStatus.FAILED for step in self.steps)

    @property
    def complete(self) -> bool:
        """Every step ran through on an --apply, so nothing is left."""
        return not (self.dry_run or self.blockers or self.failed)


class Subject(BaseModel):
    """What the steps need to know about the account, read once before any of
    them runs: the platform rows go last, and some steps (MailerLite, the
    storage prefixes) cannot be found again once they are gone."""

    user_id: str
    email: str | None = None
    has_user_row: bool = False
    has_auth_identity: bool = False
    workspace_id: str | None = None
    oauth_app_ids: list[str] = []
    # Organizations only this account belongs to: its personal org, and any
    # other org it is the sole member of. They go with the account.
    sole_org_ids: list[str] = []
    # The account's own Stripe customer and those of its sole orgs.
    stripe_customer_ids: list[str] = []
    # Experts it owns, each with its own memory graph.
    expert_ids: list[str] = []
    # When `delete-account --lock` stopped the account; the full deletion
    # waits `UNDO_WINDOW` from then.
    locked_at: datetime | None = None
    # Its agents other accounts still use (`shared_agents.py`): handed to the
    # "deleted user" account instead of deleted, all versions of each.
    shared_graph_ids: list[str] = []
    # Its marketplace listings of those agents, handed over with them.
    kept_listing_ids: list[str] = []


class AccountDeletionBlocked(RuntimeError):
    def __init__(self, report: DeletionReport):
        names = ", ".join(blocker.name for blocker in report.blockers)
        super().__init__(f"Account deletion blocked: {names}")
        self.report = report
