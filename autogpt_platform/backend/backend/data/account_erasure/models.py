import re
from enum import Enum

from prisma.enums import AccountDeletionSource
from pydantic import BaseModel

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


class ErasureReport(BaseModel):
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
    # The Stripe customers of its sole orgs, other than the account's own:
    # closing the account (`account_deletion.service.delete_account`) already
    # cancelled that one's subscriptions.
    org_stripe_customer_ids: list[str] = []
    # Experts it owns, each with its own memory graph.
    expert_ids: list[str] = []
    # The `source` of its `AccountDeletionRequest` (None without one): the
    # account is closed, and the erasure may run. Without one it is refused
    # (`preflight.find_blockers`). The confirmation email words its reason by
    # it.
    request_source: AccountDeletionSource | None = None
    # Whether the account-deleted email went out (the request's
    # `confirmationSentAt`); the erasure sends it first when it did not.
    confirmation_sent: bool = False
    # Its sign-in is banned for a reason other than closing it (an abuse
    # ban, say). Settings refuses to delete such an account, and so does
    # `--close` unless the operator allows it (`preflight.find_blockers`).
    banned_for_another_reason: bool = False
    # The request's `completedAt` is set: an earlier erasure finished.
    request_completed: bool = False
    # Its agents other accounts still use (`shared_agents.py`): handed to the
    # "deleted user" account instead of deleted, all versions of each.
    shared_graph_ids: list[str] = []
    # Its marketplace listings of those agents, handed over with them.
    kept_listing_ids: list[str] = []

    @property
    def has_request(self) -> bool:
        return self.request_source is not None


class AccountErasureBlocked(RuntimeError):
    def __init__(self, report: ErasureReport):
        names = ", ".join(blocker.name for blocker in report.blockers)
        super().__init__(f"Account erasure blocked: {names}")
        self.report = report
