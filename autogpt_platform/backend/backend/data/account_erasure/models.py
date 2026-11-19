import re
from datetime import datetime
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
    # Why an --apply ran no step although nothing blocked it: the request
    # could not be claimed (`request.claim_request`), because the owner kept
    # the account, its erasure date is still ahead, or it is erased already.
    not_claimed: str = ""

    @property
    def failed(self) -> bool:
        return any(step.status == StepStatus.FAILED for step in self.steps)

    @property
    def complete(self) -> bool:
        """Every step ran through on an --apply, so nothing is left."""
        return not (self.dry_run or self.blockers or self.failed or self.not_claimed)


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
    # Its own Stripe customer and those of its sole orgs. Closing the account
    # (`account_deletion.service.delete_account`) cancelled its own customer's
    # subscriptions, but one paid for in the seconds after the close survives.
    stripe_customer_ids: list[str] = []
    # Experts it owns, each with its own memory graph.
    expert_ids: list[str] = []
    # The `source` of its `AccountDeletionRequest` (None without one): the
    # account is closed, and the erasure may run. Without one it is refused
    # (`preflight.find_blockers`). The confirmation email words its reason by
    # it.
    request_source: AccountDeletionSource | None = None
    # The request's ID once this erasure claimed it (`request.claim_request`):
    # the completion marks that row, and only that one.
    request_id: str | None = None
    # When the owner's 14 days to keep the account end: the erasure may claim
    # the request only after it.
    erase_after: datetime | None = None
    # The owner asked to be erased at once: the request's 14 days were cut
    # short (`--close --now`). The late confirmation says so.
    erase_asked_now: bool = False
    # Whether the account-deleted email went out (the request's
    # `confirmationSentAt`); the erasure sends it first when it did not.
    confirmation_sent: bool = False
    # The operator closed it without the email (`confirmationSkipped`,
    # `--no-email`): the erasure sends none.
    confirmation_skipped: bool = False
    # The request's `completedAt` is set: an earlier erasure finished.
    request_completed: bool = False
    # Its agents other accounts still use (`shared_agents.py`): handed to the
    # "deleted user" account instead of deleted, all versions of each.
    shared_graph_ids: list[str] = []
    # Its marketplace listings of those agents, handed over with them.
    kept_listing_ids: list[str] = []
    # A ban on the auth identity still in force when the erasure reads it:
    # the erasure keeps a ban record (`ban_record.py`) so the address can't
    # sign up again while it lasts. `ban_expires` None is a ban for good.
    banned: bool = False
    ban_expires: datetime | None = None

    @property
    def has_request(self) -> bool:
        return self.request_source is not None


class AccountErasureBlocked(RuntimeError):
    def __init__(self, report: ErasureReport):
        names = ", ".join(blocker.name for blocker in report.blockers)
        super().__init__(f"Account erasure blocked: {names}")
        self.report = report
