"""The undo window. `delete-account --lock` stops the account (the first steps
of the deletion) and records when; the full deletion waits `UNDO_WINDOW` from
then (`preflight.find_blockers`); `--unlock` lets the account back in.

The unlock lifts only what the deletion did: its own ban (a ban already in
force when the account was locked was kept, and stays), and the marketing
opt-out it set (one the account had chosen itself stays). Cancelled
subscriptions and deleted schedules are not restored.
"""

from datetime import datetime, timezone

from prisma.models import AuthUser, User

from backend.data.account_deletion.activity import BAN_REASON, OPT_OUT_SOURCE
from backend.data.account_deletion.models import (
    UNDO_WINDOW,
    StepReport,
    StepStatus,
    Subject,
)

NOT_RESTORED = (
    "Subscriptions stay cancelled and deleted schedules are not restored; the "
    "requester sets them up again."
)


async def record_lock(subject: Subject, *, dry_run: bool) -> StepReport:
    """Record the lock, once: locking again keeps the first date."""
    if not subject.has_user_row:
        return StepReport(
            name="lock",
            status=StepStatus.SKIPPED,
            note="no platform account row to record the lock on; the full "
            "deletion needs --force",
        )
    locked_at = subject.locked_at or _now()
    if not dry_run and subject.locked_at is None:
        await User.prisma().update_many(
            where={"id": subject.user_id, "deletionLockedAt": None},
            data={"deletionLockedAt": locked_at},
        )
    return StepReport(
        name="lock",
        status=StepStatus.WOULD_DO if dry_run else StepStatus.DONE,
        counts={"locked": 0 if subject.locked_at else 1},
        note=f"full deletion from {locked_at + UNDO_WINDOW:%Y-%m-%d %H:%M} UTC",
    )


async def lift_lock(subject: Subject, *, dry_run: bool) -> StepReport:
    uid = subject.user_id
    # The ban the lock wrote: never one with an expiry, and never one that was
    # in force before (`activity.disable_sign_in` keeps those as they are).
    banned = {"id": uid, "banned": True, "banReason": BAN_REASON, "banExpires": None}
    opted_out = {"id": uid, "marketingOptOutSource": OPT_OUT_SOURCE}
    locked = {"id": uid, "deletionLockedAt": {"not": None}}
    if dry_run:
        counts = {
            "ban_lifted": await AuthUser.prisma().count(where=banned),
            "marketing_opt_out_reverted": await User.prisma().count(where=opted_out),
            "lock_cleared": await User.prisma().count(where=locked),
        }
    else:
        counts = {
            "ban_lifted": await AuthUser.prisma().update_many(
                where=banned,
                data={"banned": False, "banReason": None, "banExpires": None},
            ),
            "marketing_opt_out_reverted": await User.prisma().update_many(
                where=opted_out,
                data={"marketingOptOutAt": None, "marketingOptOutSource": None},
            ),
            "lock_cleared": await User.prisma().update_many(
                where=locked, data={"deletionLockedAt": None}
            ),
        }
    return StepReport(
        name="unlock",
        status=StepStatus.WOULD_DO if dry_run else StepStatus.DONE,
        counts=counts,
        note=NOT_RESTORED,
    )


def _now() -> datetime:
    return datetime.now(timezone.utc)
