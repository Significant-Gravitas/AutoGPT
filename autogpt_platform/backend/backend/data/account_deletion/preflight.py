"""Read-only: what the account holds, and what would stop its deletion.

Runs before anything is deleted, so a blocked deletion leaves the account
exactly as it was.
"""

from datetime import datetime, timedelta, timezone

from prisma.enums import OrgMemberStatus
from prisma.models import (
    AuthUser,
    Expert,
    OAuthApplication,
    Organization,
    OrganizationSubscription,
    OrgMember,
    User,
    UserWorkspace,
)

from backend.data.account_deletion.deleted_user import DELETED_USER_ID
from backend.data.account_deletion.models import SAFE_ID, UNDO_WINDOW, Blocker, Subject


async def load_subject(user_id: str, email: str | None = None) -> Subject:
    user = await User.prisma().find_unique(where={"id": user_id})
    identity = await AuthUser.prisma().find_unique(where={"id": user_id})
    workspace = await UserWorkspace.prisma().find_unique(where={"userId": user_id})
    apps = await OAuthApplication.prisma().find_many(where={"ownerId": user_id})
    experts = await Expert.prisma().find_many(where={"ownerUserId": user_id})
    sole_orgs, _ = await _orgs(user_id)
    return Subject(
        user_id=user_id,
        email=(user and user.email) or (identity and identity.email) or email,
        has_user_row=user is not None,
        has_auth_identity=identity is not None,
        workspace_id=workspace.id if workspace else None,
        oauth_app_ids=[app.id for app in apps],
        sole_org_ids=[org.id for org in sole_orgs],
        stripe_customer_ids=await _stripe_customers(user, sole_orgs),
        expert_ids=[expert.id for expert in experts],
        locked_at=user.deletionLockedAt if user else None,
    )


async def find_blockers(
    subject: Subject, *, force: bool = False, now: datetime | None = None
) -> list[Blocker]:
    """What stops the full deletion. `force` skips the undo window, and lets
    an ID with no account left through to finish a deletion that stopped
    part way; nothing else."""
    if refused := invalid_id(subject.user_id) or system_account(subject.user_id):
        return refused
    blockers: list[Blocker] = []
    if not force:
        if not (subject.has_user_row or subject.has_auth_identity):
            blockers.append(
                Blocker(
                    name="account_not_found",
                    count=0,
                    resolution="No platform account or auth identity has this "
                    "ID: it is deleted already, or the ID is wrong. To finish a "
                    "deletion that stopped part way, run it again with --force.",
                )
            )
        elif subject.has_user_row:
            blockers += _undo_window(
                subject.locked_at, now or datetime.now(timezone.utc)
            )
    _, solely_owned_shared_orgs = await _orgs(subject.user_id)
    if solely_owned_shared_orgs:
        blockers.append(
            Blocker(
                name="shared_orgs_without_another_owner",
                count=len(solely_owned_shared_orgs),
                resolution="This account is the only owner of an organization "
                "with other members. Make another member an owner first.",
            )
        )
    return blockers


def invalid_id(user_id: str) -> list[Blocker]:
    """An ID that cannot be an account's. Refused before any step runs, so
    nothing (Stripe, the scheduler, a storage prefix) is ever asked about it."""
    if SAFE_ID.fullmatch(user_id):
        return []
    return [
        Blocker(
            name="invalid_account_id",
            count=1,
            resolution="This is not an account ID. Pass the user ID or the "
            "account's email.",
        )
    ]


def system_account(user_id: str) -> list[Blocker]:
    """The placeholder that owns deleted creators' shared agents."""
    if user_id != DELETED_USER_ID:
        return []
    return [
        Blocker(
            name="system_account",
            count=1,
            resolution="This is the account deleted creators' shared agents "
            "belong to. It is never locked or deleted.",
        )
    ]


def _undo_window(locked_at: datetime | None, now: datetime) -> list[Blocker]:
    """The deletion waits `UNDO_WINDOW` after the lock, so a request can
    still be withdrawn. The count is the days left."""
    if locked_at is None:
        return [
            Blocker(
                name="not_locked",
                count=UNDO_WINDOW.days,
                resolution="Lock the account first (--lock --apply); the "
                f"deletion can run {UNDO_WINDOW.days} days later. --force "
                "skips the wait.",
            )
        ]
    due = locked_at + UNDO_WINDOW
    if now >= due:
        return []
    return [
        Blocker(
            name="undo_window_open",
            # Whole days left, rounded up.
            count=-((now - due) // timedelta(days=1)),
            resolution=f"Locked on {locked_at:%Y-%m-%d}; the deletion can run "
            f"from {due:%Y-%m-%d %H:%M} UTC. --force skips the wait.",
        )
    ]


async def _orgs(user_id: str) -> tuple[list[Organization], list[Organization]]:
    """The orgs no one else is an active member of, and the shared orgs no
    one else actively owns. Invited, suspended and removed members count for
    neither: an org only they are left in goes with the account, its
    subscription cancelled. A personal org whose membership row is gone still
    counts as its own."""
    memberships = await OrgMember.prisma().find_many(
        where={"userId": user_id}, include={"Org": True}
    )
    owns = {m.orgId for m in memberships if m.isOwner}
    orgs = {m.orgId: m.Org for m in memberships if m.Org}
    for org in await Organization.prisma().find_many(
        where={"isPersonal": True, "bootstrapUserId": user_id}
    ):
        orgs.setdefault(org.id, org)
    sole: list[Organization] = []
    solely_owned_shared: list[Organization] = []
    for org in orgs.values():
        others = await OrgMember.prisma().find_many(
            where={
                "orgId": org.id,
                "userId": {"not": user_id},
                "status": OrgMemberStatus.ACTIVE,
            }
        )
        if not others:
            sole.append(org)
        elif org.id in owns and not any(m.isOwner for m in others):
            solely_owned_shared.append(org)
    return sole, solely_owned_shared


async def _stripe_customers(user: User | None, orgs: list[Organization]) -> list[str]:
    subscriptions = await OrganizationSubscription.prisma().find_many(
        where={"organizationId": {"in": [org.id for org in orgs]}}
    )
    ids = [user.stripeCustomerId if user else None]
    ids += [org.stripeCustomerId for org in orgs]
    ids += [sub.stripeCustomerId for sub in subscriptions]
    return list(dict.fromkeys(i for i in ids if i))
