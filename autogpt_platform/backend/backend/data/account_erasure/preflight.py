"""Read-only: what the account holds, and what would stop its erasure.

Runs before anything is deleted, so a blocked erasure leaves the account
exactly as it was.
"""

from datetime import UTC, datetime

from prisma.enums import OrgMemberStatus
from prisma.models import (
    AccountDeletionRequest,
    AuthUser,
    Expert,
    OAuthApplication,
    Organization,
    OrganizationSubscription,
    OrgMember,
    User,
    UserWorkspace,
)

from backend.data.account_deletion import ACCOUNT_DELETED_BAN_REASON
from backend.data.account_erasure.deleted_user import DELETED_USER_ID
from backend.data.account_erasure.models import SAFE_ID, Blocker, Subject


async def load_subject(user_id: str, email: str | None = None) -> Subject:
    user = await User.prisma().find_unique(where={"id": user_id})
    identity = await AuthUser.prisma().find_unique(where={"id": user_id})
    workspace = await UserWorkspace.prisma().find_unique(where={"userId": user_id})
    apps = await OAuthApplication.prisma().find_many(where={"ownerId": user_id})
    experts = await Expert.prisma().find_many(where={"ownerUserId": user_id})
    request = await AccountDeletionRequest.prisma().find_unique(
        where={"userId": user_id}
    )
    sole_orgs, _ = await _orgs(user_id)
    return Subject(
        user_id=user_id,
        email=(user and user.email) or (identity and identity.email) or email,
        has_user_row=user is not None,
        has_auth_identity=identity is not None,
        workspace_id=workspace.id if workspace else None,
        oauth_app_ids=[app.id for app in apps],
        sole_org_ids=[org.id for org in sole_orgs],
        org_stripe_customer_ids=await _org_stripe_customers(user, sole_orgs),
        expert_ids=[expert.id for expert in experts],
        request_source=request.source if request else None,
        banned_for_another_reason=banned_for_another_reason(identity),
        confirmation_sent=bool(request and request.confirmationSentAt),
        request_completed=bool(request and request.completedAt),
    )


def banned_for_another_reason(identity: AuthUser | None) -> bool:
    """A Better Auth ban that closing the account did not set."""
    return bool(
        identity
        and identity.banned
        and identity.banReason != ACCOUNT_DELETED_BAN_REASON
    )


async def find_blockers(
    subject: Subject, *, closing: bool = False, allow_banned: bool = False
) -> list[Blocker]:
    """What stops the erasure. Only a closed account (one with a deletion
    request) is erased; `closing` says the caller closes it first, as the
    CLI does for a request made by email. An account banned for another
    reason is not closed that way unless `allow_banned`, as the Settings
    route refuses it too."""
    if refused := invalid_id(subject.user_id) or system_account(subject.user_id):
        return refused
    blockers: list[Blocker] = []
    if closing and subject.banned_for_another_reason and not allow_banned:
        blockers.append(
            Blocker(
                name="banned",
                count=1,
                resolution="This account is banned for another reason (abuse, "
                "say), and Settings would refuse to delete it. Check with "
                "whoever banned it. To close and erase it anyway, keeping "
                "that ban, pass --allow-banned.",
            )
        )
    if not (subject.has_request or closing):
        blockers.append(
            Blocker(
                name="not_closed",
                count=0,
                resolution="This account has no deletion request, so it is "
                "still open. For a request made by email, pass --close to close "
                "it the way Settings does, then erase it.",
            )
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
    if apps_in_use := await _oauth_apps_others_use(subject.user_id):
        blockers.append(
            Blocker(
                name="oauth_apps_in_use",
                count=apps_in_use,
                resolution="This account registered OAuth apps that other "
                "accounts have signed in to and still use. Erasing it deletes "
                "the apps and cuts those accounts off. They are not handed to "
                'the "deleted user" account: the developer still holds the '
                "client secret and controls the redirect URIs, and nobody here "
                "could manage the app. Move each app to another owner (set "
                "OAuthApplication.ownerId to their user ID) or, if it should "
                "stop working, disable it (isActive = false). Then run again.",
            )
        )
    return blockers


async def _oauth_apps_others_use(user_id: str) -> int:
    """Active OAuth apps the account owns that another account holds a live
    grant to. A disabled app already works for nobody (its tokens fail
    validation and cannot be refreshed), so it goes with the account."""
    live = {
        "userId": {"not": user_id},
        "revokedAt": None,
        "expiresAt": {"gt": datetime.now(UTC)},
    }
    return await OAuthApplication.prisma().count(
        where={
            "ownerId": user_id,
            "isActive": True,
            "OR": [
                {"RefreshTokens": {"some": live}},
                {"AccessTokens": {"some": live}},
            ],
        }
    )


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
            "belong to. It is never erased.",
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


async def _org_stripe_customers(
    user: User | None, orgs: list[Organization]
) -> list[str]:
    """The sole orgs' customers. The account's own is left out: closing the
    account (`cancel_all_subscriptions_now`) cancelled its subscriptions."""
    subscriptions = await OrganizationSubscription.prisma().find_many(
        where={"organizationId": {"in": [org.id for org in orgs]}}
    )
    ids = [org.stripeCustomerId for org in orgs]
    ids += [sub.stripeCustomerId for sub in subscriptions]
    own = user.stripeCustomerId if user else None
    return list(dict.fromkeys(i for i in ids if i and i != own))
