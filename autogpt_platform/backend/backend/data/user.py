import asyncio
import base64
import hashlib
import hmac
import logging
import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Optional, cast
from urllib.parse import quote_plus

from autogpt_libs.auth.models import DEFAULT_USER_ID
from fastapi import HTTPException
from prisma.enums import BriefingFrequency, SubscriptionTier
from prisma.errors import RecordNotFoundError, UniqueViolationError
from prisma.models import AuthAccount, AuthUser
from prisma.models import User as PrismaUser
from prisma.types import (
    JsonFilter,
    ProfileCreateInput,
    UserCreateInput,
    UserUpdateInput,
    UserWhereInput,
)
from pydantic import BaseModel, ConfigDict

from backend.data.db import prisma, query_raw_with_schema, transaction
from backend.data.model import (
    CREDENTIALS_ADAPTER,
    Credentials,
    User,
    UserIntegrations,
    UserMetadata,
)
from backend.data.notifications import (
    AudienceAction,
    NotificationPreference,
    NotificationPreferenceDTO,
)
from backend.data.org_migration import ensure_personal_org
from backend.data.posthog_lifecycle_sync import schedule_posthog_lifecycle_sync
from backend.data.subscription_trial import get_subscription_trial
from backend.notifications.queue import queue_audience_change
from backend.notifications.subscriber_fields import audience_event
from backend.util.cache import cached
from backend.util.encryption import JSONCryptor
from backend.util.exceptions import DatabaseError, NotFoundError
from backend.util.json import SafeJson
from backend.util.product_analytics import track_signup_completed
from backend.util.settings import Settings

if TYPE_CHECKING:
    from backend.integrations.credentials_store import IntegrationCredentialsStore

logger = logging.getLogger(__name__)
settings = Settings()

# Cache decorator alias for consistent user lookup caching
cache_user_lookup = cached(maxsize=1000, ttl_seconds=300, shared_cache=True)


class UserCreationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    user: User
    # True for the one call that set the account up: it created the ``User``
    # row, or bootstrapped the personal org for a row the auth hook inserted
    # bare at sign-up. Drives the sign-up conversion (via the route's
    # ``X-AutoGPT-User-Created`` header) and the PostHog lifecycle sync.
    was_created: bool


@cache_user_lookup
async def get_or_create_user(user_data: dict) -> User:
    return (await _get_or_create_user(user_data)).user


async def get_or_create_user_with_status(user_data: dict) -> UserCreationResult:
    return await _get_or_create_user(user_data)


async def _get_or_create_user(user_data: dict) -> UserCreationResult:
    try:
        user_id = user_data.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="User ID not found in token")

        user_email = user_data.get("email")
        if not user_email:
            raise HTTPException(status_code=401, detail="Email not found in token")

        user = await prisma.user.find_unique(where={"id": user_id})
        row_created = False
        if not user:
            try:
                user = await prisma.user.create(
                    data=UserCreateInput(
                        id=user_id,
                        email=user_email,
                        name=user_data.get("user_metadata", {}).get("name"),
                    )
                )
                row_created = True
            except UniqueViolationError:
                # A concurrent first request (the verify link opened in two
                # browsers at once) created it since the lookup. If it is the
                # email that clashes instead, there is still no row: re-raise.
                user = await prisma.user.find_unique(where={"id": user_id})
                if user is None:
                    raise

        # Ensure every user has a marketplace Profile (required to publish
        # agents). Best-effort: a failure must not block user resolution — the
        # user self-heals on their next request or via the profile settings page.
        try:
            await _ensure_user_profile(user.id, user.email)
        except Exception:
            logger.warning(
                "Failed to ensure marketplace profile for user %s",
                user.id,
                exc_info=True,
            )

        # Ensure every user owns a personal org + default team. Unlike the
        # Profile above this is NOT best-effort: without an org, every
        # org-scoped endpoint (save graph, chat, ...) fails with "No
        # organization context available", so a failure here must fail the
        # request loudly instead of returning a bricked account. Idempotent and
        # race-safe (see ensure_personal_org).
        org_created = await ensure_personal_org(user.id)

        # The auth hook inserts a bare row the moment the identity is created
        # (see provision-platform-user.ts), so the row alone does not mark a
        # new account: the personal org this call bootstraps does.
        was_created = row_created or org_created
        if was_created:
            track_signup_completed(
                user_id=user.id, signup_method=await _signup_method(user.id, user_data)
            )
            schedule_posthog_lifecycle_sync(user.id)

        return UserCreationResult(user=User.from_db(user), was_created=was_created)
    except Exception as e:
        # Identify by subject only. `user_data` is the decoded JWT (email,
        # name, role); this error is logged with exc_info on the auth
        # self-heal path, so interpolating it writes user PII into Sentry
        # event bodies.
        raise DatabaseError(
            f"Failed to get or create user {user_data.get('sub')}: {e}"
        ) from e


# Better Auth's providerId for an email + password account.
_BETTER_AUTH_EMAIL_PROVIDER = "credential"


async def _signup_method(user_id: str, user_data: dict) -> str | None:
    """The auth provider the account was created with (``email``, ``google``, ...).

    Better Auth's token carries no provider, so it is read from the user's
    first ``AuthAccount`` row, which Better Auth writes before it issues the
    token. A Supabase token carries it in ``app_metadata`` instead; that is
    the fallback for a user with no account row.
    """
    try:
        account = await AuthAccount.prisma().find_first(
            where={"userId": user_id}, order={"createdAt": "asc"}
        )
    except Exception:
        # Analytics only: a failed lookup must not fail the signup.
        logger.warning("Failed to read the auth account of %s", user_id, exc_info=True)
        account = None
    if account is not None and account.providerId:
        if account.providerId == _BETTER_AUTH_EMAIL_PROVIDER:
            return "email"
        return account.providerId
    return _legacy_signup_method(user_data)


def _legacy_signup_method(user_data: dict) -> str | None:
    """The provider from a Supabase token's ``app_metadata``."""
    app_metadata = user_data.get("app_metadata")
    if not isinstance(app_metadata, dict):
        return None
    provider = app_metadata.get("provider")
    return provider if isinstance(provider, str) and provider else None


# Word lists mirror the legacy generate_username() SQL function so that app-
# and DB-generated handles look consistent.
_PROFILE_USERNAME_ADJECTIVES = (
    "happy",
    "clever",
    "swift",
    "bright",
    "wise",
    "funny",
    "cool",
    "awesome",
    "amazing",
    "fantastic",
    "wonderful",
)
_PROFILE_USERNAME_ANIMALS = (
    "fox",
    "wolf",
    "bear",
    "eagle",
    "owl",
    "tiger",
    "lion",
    "elephant",
    "giraffe",
    "zebra",
)


async def _ensure_user_profile(user_id: str, email: Optional[str]) -> None:
    """Create a default marketplace Profile for *user_id* if none exists.

    Idempotent and race-safe. A UniqueViolationError has two possible sources:
    a concurrent request that already created this user's Profile (done), or a
    collision on the unique username with *another* user (retry with a fresh
    handle — otherwise the user would be left without a Profile).
    """
    if await prisma.profile.find_unique(where={"userId": user_id}):
        return

    name = (email or "").split("@", 1)[0] or "user"
    for _ in range(3):
        try:
            await prisma.profile.create(
                data=ProfileCreateInput(
                    userId=user_id,
                    name=name,
                    username=await _generate_profile_username(),
                    description="I'm new here",
                    links=[],
                    avatarUrl="",
                )
            )
            return
        except UniqueViolationError:
            if await prisma.profile.find_unique(where={"userId": user_id}):
                # Another in-flight request (or the legacy auth.users trigger)
                # created this user's Profile — nothing to do.
                logger.debug(
                    "Profile for user %s already created concurrently", user_id
                )
                return
            # The generated username collided with another user — loop and
            # retry with a fresh handle.
    logger.warning(
        "Failed to create a unique profile handle for user %s after retries",
        user_id,
    )


async def _generate_profile_username() -> str:
    """Generate a human-friendly profile handle, avoiding obvious collisions.

    The unique constraint on Profile.username is the real guarantee; this
    pre-check just avoids retrying a create in the common case. Falls back to a
    UUID-based handle if we can't find a free friendly name.
    """
    for _ in range(10):
        candidate = (
            f"{random.choice(_PROFILE_USERNAME_ADJECTIVES)}-"
            f"{random.choice(_PROFILE_USERNAME_ANIMALS)}-"
            f"{random.randint(10000, 99999)}"
        )
        if not await prisma.profile.find_unique(where={"username": candidate}):
            return candidate
    return f"user-{uuid.uuid4().hex[:12]}"


@cache_user_lookup
async def get_user_by_id(user_id: str) -> User:
    user = await prisma.user.find_unique(where={"id": user_id})
    if not user:
        raise ValueError(f"User not found with ID: {user_id}")
    return User.from_db(user)


async def get_user_subscription_tier(user_id: str) -> SubscriptionTier:
    """Read the authoritative tier without using the cached full-user lookup."""
    user = await prisma.user.find_unique(where={"id": user_id})
    if not user:
        raise ValueError(f"User not found with ID: {user_id}")
    if user.subscriptionTier == SubscriptionTier.TRIAL:
        trial = await get_subscription_trial(user_id)
        if trial is None or not trial.active:
            return SubscriptionTier.NO_TIER
        return SubscriptionTier(trial.offer.tier)
    return user.subscriptionTier or SubscriptionTier.NO_TIER


async def get_user_email_by_id(user_id: str) -> Optional[str]:
    try:
        user = await prisma.user.find_unique(where={"id": user_id})
        return user.email if user else None
    except Exception as e:
        raise DatabaseError(f"Failed to get user email for user {user_id}: {e}") from e


class AuthUserFlagFields(BaseModel):
    """Minimal AuthUser attributes used to build a LaunchDarkly context.

    A plain serializable shape (not an ``ldclient.Context``) so it can cross
    the DatabaseManager RPC boundary — feature-flag evaluation runs in
    Prisma-less workers (scheduler, copilot-executor) that reach the auth
    table via the RPC client rather than a locally-connected Prisma engine.
    """

    role: Optional[str] = None
    email: Optional[str] = None
    created_at: Optional[datetime] = None


async def get_auth_user_flag_fields(user_id: str) -> Optional[AuthUserFlagFields]:
    """Fetch the AuthUser fields used for LaunchDarkly targeting.

    Returns ``None`` when no auth row exists (e.g. mid auth-migration bridge
    window) so the caller can avoid caching a not-found as an anonymous
    context.
    """
    user = await AuthUser.prisma().find_unique(where={"id": user_id})
    if user is None:
        return None
    return AuthUserFlagFields(
        role=user.role,
        email=user.email,
        created_at=user.createdAt,
    )


class OrphanedAuthIdentity(BaseModel):
    """An auth identity (Better Auth user) with no platform ``User`` row."""

    id: str
    email: str
    name: Optional[str] = None
    createdAt: datetime
    # Set when a *different* platform User already owns this email, which the
    # unique index on ``User.email`` turns into an unprovisionable account.
    email_owner_id: Optional[str] = None

    @property
    def has_email_collision(self) -> bool:
        return self.email_owner_id is not None and self.email_owner_id != self.id


class OrphanedAuthIdentityReport(BaseModel):
    healed: list[str] = []
    collided: list[OrphanedAuthIdentity] = []
    failed: list[str] = []

    @property
    def is_clean(self) -> bool:
        return not (self.healed or self.collided or self.failed)


async def find_orphaned_auth_identities(
    older_than: datetime, limit: int = 100
) -> list[OrphanedAuthIdentity]:
    """Auth identities created before *older_than* that have no ``User`` row.

    Only verified identities, or ones with a session: an unverified sign-up
    gets its row when its link is opened. Identities whose email another
    platform User owns sort last, so they can't fill every batch.
    """
    rows = await query_raw_with_schema(
        # The owner lookup is case-insensitive on purpose: the auth migration
        # copied emails as stored, so a migrated identity can differ from its
        # platform row only by case, and missing that owner would heal a
        # duplicate account. It is a scalar subquery rather than a join so an
        # identity yields exactly one row even when several platform rows
        # carry case-variants of its email -- a join would return the identity
        # once per variant and let duplicates eat into the batch limit.
        'SELECT a.id, a.email, a.name, a."createdAt", '
        '(SELECT owner.id FROM {schema_prefix}"User" owner '
        "WHERE LOWER(owner.email) = LOWER(a.email) "
        'ORDER BY owner."createdAt" ASC LIMIT 1) AS email_owner_id '
        'FROM {schema_prefix}"UserAuthIdentity" a '
        'LEFT JOIN {schema_prefix}"User" u ON u.id = a.id '
        'WHERE u.id IS NULL AND a."createdAt" < $1::timestamptz '
        'AND (a."emailVerified" OR EXISTS (SELECT 1 FROM '
        '{schema_prefix}"UserAuthSession" s WHERE s."userId" = a.id)) '
        'ORDER BY EXISTS (SELECT 1 FROM {schema_prefix}"User" o '
        "WHERE LOWER(o.email) = LOWER(a.email)), "
        'a."createdAt" ASC '
        "LIMIT $2::int",
        older_than.isoformat(),
        limit,
    )
    return [OrphanedAuthIdentity(**row) for row in rows]


async def heal_orphaned_auth_identities(
    grace_secs: int = 300, limit: int = 100
) -> OrphanedAuthIdentityReport:
    """Provision a platform ``User`` for every orphaned auth identity.

    Runs the same provisioning as ``POST /auth/user`` (User + marketplace
    Profile + personal org) from the identity's own email, so a healed account
    is indistinguishable from one that signed up cleanly. Identities younger
    than *grace_secs* are left alone: their sign-up is still in flight.

    An identity whose email is already owned by a different platform User
    (compared case-insensitively) is reported, not healed -- the unique index
    makes it unprovisionable, and guessing which account the person meant is
    not this function's call.
    """
    older_than = datetime.now(timezone.utc) - timedelta(seconds=grace_secs)
    report = OrphanedAuthIdentityReport()
    for identity in await find_orphaned_auth_identities(older_than, limit):
        if identity.has_email_collision:
            report.collided.append(identity)
            continue
        try:
            result = await get_or_create_user_with_status(
                {
                    "sub": identity.id,
                    "email": identity.email,
                    "user_metadata": {"name": identity.name},
                }
            )
        except Exception:
            logger.error(
                f"Failed to heal orphaned auth identity {identity.id}", exc_info=True
            )
            report.failed.append(identity.id)
            continue
        # A sign-in that landed between the query and here provisioned it
        # already: nothing was broken, so nothing to page about.
        if result.was_created:
            report.healed.append(identity.id)
    return report


@cache_user_lookup
async def get_user_by_email(email: str) -> Optional[User]:
    try:
        user = await prisma.user.find_unique(where={"email": email})
        return User.from_db(user) if user else None
    except Exception as e:
        raise DatabaseError(f"Failed to get user by email {email}: {e}") from e


async def search_users(query: str, limit: int = 20) -> list[tuple[str, str | None]]:
    """Search users by partial email or name.

    Returns a list of ``(user_id, email)`` tuples, up to *limit* results.
    Searches the User table directly — no dependency on credit history.
    """
    query = query.strip()
    if not query or len(query) < 3:
        return []
    users = await prisma.user.find_many(
        where={
            "OR": [
                {"email": {"contains": query, "mode": "insensitive"}},
                {"name": {"contains": query, "mode": "insensitive"}},
            ],
        },
        take=limit,
        order={"email": "asc"},
    )
    return [(u.id, u.email) for u in users]


async def update_user_email(user_id: str, email: str):
    try:
        # Get old email first for cache invalidation
        old_user = await prisma.user.find_unique(where={"id": user_id})
        old_email = old_user.email if old_user else None

        await prisma.user.update(where={"id": user_id}, data={"email": email})

        # Selectively invalidate only the specific user entries
        get_user_by_id.cache_delete(user_id)
        if old_email:
            get_user_by_email.cache_delete(old_email)
        get_user_by_email.cache_delete(email)
    except Exception as e:
        raise DatabaseError(
            f"Failed to update user email for user {user_id}: {e}"
        ) from e


async def create_default_user() -> Optional[User]:
    user = await prisma.user.find_unique(where={"id": DEFAULT_USER_ID})
    if not user:
        user = await prisma.user.create(
            data=UserCreateInput(
                id=DEFAULT_USER_ID,
                email="default@example.com",
                name="Default User",
            )
        )
    return User.from_db(user)


async def get_user_integrations(user_id: str) -> UserIntegrations:
    user = await PrismaUser.prisma().find_unique_or_raise(
        where={"id": user_id},
    )

    encrypted_integrations = user.integrations
    if not encrypted_integrations:
        return UserIntegrations()
    else:
        return UserIntegrations.model_validate(
            JSONCryptor().decrypt(encrypted_integrations)
        )


async def update_user_integrations(user_id: str, data: UserIntegrations):
    encrypted_data = JSONCryptor().encrypt(data.model_dump(exclude_none=True))
    await PrismaUser.prisma().update(
        where={"id": user_id},
        data={"integrations": encrypted_data},
    )
    # Invalidate cache for this user
    get_user_by_id.cache_delete(user_id)


async def get_user_credentials(user_id: str) -> list[Credentials]:
    """Read the user's credentials from the IntegrationCredential table.

    Source of truth post blob→table migration (the UserIntegrations blob
    is retained only as a rollback artifact). Returns USER-scoped active
    rows; TEAM/ORG-scoped credentials are resolved separately via
    ``backend.integrations.scoped_credentials``.
    """
    rows = await prisma.integrationcredential.find_many(
        where={"ownerType": "USER", "ownerId": user_id, "status": "active"},
        order={"createdAt": "asc"},
    )
    cryptor = JSONCryptor()
    credentials: list[Credentials] = []
    for row in rows:
        try:
            credentials.append(
                CREDENTIALS_ADAPTER.validate_python(
                    cryptor.decrypt(row.encryptedPayload)
                )
            )
        except Exception:
            logger.error(
                f"Corrupt credential row {row.id} for user {user_id}; skipping",
                exc_info=True,
            )
    return credentials


async def set_user_credentials(user_id: str, credentials: list[Credentials]) -> None:
    """Full-list replace of the user's USER-scoped credential rows.

    Mirrors the old blob replace semantics the credential store is built
    around: rows missing from ``credentials`` are revoked (soft delete),
    new ids are created, existing ids get their payload refreshed
    (OAuth token rotation runs through here constantly).
    """
    cryptor = JSONCryptor()
    existing_rows = await prisma.integrationcredential.find_many(
        where={"ownerType": "USER", "ownerId": user_id},
    )
    existing_by_id = {row.id: row for row in existing_rows}
    incoming_ids = {c.id for c in credentials}

    org_id: str | None = None
    for cred in credentials:
        row = existing_by_id.get(cred.id)
        encrypted = cryptor.encrypt(cred.model_dump())
        if row is not None:
            await prisma.integrationcredential.update(
                where={"id": cred.id},
                data={
                    "encryptedPayload": encrypted,
                    "displayName": cred.title or cred.provider,
                    "status": "active",
                },
            )
            continue
        if org_id is None:
            org_row = await prisma.organization.find_first(
                where={
                    "isPersonal": True,
                    "Members": {"some": {"userId": user_id, "isOwner": True}},
                }
            )
            if org_row is None:
                raise DatabaseError(
                    f"Cannot store credentials for user {user_id}: "
                    "personal org not bootstrapped"
                )
            org_id = org_row.id
        await prisma.integrationcredential.create(
            data={
                "id": cred.id,
                "organizationId": org_id,
                "ownerType": "USER",
                "ownerId": user_id,
                "provider": cred.provider,
                "credentialType": cred.type,
                "displayName": cred.title or cred.provider,
                "encryptedPayload": encrypted,
                "createdByUserId": user_id,
            }
        )

    for row in existing_rows:
        if row.id not in incoming_ids and row.status == "active":
            await prisma.integrationcredential.update(
                where={"id": row.id},
                data={"status": "revoked"},
            )


async def migrate_and_encrypt_user_integrations():
    """Migrate integration credentials and OAuth states from metadata to integrations column."""
    users = await PrismaUser.prisma().find_many(
        where={
            "metadata": cast(
                JsonFilter,
                {
                    "path": ["integration_credentials"],
                    "not": SafeJson(
                        {"a": "yolo"}
                    ),  # bogus value works to check if key exists
                },
            )
        }
    )
    logger.info(f"Migrating integration credentials for {len(users)} users")

    for user in users:
        raw_metadata = cast(dict, user.metadata)
        metadata = UserMetadata.model_validate(raw_metadata)

        # Get existing integrations data
        integrations = await get_user_integrations(user_id=user.id)

        # Copy credentials and oauth states from metadata if they exist
        if metadata.integration_credentials and not integrations.credentials:
            integrations.credentials = metadata.integration_credentials
        if metadata.integration_oauth_states:
            integrations.oauth_states = metadata.integration_oauth_states

        # Save to integrations column
        await update_user_integrations(user_id=user.id, data=integrations)

        # Remove from metadata
        raw_metadata.pop("integration_credentials", None)
        raw_metadata.pop("integration_oauth_states", None)

        # Update metadata without integration data
        await PrismaUser.prisma().update(
            where={"id": user.id},
            data={"metadata": SafeJson(raw_metadata)},
        )


async def get_active_user_ids_in_timerange(start_time: str, end_time: str) -> list[str]:
    try:
        users = await PrismaUser.prisma().find_many(
            where={
                "AgentGraphExecutions": {
                    "some": {
                        "createdAt": {
                            "gte": datetime.fromisoformat(start_time),
                            "lte": datetime.fromisoformat(end_time),
                        }
                    }
                }
            },
        )
        return [user.id for user in users]

    except Exception as e:
        raise DatabaseError(
            f"Failed to get active user ids in timerange {start_time} to {end_time}: {e}"
        ) from e


async def get_active_users_ids() -> list[str]:
    user_ids = await get_active_user_ids_in_timerange(
        (datetime.now() - timedelta(days=30)).isoformat(),
        datetime.now().isoformat(),
    )
    return user_ids


async def get_user_notification_preference(user_id: str) -> NotificationPreference:
    """The volume knob: a Briefing frequency plus two switches. Billing and
    account messages are service mail and are not gated by any of it."""
    try:
        user = await PrismaUser.prisma().find_unique_or_raise(where={"id": user_id})
        return _preference_from_user(user)
    except Exception as e:
        raise DatabaseError(
            f"Failed to get user notification preference for user {user_id}: {e}"
        ) from e


async def update_user_notification_preference(
    user_id: str, data: NotificationPreferenceDTO
) -> NotificationPreference:
    try:
        update_data: UserUpdateInput = {
            "briefingFrequency": data.briefing_frequency,
            "alertsEnabled": data.alerts_enabled,
            "notifyOnStoreVerdict": data.store_verdicts_enabled,
        }
        if data.email:
            update_data["email"] = data.email
        # `is not None`, not truthiness: 0 is the documented "send nothing"
        # value that one-click unsubscribe writes, and a falsy check silently
        # kept the previous limit instead.
        if data.daily_limit is not None:
            update_data["maxEmailsPerDay"] = data.daily_limit

        user = await PrismaUser.prisma().update(where={"id": user_id}, data=update_data)
        if not user:
            raise ValueError(f"User not found with ID: {user_id}")

        # Invalidate cache for this user since notification preferences are
        # part of user data
        get_user_by_id.cache_delete(user_id)
        return _preference_from_user(user)
    except Exception as e:
        raise DatabaseError(
            f"Failed to update user notification preference for user {user_id}: {e}"
        ) from e


def _preference_from_user(user: PrismaUser) -> NotificationPreference:
    return NotificationPreference(
        user_id=user.id,
        email=user.email,
        briefing_frequency=BriefingFrequency(user.briefingFrequency),
        alerts_enabled=user.alertsEnabled,
        store_verdicts_enabled=user.notifyOnStoreVerdict,
        # Not `or 3`: the column is non-nullable, so the only value that
        # coalesce would catch is 0 — which is precisely what a one-click
        # unsubscribe sets, and means "send nothing".
        daily_limit=user.maxEmailsPerDay,
    )


async def set_user_email_verification(user_id: str, verified: bool) -> None:
    """Set the email verification status for a user."""
    try:
        await PrismaUser.prisma().update(
            where={"id": user_id},
            data={"emailVerified": verified},
        )
        # Invalidate cache for this user
        get_user_by_id.cache_delete(user_id)
    except Exception as e:
        raise DatabaseError(
            f"Failed to set email verification status for user {user_id}: {e}"
        ) from e


async def disable_all_user_notifications(user_id: str) -> None:
    """Turn the volume knob all the way down.

    Used when a user's email bounces or is marked inactive, so we stop trying
    to reach an address that cannot receive mail.
    """
    try:
        await PrismaUser.prisma().update(
            where={"id": user_id},
            data={
                "briefingFrequency": BriefingFrequency.OFF,
                "alertsEnabled": False,
                "notifyOnStoreVerdict": False,
            },
        )
        # Invalidate cache for this user
        get_user_by_id.cache_delete(user_id)
        logger.info(f"Disabled all notification preferences for user {user_id}")
    except Exception as e:
        raise DatabaseError(
            f"Failed to disable notifications for user {user_id}: {e}"
        ) from e


async def get_user_email_verification(user_id: str) -> bool:
    """Get the email verification status for a user."""
    try:
        user = await PrismaUser.prisma().find_unique_or_raise(
            where={"id": user_id},
        )
        return user.emailVerified
    except Exception as e:
        raise DatabaseError(
            f"Failed to get email verification status for user {user_id}: {e}"
        ) from e


def generate_unsubscribe_link(user_id: str) -> str:
    """Generate a link to unsubscribe from all notifications"""
    # Create an HMAC using a secret key
    secret_key = settings.secrets.unsubscribe_secret_key
    signature = hmac.new(
        secret_key.encode("utf-8"), user_id.encode("utf-8"), hashlib.sha256
    ).digest()

    # Create a token that combines the user_id and signature
    token = base64.urlsafe_b64encode(
        f"{user_id}:{signature.hex()}".encode("utf-8")
    ).decode("utf-8")
    logger.info(f"Generating unsubscribe link for user {user_id}")

    base_url = settings.config.platform_base_url
    return f"{base_url}/api/email/unsubscribe?token={quote_plus(token)}"


async def unsubscribe_user_by_token(token: str) -> None:
    """Unsubscribe a user from all notifications using the token"""
    try:
        # Decode the token
        decoded = base64.urlsafe_b64decode(token).decode("utf-8")
        user_id, received_signature_hex = decoded.split(":", 1)

        # Verify the signature
        secret_key = settings.secrets.unsubscribe_secret_key
        expected_signature = hmac.new(
            secret_key.encode("utf-8"), user_id.encode("utf-8"), hashlib.sha256
        ).digest()

        if not hmac.compare_digest(expected_signature.hex(), received_signature_hex):
            raise ValueError("Invalid token signature")

        user = await get_user_by_id(user_id)
        # One-click unsubscribe turns everything off, including store
        # verdicts — this is the trapdoor, and the volume knob in the Briefing
        # footer is what most people should be using instead.
        await update_user_notification_preference(
            user.id,
            NotificationPreferenceDTO(
                email=user.email,
                briefing_frequency=BriefingFrequency.OFF,
                alerts_enabled=False,
                store_verdicts_enabled=False,
                daily_limit=0,
            ),
        )
    except Exception as e:
        raise DatabaseError(f"Failed to unsubscribe user by token {token}: {e}") from e


async def cleanup_user_managed_credentials(
    user_id: str,
    store: Optional["IntegrationCredentialsStore"] = None,
) -> None:
    """Revoke all externally-provisioned managed credentials for *user_id*.

    Call this before deleting a user account so that external resources
    (e.g. AgentMail pods, pod-scoped API keys) are properly cleaned up.
    The credential rows themselves are cascade-deleted with the User row.

    Pass an existing *store* for testability; when omitted a fresh instance
    is created.
    """
    from backend.integrations.credentials_store import IntegrationCredentialsStore
    from backend.integrations.managed_credentials import cleanup_managed_credentials

    if store is None:
        store = IntegrationCredentialsStore()
    await cleanup_managed_credentials(user_id, store)


# Strong refs to fire-and-forget tasks — the event loop only keeps weak
# references, so an unretained task can be GC'd mid-flight and its
# exception is never observed. Same pattern as
# ``backend/copilot/chat_session_embeddings.py``.
_background_tasks: set[asyncio.Task] = set()


def _on_background_task_done(task: asyncio.Task) -> None:
    _background_tasks.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.warning("Background task %s failed", task.get_name(), exc_info=exc)


async def update_user_timezone(user_id: str, timezone: str) -> User:
    """Update a user's timezone setting."""
    try:
        user = await PrismaUser.prisma().update(
            where={"id": user_id},
            data={"timezone": timezone},
        )
        if not user:
            raise ValueError(f"User not found with ID: {user_id}")

        # Invalidate user caches so subsequent reads see the new timezone.
        # get_user_by_id and get_user_by_email are keyed by a single value
        # and can be deleted surgically; get_or_create_user is keyed by the
        # JWT-payload dict so we can't delete a single entry — clear it
        # entirely.
        get_user_by_id.cache_delete(user_id)
        if user.email:
            get_user_by_email.cache_delete(user.email)
        get_or_create_user.cache_clear()

        # Dream-system schedules are bound to the timezone at job-creation
        # time; without an eager re-register they'd keep firing at the old
        # local time. Fire-and-forget so this profile update returns
        # immediately — the helper's lazy drift-detection path (via the
        # Redis dedup key value) is the durable backstop if this
        # fails or the user doesn't trigger a memory write within the
        # 7-day key TTL.
        try:
            from backend.copilot.dream.scheduling import ensure_dream_system_scheduled

            task = asyncio.create_task(
                ensure_dream_system_scheduled(user_id, force_refresh=True),
                name=f"tz-reregister-{user_id[:12]}",
            )
            _background_tasks.add(task)
            task.add_done_callback(_on_background_task_done)
        except Exception:
            logger.warning(
                "Failed to spawn dream-system re-register after timezone "
                "update for user %s — lazy drift detection will catch it",
                user_id[:12],
                exc_info=True,
            )

        # Same rationale for the morning-briefing cron: clear the stored
        # marker first, since its mere presence short-circuits the helper,
        # then re-ensure so the profile change takes effect immediately
        # instead of waiting out the marker's TTL. Fire-and-forget like the dream task
        # above so the profile update doesn't block on Redis/scheduler
        # I/O, and guard the import so a failure here can't surface as a
        # false "failed to update timezone" after the row committed.
        try:
            from backend.copilot.briefing.scheduling import (
                clear_briefing_registration_marker,
                ensure_morning_briefing_scheduled,
            )

            async def _reregister_briefing() -> None:
                await clear_briefing_registration_marker(user_id)
                await ensure_morning_briefing_scheduled(user_id)

            task = asyncio.create_task(
                _reregister_briefing(),
                name=f"briefing-tz-reregister-{user_id[:12]}",
            )
            _background_tasks.add(task)
            task.add_done_callback(_on_background_task_done)
        except Exception:
            logger.warning(
                "Failed to spawn morning-briefing re-register after timezone "
                "update for user %s — lazy drift detection will catch it",
                user_id[:12],
                exc_info=True,
            )

        return User.from_db(user)
    except Exception as e:
        raise DatabaseError(f"Failed to update timezone for user {user_id}: {e}") from e


MARKETING_OPT_OUT_SOURCE_SIGNUP = "signup"
MARKETING_OPT_OUT_SOURCE_EMAIL_UNSUBSCRIBE = "email_unsubscribe"


async def record_signup_consent(
    user_id: str, terms_version: str, marketing_opt_out: bool
) -> User:
    """Record what the user agreed to on the signup page.

    Idempotent, so a retried call is harmless: the terms are stamped again only
    for a newer version (an older one never replaces a newer acceptance), and
    an opt-out keeps its first date and source, including one recorded after
    this call read the row (a concurrent call, or an unsubscribe). Both are
    written in one transaction, so a failure leaves neither half behind.
    `marketing_opt_out=False` changes nothing: this can refuse marketing but
    never take a refusal back, which is a settings action.

    A refusal this call records is passed on to MailerLite, so someone who is
    already a subscriber there stops getting marketing email too.

    Returns the row as read after any write, so it shows whichever refusal won.
    """
    try:
        current = await PrismaUser.prisma().find_unique(where={"id": user_id})
        if current is None:
            raise NotFoundError(f"User not found with ID: {user_id}")

        # Versions are YYYY-MM or YYYY-MM-DD, so they order as strings.
        stamp_terms = (
            current.termsAcceptedAt is None
            or current.termsVersion is None
            or terms_version > current.termsVersion
        )
        opt_out = marketing_opt_out and current.marketingOptOutAt is None
        if not stamp_terms and not opt_out:
            return User.from_db(current)

        terms_stamped, opted_out = await _write_signup_consent(
            user_id, terms_version if stamp_terms else None, opt_out
        )
        if terms_stamped or opted_out:
            # Same invalidation as update_user_timezone: the MailerLite gate
            # reads the opt-out through get_user_by_id, so a stale cached user
            # would let a checkout opened right after signup through.
            _invalidate_user_caches(user_id, current.email)
        if opted_out and current.email:
            await _unsubscribe_from_marketing(user_id, current.email)

        user = await PrismaUser.prisma().find_unique(where={"id": user_id})
        if user is None:
            raise NotFoundError(f"User not found with ID: {user_id}")
        return User.from_db(user)
    except NotFoundError:
        raise
    except RecordNotFoundError as e:
        # The account was deleted between the read and the write. Prisma's
        # `update` returns None for that, but a write that requires the row
        # can raise it instead; either way it is a 404, not a database error.
        raise NotFoundError(f"User not found with ID: {user_id}") from e
    except Exception as e:
        raise DatabaseError(
            f"Failed to record signup consent for user {user_id}: {e}"
        ) from e


async def _write_signup_consent(
    user_id: str, terms_version: str | None, opt_out: bool
) -> tuple[bool, bool]:
    """Stamp the terms (unless `terms_version` is None) and, when `opt_out`,
    the opt-out, in one transaction. Returns whether each reached the row.

    The opt-out is conditional on none being stored, so a refusal recorded
    since the caller read the row keeps its own date and source.
    """
    now = datetime.now(timezone.utc)
    terms_stamped = opted_out = False
    async with transaction() as tx:
        if terms_version is not None:
            stamped = await PrismaUser.prisma(tx).update(
                where={"id": user_id},
                data={"termsAcceptedAt": now, "termsVersion": terms_version},
            )
            terms_stamped = stamped is not None
        if opt_out:
            opted_out = (
                await PrismaUser.prisma(tx).update_many(
                    where={"id": user_id, "marketingOptOutAt": None},
                    data={
                        "marketingOptOutAt": now,
                        "marketingOptOutSource": MARKETING_OPT_OUT_SOURCE_SIGNUP,
                    },
                )
                > 0
            )
    return terms_stamped, opted_out


async def _unsubscribe_from_marketing(user_id: str, email: str) -> None:
    """Queue the change that marks an existing MailerLite subscriber
    unsubscribed (it never creates one). The refusal is already stored, so a
    failure here is logged rather than raised: the consumer re-reads it before
    every other MailerLite write, so only the unsubscribe itself is lost."""
    try:
        event = audience_event(AudienceAction.UNSUBSCRIBE, email, user_id)
        if event is None:
            return
        result = await queue_audience_change(event)
        if not result.success:
            logger.error(
                f"Could not queue the MailerLite unsubscribe for user {user_id}: "
                f"{result.message}"
            )
    except Exception:
        logger.exception(
            f"Could not queue the MailerLite unsubscribe for user {user_id}"
        )


def _escape_like(value: str) -> str:
    """`value` as a literal LIKE/ILIKE pattern (backslash is the default escape)."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


async def record_marketing_opt_out_by_email(email: str, source: str) -> str | None:
    """Record that the account at `email` refused marketing, from `source`.

    For refusals made outside the app, such as an unsubscribe in MailerLite.
    The address is matched exactly, then case-insensitively. First refusal
    wins: an account already opted out keeps its date and source, so a
    redelivered call changes nothing. Returns the account's ID only when this
    call recorded the refusal; None for an unknown address or a repeat.
    """
    try:
        row = await PrismaUser.prisma().find_unique(where={"email": email})
        if row is None:
            # Prisma's case-insensitive `equals` is a Postgres ILIKE, where `_`
            # and `%` are wildcards: escape them so only this address matches,
            # and act only when exactly one account does.
            matches = await PrismaUser.prisma().find_many(
                where={"email": {"equals": _escape_like(email), "mode": "insensitive"}},
                take=2,
            )
            row = matches[0] if len(matches) == 1 else None
        if row is None or row.marketingOptOutAt is not None:
            return None
        written = await PrismaUser.prisma().update_many(
            where={"id": row.id, "marketingOptOutAt": None},
            data={
                "marketingOptOutAt": datetime.now(timezone.utc),
                "marketingOptOutSource": source,
            },
        )
        if not written:
            return None
        _invalidate_user_caches(row.id, row.email)
        return row.id
    except Exception as e:
        # The address stays out of the message: it would reach the logs.
        raise DatabaseError(f"Failed to record a marketing opt-out: {e}") from e


async def is_marketing_opted_out(user_id: str) -> bool:
    """Whether the account refused marketing, read from the database rather
    than the user cache, for the last check before a MailerLite write. An
    account that no longer exists counts as opted out: nothing about it should
    reach MailerLite either."""
    try:
        row = await PrismaUser.prisma().find_unique(where={"id": user_id})
    except Exception as e:
        raise DatabaseError(
            f"Failed to read the marketing opt-out for user {user_id}: {e}"
        ) from e
    return row is None or row.marketingOptOutAt is not None


def _invalidate_user_caches(user_id: str, email: str | None) -> None:
    get_user_by_id.cache_delete(user_id)
    if email:
        get_user_by_email.cache_delete(email)
    get_or_create_user.cache_clear()


class BriefingCandidate(BaseModel):
    """The fields the briefing pass needs to decide whether a user is due.

    A narrow model rather than the full `User`, because this crosses the
    DatabaseManager RPC boundary once per page of candidates.
    """

    id: str
    email: str
    timezone: str
    briefing_frequency: BriefingFrequency
    last_briefing_at: datetime | None
    alerts_enabled: bool


async def get_briefing_candidates(
    timezones: list[str], after_id: str | None, limit: int
) -> list[BriefingCandidate]:
    """One page of users for whom it is currently the local briefing hour.

    Keyset-paged on `id` so the caller can walk the whole set: a single capped
    read would strand every user past the cap behind the same first page,
    because the ordering never changes and the filter does not exclude users
    already briefed.
    """
    try:
        where: UserWhereInput = {
            "briefingFrequency": {"not": BriefingFrequency.OFF},
            "timezone": {"in": timezones},
        }
        if after_id:
            where["id"] = {"gt": after_id}
        rows = await prisma.user.find_many(where=where, take=limit, order={"id": "asc"})
        return [
            BriefingCandidate(
                id=row.id,
                email=row.email,
                timezone=row.timezone,
                briefing_frequency=BriefingFrequency(row.briefingFrequency),
                last_briefing_at=row.lastBriefingAt,
                alerts_enabled=row.alertsEnabled,
            )
            for row in rows
        ]
    except Exception as e:
        raise DatabaseError(f"Failed to list briefing candidates: {e}") from e


async def get_briefing_candidate(user_id: str) -> BriefingCandidate | None:
    """One user's briefing settings, for work picked up off the queue.

    The pass publishes only a user id, so the consumer re-reads rather than
    trusting settings captured a tick earlier — a user who switched the
    briefing off in between must not receive one.
    """
    try:
        row = await prisma.user.find_unique(where={"id": user_id})
        if row is None:
            return None
        return BriefingCandidate(
            id=row.id,
            email=row.email,
            timezone=row.timezone,
            briefing_frequency=BriefingFrequency(row.briefingFrequency),
            last_briefing_at=row.lastBriefingAt,
            alerts_enabled=row.alertsEnabled,
        )
    except Exception as e:
        raise DatabaseError(f"Failed to load briefing candidate {user_id}: {e}") from e


async def set_last_briefing_at(user_id: str, sent_at: datetime) -> None:
    """Advance the cadence clock, once the briefing is safely on the queue."""
    try:
        await prisma.user.update(
            where={"id": user_id}, data={"lastBriefingAt": sent_at}
        )
    except Exception as e:
        raise DatabaseError(
            f"Failed to record briefing time for user {user_id}: {e}"
        ) from e


class BillingEmailRecipient(BaseModel):
    """The fields the billing emails need about a customer.

    A narrow model rather than the Prisma `User`, because this crosses the
    DatabaseManager RPC boundary: the lifecycle handlers run in the REST API
    when a Stripe webhook arrives, and in the notification service when the
    welcome is picked up off the work queue. That second process has no Prisma
    connection.
    """

    id: str
    email: str
    name: str | None = None
    welcome_email_sent_at: datetime | None = None
    # Set when the customer refused marketing: billing emails still go out,
    # MailerLite never hears of them (notifications/consent.py).
    marketing_opt_out_at: datetime | None = None


async def get_billing_email_recipient(
    stripe_customer_id: str,
) -> BillingEmailRecipient | None:
    """The account behind a Stripe customer, or None for a deleted or unknown
    one — better to skip than to email into the void."""
    try:
        row = await prisma.user.find_first(
            where={"stripeCustomerId": stripe_customer_id}
        )
        if row is None:
            return None
        return BillingEmailRecipient(
            id=row.id,
            email=row.email,
            name=row.name,
            welcome_email_sent_at=row.welcomeEmailSentAt,
            marketing_opt_out_at=row.marketingOptOutAt,
        )
    except Exception as e:
        raise DatabaseError(
            f"Failed to look up the account for Stripe customer "
            f"{stripe_customer_id}: {e}"
        ) from e


async def claim_welcome_email(user_id: str) -> bool:
    """Take the one-shot welcome claim. True only for the caller that set it.

    Conditional on the column still being null, so two webhook deliveries
    racing each other cannot both send.
    """
    try:
        claimed = await prisma.user.update_many(
            where={"id": user_id, "welcomeEmailSentAt": None},
            data={"welcomeEmailSentAt": datetime.now(tz=timezone.utc)},
        )
        return claimed > 0
    except Exception as e:
        raise DatabaseError(
            f"Failed to claim the welcome email for user {user_id}: {e}"
        ) from e


async def release_welcome_email(user_id: str) -> None:
    """Give the claim back when the send fails.

    This is a column, not a key with a TTL: left set after a failed publish it
    marks the customer welcomed forever, and every retry takes the
    returning-customer branch instead.
    """
    try:
        await prisma.user.update_many(
            where={"id": user_id}, data={"welcomeEmailSentAt": None}
        )
    except Exception:
        logger.warning(
            "Could not release the welcome claim for user %s; they will not be "
            "greeted on a retry",
            user_id,
            exc_info=True,
        )


# The volume-knob choices a Briefing footer can carry.
FOOTER_CHOICES = frozenset({"daily", "weekly", "monthly", "alerts", "off"})


def generate_preference_link(user_id: str, choice: str) -> str:
    """A footer link that changes one setting in one click.

    The choice is bound to the recipient by an HMAC, for the same reason the
    unsubscribe link is: the settings page applies it on arrival, so a bare
    `?f=off` would let any third party change an authenticated reader's
    preferences simply by getting them to follow a link. Signing it means only
    a link we generated, for that person, for that choice, is honoured.
    """
    if choice not in FOOTER_CHOICES:
        raise ValueError(f"Unknown footer choice: {choice}")
    token = _sign_preference_choice(user_id, choice)
    base_url = settings.config.frontend_base_url or settings.config.platform_base_url
    return f"{base_url}/settings/account?f={choice}&t={quote_plus(token)}"


def verify_preference_token(token: str, choice: str) -> str | None:
    """The user id this token authorises for this choice, or None.

    Returns None rather than raising: a bad token means the link is ignored and
    the page loads normally, which is the right outcome for a stale forward or
    a tampered URL.
    """
    try:
        decoded = base64.urlsafe_b64decode(token).decode("utf-8")
        user_id, received = decoded.rsplit(":", 1)
    except Exception:
        return None
    expected = _sign_preference_choice(user_id, choice)
    try:
        expected_sig = (
            base64.urlsafe_b64decode(expected).decode("utf-8").rsplit(":", 1)[1]
        )
    except Exception:
        return None
    if not hmac.compare_digest(expected_sig, received):
        return None
    return user_id


def _sign_preference_choice(user_id: str, choice: str) -> str:
    secret_key = settings.secrets.unsubscribe_secret_key
    # The choice is inside the signed payload, so a token minted for "daily"
    # cannot be replayed as "off".
    signature = hmac.new(
        secret_key.encode("utf-8"),
        f"{user_id}:{choice}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return base64.urlsafe_b64encode(
        f"{user_id}:{signature.hex()}".encode("utf-8")
    ).decode("utf-8")


async def get_user_default_chat_route(
    user_id: str,
) -> tuple[Optional[str], Optional[str]]:
    """Read the saved default chat transport exactly as stored.

    Returned verbatim rather than validated here: what a provider string
    means belongs to the copilot layer, and the data layer must not import
    it (``copilot.rate_limit`` already imports ``data.user``).

    A user who has authenticated but has no platform row yet — the row is
    created by ``POST /auth/user``, which the frontend calls separately after
    sign-in — reads as "nothing saved". Raising here would take transport
    discovery and session creation down for that window, neither of which
    touched the user table before this setting existed.
    """
    try:
        user = await get_user_by_id(user_id)
    except ValueError:
        return None, None
    # Shared-cache entries can outlive a rolling deploy. An object pickled by
    # the previous version has no attributes for fields introduced here, so
    # read them defensively until that cache entry expires.
    return (
        getattr(user, "default_chat_auth_provider", None),
        getattr(user, "default_chat_credential_id", None),
    )


async def set_user_default_chat_route(
    user_id: str,
    auth_provider: Optional[str],
    credential_id: Optional[str],
) -> None:
    """Save (or, with ``auth_provider=None``, clear) the default chat transport."""
    try:
        prisma_user = PrismaUser.prisma()
        user = await prisma_user.find_unique(where={"id": user_id})
        if user is None:
            raise NotFoundError(f"User {user_id} not found")

        updated_count = await prisma_user.update_many(
            where={"id": user_id},
            data={
                "defaultChatAuthProvider": auth_provider,
                "defaultChatCredentialId": (
                    credential_id if auth_provider is not None else None
                ),
            },
        )
        if updated_count == 0:
            raise NotFoundError(f"User {user_id} not found")

        # Same cache invalidation as update_user_timezone — the route is read
        # through the cached full-user lookup on every unrouted session create.
        get_user_by_id.cache_delete(user_id)
        if user.email:
            get_user_by_email.cache_delete(user.email)
        get_or_create_user.cache_clear()
    except NotFoundError:
        raise
    except Exception as e:
        raise DatabaseError(
            f"Failed to update default chat route for user {user_id}: {e}"
        ) from e
