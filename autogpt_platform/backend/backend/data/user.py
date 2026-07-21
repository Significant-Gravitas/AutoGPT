import asyncio
import base64
import hashlib
import hmac
import logging
import random
import uuid
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional, cast
from urllib.parse import quote_plus

from autogpt_libs.auth.models import DEFAULT_USER_ID
from fastapi import HTTPException
from prisma.enums import NotificationType
from prisma.errors import UniqueViolationError
from prisma.models import User as PrismaUser
from prisma.types import (
    JsonFilter,
    ProfileCreateInput,
    UserCreateInput,
    UserUpdateInput,
)

from backend.data.db import prisma
from backend.data.model import (
    CREDENTIALS_ADAPTER,
    Credentials,
    User,
    UserIntegrations,
    UserMetadata,
)
from backend.data.notifications import NotificationPreference, NotificationPreferenceDTO
from backend.data.org_migration import ensure_personal_org
from backend.util.cache import cached
from backend.util.encryption import JSONCryptor
from backend.util.exceptions import DatabaseError
from backend.util.json import SafeJson
from backend.util.settings import Settings

if TYPE_CHECKING:
    from backend.integrations.credentials_store import IntegrationCredentialsStore

logger = logging.getLogger(__name__)
settings = Settings()

# Cache decorator alias for consistent user lookup caching
cache_user_lookup = cached(maxsize=1000, ttl_seconds=300, shared_cache=True)


@cache_user_lookup
async def get_or_create_user(user_data: dict) -> User:
    try:
        user_id = user_data.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="User ID not found in token")

        user_email = user_data.get("email")
        if not user_email:
            raise HTTPException(status_code=401, detail="Email not found in token")

        user = await prisma.user.find_unique(where={"id": user_id})
        if not user:
            user = await prisma.user.create(
                data=UserCreateInput(
                    id=user_id,
                    email=user_email,
                    name=user_data.get("user_metadata", {}).get("name"),
                )
            )

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
        await ensure_personal_org(user.id)

        return User.from_db(user)
    except Exception as e:
        raise DatabaseError(f"Failed to get or create user {user_data}: {e}") from e


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


async def get_user_email_by_id(user_id: str) -> Optional[str]:
    try:
        user = await prisma.user.find_unique(where={"id": user_id})
        return user.email if user else None
    except Exception as e:
        raise DatabaseError(f"Failed to get user email for user {user_id}: {e}") from e


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
    try:
        user = await PrismaUser.prisma().find_unique_or_raise(
            where={"id": user_id},
        )

        # enable notifications by default if user has no notification preference (shouldn't ever happen though)
        preferences: dict[NotificationType, bool] = {
            NotificationType.AGENT_RUN: user.notifyOnAgentRun or False,
            NotificationType.ZERO_BALANCE: user.notifyOnZeroBalance or False,
            NotificationType.LOW_BALANCE: user.notifyOnLowBalance or False,
            NotificationType.BLOCK_EXECUTION_FAILED: user.notifyOnBlockExecutionFailed
            or False,
            NotificationType.CONTINUOUS_AGENT_ERROR: user.notifyOnContinuousAgentError
            or False,
            NotificationType.DAILY_SUMMARY: user.notifyOnDailySummary or False,
            NotificationType.WEEKLY_SUMMARY: user.notifyOnWeeklySummary or False,
            NotificationType.MONTHLY_SUMMARY: user.notifyOnMonthlySummary or False,
            NotificationType.AGENT_APPROVED: user.notifyOnAgentApproved or False,
            NotificationType.AGENT_REJECTED: user.notifyOnAgentRejected or False,
        }
        daily_limit = user.maxEmailsPerDay or 3
        notification_preference = NotificationPreference(
            user_id=user.id,
            email=user.email,
            preferences=preferences,
            daily_limit=daily_limit,
            # TODO with other changes later, for now we just will email them
            emails_sent_today=0,
            last_reset_date=datetime.now(),
        )
        return NotificationPreference.model_validate(notification_preference)

    except Exception as e:
        raise DatabaseError(
            f"Failed to upsert user notification preference for user {user_id}: {e}"
        ) from e


async def update_user_notification_preference(
    user_id: str, data: NotificationPreferenceDTO
) -> NotificationPreference:
    try:
        update_data: UserUpdateInput = {}
        if data.email:
            update_data["email"] = data.email
        if NotificationType.AGENT_RUN in data.preferences:
            update_data["notifyOnAgentRun"] = data.preferences[
                NotificationType.AGENT_RUN
            ]
        if NotificationType.ZERO_BALANCE in data.preferences:
            update_data["notifyOnZeroBalance"] = data.preferences[
                NotificationType.ZERO_BALANCE
            ]
        if NotificationType.LOW_BALANCE in data.preferences:
            update_data["notifyOnLowBalance"] = data.preferences[
                NotificationType.LOW_BALANCE
            ]
        if NotificationType.BLOCK_EXECUTION_FAILED in data.preferences:
            update_data["notifyOnBlockExecutionFailed"] = data.preferences[
                NotificationType.BLOCK_EXECUTION_FAILED
            ]
        if NotificationType.CONTINUOUS_AGENT_ERROR in data.preferences:
            update_data["notifyOnContinuousAgentError"] = data.preferences[
                NotificationType.CONTINUOUS_AGENT_ERROR
            ]
        if NotificationType.DAILY_SUMMARY in data.preferences:
            update_data["notifyOnDailySummary"] = data.preferences[
                NotificationType.DAILY_SUMMARY
            ]
        if NotificationType.WEEKLY_SUMMARY in data.preferences:
            update_data["notifyOnWeeklySummary"] = data.preferences[
                NotificationType.WEEKLY_SUMMARY
            ]
        if NotificationType.MONTHLY_SUMMARY in data.preferences:
            update_data["notifyOnMonthlySummary"] = data.preferences[
                NotificationType.MONTHLY_SUMMARY
            ]
        if NotificationType.AGENT_APPROVED in data.preferences:
            update_data["notifyOnAgentApproved"] = data.preferences[
                NotificationType.AGENT_APPROVED
            ]
        if NotificationType.AGENT_REJECTED in data.preferences:
            update_data["notifyOnAgentRejected"] = data.preferences[
                NotificationType.AGENT_REJECTED
            ]
        if data.daily_limit:
            update_data["maxEmailsPerDay"] = data.daily_limit

        user = await PrismaUser.prisma().update(
            where={"id": user_id},
            data=update_data,
        )
        if not user:
            raise ValueError(f"User not found with ID: {user_id}")

        # Invalidate cache for this user since notification preferences are part of user data
        get_user_by_id.cache_delete(user_id)

        preferences: dict[NotificationType, bool] = {
            NotificationType.AGENT_RUN: user.notifyOnAgentRun or True,
            NotificationType.ZERO_BALANCE: user.notifyOnZeroBalance or True,
            NotificationType.LOW_BALANCE: user.notifyOnLowBalance or True,
            NotificationType.BLOCK_EXECUTION_FAILED: user.notifyOnBlockExecutionFailed
            or True,
            NotificationType.CONTINUOUS_AGENT_ERROR: user.notifyOnContinuousAgentError
            or True,
            NotificationType.DAILY_SUMMARY: user.notifyOnDailySummary or True,
            NotificationType.WEEKLY_SUMMARY: user.notifyOnWeeklySummary or True,
            NotificationType.MONTHLY_SUMMARY: user.notifyOnMonthlySummary or True,
            NotificationType.AGENT_APPROVED: user.notifyOnAgentApproved or True,
            NotificationType.AGENT_REJECTED: user.notifyOnAgentRejected or True,
        }
        notification_preference = NotificationPreference(
            user_id=user.id,
            email=user.email,
            preferences=preferences,
            daily_limit=user.maxEmailsPerDay or 3,
            # TODO with other changes later, for now we just will email them
            emails_sent_today=0,
            last_reset_date=datetime.now(),
        )
        return NotificationPreference.model_validate(notification_preference)

    except Exception as e:
        raise DatabaseError(
            f"Failed to update user notification preference for user {user_id}: {e}"
        ) from e


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
    """Disable all notification preferences for a user.

    Used when user's email bounces/is inactive to prevent any future notifications.
    """
    try:
        await PrismaUser.prisma().update(
            where={"id": user_id},
            data={
                "notifyOnAgentRun": False,
                "notifyOnZeroBalance": False,
                "notifyOnLowBalance": False,
                "notifyOnBlockExecutionFailed": False,
                "notifyOnContinuousAgentError": False,
                "notifyOnDailySummary": False,
                "notifyOnWeeklySummary": False,
                "notifyOnMonthlySummary": False,
                "notifyOnAgentApproved": False,
                "notifyOnAgentRejected": False,
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
        await update_user_notification_preference(
            user.id,
            NotificationPreferenceDTO(
                email=user.email,
                daily_limit=0,
                preferences={
                    NotificationType.AGENT_RUN: False,
                    NotificationType.ZERO_BALANCE: False,
                    NotificationType.LOW_BALANCE: False,
                    NotificationType.BLOCK_EXECUTION_FAILED: False,
                    NotificationType.CONTINUOUS_AGENT_ERROR: False,
                    NotificationType.DAILY_SUMMARY: False,
                    NotificationType.WEEKLY_SUMMARY: False,
                    NotificationType.MONTHLY_SUMMARY: False,
                },
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

        return User.from_db(user)
    except Exception as e:
        raise DatabaseError(f"Failed to update timezone for user {user_id}: {e}") from e
