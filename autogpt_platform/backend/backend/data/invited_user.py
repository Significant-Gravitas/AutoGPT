import asyncio
import csv
import io
import logging
import os
import re
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Optional
from uuid import uuid4

import prisma.enums
import prisma.models
import prisma.types
from prisma.errors import UniqueViolationError
from pydantic import BaseModel, EmailStr, TypeAdapter, ValidationError

from backend.data.db import transaction
from backend.data.model import User
from backend.data.redis_client import get_redis_async
from backend.data.tally import get_business_understanding_input_from_tally, mask_email
from backend.data.understanding import (
    BusinessUnderstandingInput,
    merge_business_understanding_data,
)
from backend.data.user import get_user_by_email, get_user_by_id
from backend.executor.cluster_lock import AsyncClusterLock
from backend.util.exceptions import (
    NotAuthorizedError,
    NotFoundError,
    PreconditionFailed,
)
from backend.util.json import SafeJson
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
_settings = Settings()

_WORKER_ID = f"{socket.gethostname()}:{os.getpid()}"

_tally_seed_tasks: dict[str, asyncio.Task] = {}
_TALLY_STALE_SECONDS = 300
_MAX_TALLY_ERROR_LENGTH = 200
_email_adapter = TypeAdapter(EmailStr)

MAX_BULK_INVITE_FILE_BYTES = 1024 * 1024
MAX_BULK_INVITE_ROWS = 500


class InvitedUserRecord(BaseModel):
    id: str
    email: str
    status: prisma.enums.InvitedUserStatus
    auth_user_id: Optional[str] = None
    name: Optional[str] = None
    tally_understanding: Optional[dict[str, Any]] = None
    tally_status: prisma.enums.TallyComputationStatus
    tally_computed_at: Optional[datetime] = None
    tally_error: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_db(cls, invited_user: "prisma.models.InvitedUser") -> "InvitedUserRecord":
        payload = (
            invited_user.tallyUnderstanding
            if isinstance(invited_user.tallyUnderstanding, dict)
            else None
        )
        return cls(
            id=invited_user.id,
            email=invited_user.email,
            status=invited_user.status,
            auth_user_id=invited_user.authUserId,
            name=invited_user.name,
            tally_understanding=payload,
            tally_status=invited_user.tallyStatus,
            tally_computed_at=invited_user.tallyComputedAt,
            tally_error=invited_user.tallyError,
            created_at=invited_user.createdAt,
            updated_at=invited_user.updatedAt,
        )


class BulkInvitedUserRowResult(BaseModel):
    row_number: int
    email: Optional[str] = None
    name: Optional[str] = None
    status: Literal["CREATED", "SKIPPED", "ERROR"]
    message: str
    invited_user: Optional[InvitedUserRecord] = None


class BulkInvitedUsersResult(BaseModel):
    created_count: int
    skipped_count: int
    error_count: int
    results: list[BulkInvitedUserRowResult]


@dataclass(frozen=True)
class _ParsedInviteRow:
    row_number: int
    email: str
    name: Optional[str]


def normalize_email(email: str) -> str:
    return email.strip().lower()


def is_internal_email(email: str) -> bool:
    """Return True for @agpt.co addresses, which always bypass the invite gate."""
    return normalize_email(email).endswith("@agpt.co")


def _normalize_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    normalized = name.strip()
    return normalized or None


def _default_profile_name(email: str, preferred_name: Optional[str]) -> str:
    if preferred_name:
        return preferred_name
    local_part = email.split("@", 1)[0].strip()
    return local_part or "user"


def _sanitize_username_base(email: str) -> str:
    local_part = email.split("@", 1)[0].lower()
    sanitized = re.sub(r"[^a-z0-9-]", "", local_part)
    sanitized = sanitized.strip("-")
    return sanitized[:40] or "user"


async def _generate_unique_profile_username(email: str, tx) -> str:
    base = _sanitize_username_base(email)

    for _ in range(2):
        candidate = f"{base}-{uuid4().hex[:6]}"
        existing = await prisma.models.Profile.prisma(tx).find_unique(
            where={"username": candidate}
        )
        if existing is None:
            return candidate

    raise RuntimeError(f"Unable to generate unique username for {email}")


async def _ensure_default_profile(
    user_id: str,
    email: str,
    preferred_name: Optional[str],
    tx,
) -> None:
    existing_profile = await prisma.models.Profile.prisma(tx).find_unique(
        where={"userId": user_id}
    )
    if existing_profile is not None:
        return

    username = await _generate_unique_profile_username(email, tx)
    await prisma.models.Profile.prisma(tx).create(
        data=prisma.types.ProfileCreateInput(
            userId=user_id,
            name=_default_profile_name(email, preferred_name),
            username=username,
            description="I'm new here",
            links=[],
            avatarUrl="",
        )
    )


async def _ensure_default_onboarding(user_id: str, tx) -> None:
    await prisma.models.UserOnboarding.prisma(tx).upsert(
        where={"userId": user_id},
        data={
            "create": prisma.types.UserOnboardingCreateInput(userId=user_id),
            "update": {},
        },
    )


async def _apply_tally_understanding(
    user_id: str,
    invited_user: "prisma.models.InvitedUser",
    tx,
) -> None:
    if not isinstance(invited_user.tallyUnderstanding, dict):
        return

    try:
        input_data = BusinessUnderstandingInput.model_validate(
            invited_user.tallyUnderstanding
        )
    except Exception:
        logger.warning(
            "Malformed tallyUnderstanding for invited user %s; skipping",
            invited_user.id,
            exc_info=True,
        )
        return

    payload = merge_business_understanding_data({}, input_data)
    await prisma.models.CoPilotUnderstanding.prisma(tx).upsert(
        where={"userId": user_id},
        data={
            "create": {"userId": user_id, "data": SafeJson(payload)},
            "update": {"data": SafeJson(payload)},
        },
    )


async def check_invite_eligibility(email: str) -> bool:
    """Check if an email is allowed to sign up based on the invite list.

    Args:
        email: The email to check (will be normalized internally).

    Returns True if the email has an active (INVITED) invite record.
    Does NOT check enable_invite_gate — the caller is responsible for that.
    """
    email = normalize_email(email)
    invited_user = await prisma.models.InvitedUser.prisma().find_unique(
        where={"email": email}
    )
    return (
        invited_user is not None
        and invited_user.status == prisma.enums.InvitedUserStatus.INVITED
    )


async def list_invited_users(
    page: int = 1,
    page_size: int = 50,
) -> tuple[list[InvitedUserRecord], int]:
    total = await prisma.models.InvitedUser.prisma().count()
    invited_users = await prisma.models.InvitedUser.prisma().find_many(
        order={"createdAt": "desc"},
        skip=(page - 1) * page_size,
        take=page_size,
    )
    return [InvitedUserRecord.from_db(iu) for iu in invited_users], total


async def create_invited_user(
    email: str, name: Optional[str] = None
) -> InvitedUserRecord:
    normalized_email = normalize_email(email)
    normalized_name = _normalize_name(name)

    existing_user = await prisma.models.User.prisma().find_unique(
        where={"email": normalized_email}
    )
    if existing_user is not None:
        raise PreconditionFailed("An active user with this email already exists")

    existing_invited_user = await prisma.models.InvitedUser.prisma().find_unique(
        where={"email": normalized_email}
    )
    if existing_invited_user is not None:
        raise PreconditionFailed("An invited user with this email already exists")

    try:
        invited_user = await prisma.models.InvitedUser.prisma().create(
            data={
                "email": normalized_email,
                "name": normalized_name,
                "status": prisma.enums.InvitedUserStatus.INVITED,
                "tallyStatus": prisma.enums.TallyComputationStatus.PENDING,
            }
        )
    except UniqueViolationError:
        raise PreconditionFailed("An invited user with this email already exists")
    schedule_invited_user_tally_precompute(invited_user.id)
    return InvitedUserRecord.from_db(invited_user)


async def revoke_invited_user(invited_user_id: str) -> InvitedUserRecord:
    invited_user = await prisma.models.InvitedUser.prisma().find_unique(
        where={"id": invited_user_id}
    )
    if invited_user is None:
        raise NotFoundError(f"Invited user {invited_user_id} not found")

    if invited_user.status == prisma.enums.InvitedUserStatus.CLAIMED:
        raise PreconditionFailed("Claimed invited users cannot be revoked")

    if invited_user.status == prisma.enums.InvitedUserStatus.REVOKED:
        return InvitedUserRecord.from_db(invited_user)

    revoked_user = await prisma.models.InvitedUser.prisma().update(
        where={"id": invited_user_id},
        data={"status": prisma.enums.InvitedUserStatus.REVOKED},
    )
    if revoked_user is None:
        raise NotFoundError(f"Invited user {invited_user_id} not found")
    return InvitedUserRecord.from_db(revoked_user)


async def retry_invited_user_tally(invited_user_id: str) -> InvitedUserRecord:
    invited_user = await prisma.models.InvitedUser.prisma().find_unique(
        where={"id": invited_user_id}
    )
    if invited_user is None:
        raise NotFoundError(f"Invited user {invited_user_id} not found")

    if invited_user.status == prisma.enums.InvitedUserStatus.REVOKED:
        raise PreconditionFailed("Revoked invited users cannot retry Tally seeding")

    refreshed_user = await prisma.models.InvitedUser.prisma().update(
        where={"id": invited_user_id},
        data={
            "tallyUnderstanding": None,
            "tallyStatus": prisma.enums.TallyComputationStatus.PENDING,
            "tallyComputedAt": None,
            "tallyError": None,
        },
    )
    if refreshed_user is None:
        raise NotFoundError(f"Invited user {invited_user_id} not found")
    schedule_invited_user_tally_precompute(invited_user_id)
    return InvitedUserRecord.from_db(refreshed_user)


def _decode_bulk_invite_file(content: bytes) -> str:
    if len(content) > MAX_BULK_INVITE_FILE_BYTES:
        raise ValueError("Invite file exceeds the maximum size of 1 MB")

    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Invite file must be UTF-8 encoded") from exc


def _parse_bulk_invite_csv(text: str) -> list[_ParsedInviteRow]:
    indexed_rows: list[tuple[int, list[str]]] = []

    for row_number, row in enumerate(csv.reader(io.StringIO(text)), start=1):
        normalized_row = [cell.strip() for cell in row]
        if any(normalized_row):
            indexed_rows.append((row_number, normalized_row))

    if not indexed_rows:
        return []

    header = [cell.lower() for cell in indexed_rows[0][1]]
    has_header = "email" in header
    email_index = header.index("email") if has_header else 0
    name_index: Optional[int] = (
        header.index("name")
        if has_header and "name" in header
        else (1 if not has_header else None)
    )
    data_rows = indexed_rows[1:] if has_header else indexed_rows

    parsed_rows: list[_ParsedInviteRow] = []
    for row_number, row in data_rows:
        if len(parsed_rows) >= MAX_BULK_INVITE_ROWS:
            break
        email = row[email_index].strip() if len(row) > email_index else ""
        name = (
            row[name_index].strip()
            if name_index is not None and len(row) > name_index
            else ""
        )
        parsed_rows.append(
            _ParsedInviteRow(
                row_number=row_number,
                email=email,
                name=name or None,
            )
        )

    return parsed_rows


def _parse_bulk_invite_text(text: str) -> list[_ParsedInviteRow]:
    parsed_rows: list[_ParsedInviteRow] = []

    for row_number, raw_line in enumerate(text.splitlines(), start=1):
        if len(parsed_rows) >= MAX_BULK_INVITE_ROWS:
            break
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parsed_rows.append(
            _ParsedInviteRow(
                row_number=row_number,
                email=line,
                name=None,
            )
        )

    return parsed_rows


def _parse_bulk_invite_file(
    filename: Optional[str],
    content: bytes,
) -> list[_ParsedInviteRow]:
    text = _decode_bulk_invite_file(content)
    file_name = filename.lower() if filename else ""
    parsed_rows = (
        _parse_bulk_invite_csv(text)
        if file_name.endswith(".csv")
        else _parse_bulk_invite_text(text)
    )

    if not parsed_rows:
        raise ValueError("Invite file did not contain any emails")

    return parsed_rows


async def bulk_create_invited_users_from_file(
    filename: Optional[str],
    content: bytes,
) -> BulkInvitedUsersResult:
    parsed_rows = _parse_bulk_invite_file(filename, content)

    created_count = 0
    skipped_count = 0
    error_count = 0
    results: list[BulkInvitedUserRowResult] = []
    seen_emails: set[str] = set()

    for row in parsed_rows:
        row_name = _normalize_name(row.name)

        try:
            validated_email = _email_adapter.validate_python(row.email)
        except ValidationError:
            error_count += 1
            results.append(
                BulkInvitedUserRowResult(
                    row_number=row.row_number,
                    email=row.email or None,
                    name=row_name,
                    status="ERROR",
                    message="Invalid email address",
                )
            )
            continue

        normalized_email = normalize_email(str(validated_email))
        if normalized_email in seen_emails:
            skipped_count += 1
            results.append(
                BulkInvitedUserRowResult(
                    row_number=row.row_number,
                    email=normalized_email,
                    name=row_name,
                    status="SKIPPED",
                    message="Duplicate email in upload file",
                )
            )
            continue

        seen_emails.add(normalized_email)

        try:
            invited_user = await create_invited_user(normalized_email, row_name)
        except PreconditionFailed as exc:
            skipped_count += 1
            results.append(
                BulkInvitedUserRowResult(
                    row_number=row.row_number,
                    email=normalized_email,
                    name=row_name,
                    status="SKIPPED",
                    message=str(exc),
                )
            )
        except Exception:
            masked = mask_email(normalized_email)
            logger.exception(
                "Failed to create bulk invite for row %s (%s)",
                row.row_number,
                masked,
            )
            error_count += 1
            results.append(
                BulkInvitedUserRowResult(
                    row_number=row.row_number,
                    email=normalized_email,
                    name=row_name,
                    status="ERROR",
                    message="Unexpected error creating invite",
                )
            )
        else:
            created_count += 1
            results.append(
                BulkInvitedUserRowResult(
                    row_number=row.row_number,
                    email=normalized_email,
                    name=row_name,
                    status="CREATED",
                    message="Invite created",
                    invited_user=invited_user,
                )
            )

    return BulkInvitedUsersResult(
        created_count=created_count,
        skipped_count=skipped_count,
        error_count=error_count,
        results=results,
    )


async def _compute_invited_user_tally_seed(invited_user_id: str) -> None:
    invited_user = await prisma.models.InvitedUser.prisma().find_unique(
        where={"id": invited_user_id}
    )
    if invited_user is None:
        return

    if invited_user.status == prisma.enums.InvitedUserStatus.REVOKED:
        return

    try:
        r = await get_redis_async()
    except Exception:
        r = None

    lock: AsyncClusterLock | None = None

    if r is not None:
        lock = AsyncClusterLock(
            redis=r,
            key=f"tally_seed:{invited_user_id}",
            owner_id=_WORKER_ID,
            timeout=_TALLY_STALE_SECONDS,
        )
        current_owner = await lock.try_acquire()

        if current_owner is None:
            logger.warn("Redis unvailable for tally lock - skipping tally enrichement")
            return
        elif current_owner != _WORKER_ID:
            logger.debug(
                "Tally seed for %s already locked by %s, skipping",
                invited_user_id,
                current_owner,
            )
            return
    if (
        invited_user.tallyStatus == prisma.enums.TallyComputationStatus.RUNNING
        and invited_user.updatedAt is not None
    ):
        age = (datetime.now(timezone.utc) - invited_user.updatedAt).total_seconds()
        if age < _TALLY_STALE_SECONDS:
            logger.debug(
                "Tally task for %s still RUNNING (age=%ds), skipping",
                invited_user_id,
                int(age),
            )
            return
        logger.info(
            "Tally task for %s is stale (age=%ds), re-running",
            invited_user_id,
            int(age),
        )

    await prisma.models.InvitedUser.prisma().update(
        where={"id": invited_user_id},
        data={
            "tallyStatus": prisma.enums.TallyComputationStatus.RUNNING,
            "tallyError": None,
        },
    )

    try:
        input_data = await get_business_understanding_input_from_tally(
            invited_user.email,
            require_api_key=True,
        )
        payload = (
            SafeJson(input_data.model_dump(exclude_none=True))
            if input_data is not None
            else None
        )
        await prisma.models.InvitedUser.prisma().update(
            where={"id": invited_user_id},
            data={
                "tallyUnderstanding": payload,
                "tallyStatus": prisma.enums.TallyComputationStatus.READY,
                "tallyComputedAt": datetime.now(timezone.utc),
                "tallyError": None,
            },
        )
    except Exception as exc:
        logger.exception(
            "Failed to compute Tally understanding for invited user %s",
            invited_user_id,
        )
        sanitized_error = re.sub(
            r"https?://\S+", "<url>", f"{type(exc).__name__}: {exc}"
        )[:_MAX_TALLY_ERROR_LENGTH]
        await prisma.models.InvitedUser.prisma().update(
            where={"id": invited_user_id},
            data={
                "tallyStatus": prisma.enums.TallyComputationStatus.FAILED,
                "tallyError": sanitized_error,
            },
        )


def schedule_invited_user_tally_precompute(invited_user_id: str) -> None:
    existing = _tally_seed_tasks.get(invited_user_id)
    if existing is not None and not existing.done():
        logger.debug("Tally task already running for %s, skipping", invited_user_id)
        return

    task = asyncio.create_task(_compute_invited_user_tally_seed(invited_user_id))
    _tally_seed_tasks[invited_user_id] = task

    def _on_done(t: asyncio.Task, _id: str = invited_user_id) -> None:
        if _tally_seed_tasks.get(_id) is t:
            del _tally_seed_tasks[_id]

    task.add_done_callback(_on_done)


async def _open_signup_create_user(
    auth_user_id: str,
    normalized_email: str,
    metadata_name: Optional[str],
) -> User:
    """Create a user without requiring an invite (open signup mode)."""
    preferred_name = _normalize_name(metadata_name)
    try:
        async with transaction() as tx:
            user = await prisma.models.User.prisma(tx).create(
                data=prisma.types.UserCreateInput(
                    id=auth_user_id,
                    email=normalized_email,
                    name=preferred_name,
                )
            )
            await _ensure_default_profile(
                auth_user_id, normalized_email, preferred_name, tx
            )
            await _ensure_default_onboarding(auth_user_id, tx)
    except UniqueViolationError:
        existing = await prisma.models.User.prisma().find_unique(
            where={"id": auth_user_id}
        )
        if existing is not None:
            return User.from_db(existing)
        raise

    return User.from_db(user)


# TODO: We need to change this functions logic before going live
async def get_or_activate_user(user_data: dict) -> User:
    auth_user_id = user_data.get("sub")
    if not auth_user_id:
        raise NotAuthorizedError("User ID not found in token")

    auth_email = user_data.get("email")
    if not auth_email:
        raise NotAuthorizedError("Email not found in token")

    normalized_email = normalize_email(auth_email)
    user_metadata = user_data.get("user_metadata")
    metadata_name = (
        user_metadata.get("name") if isinstance(user_metadata, dict) else None
    )

    existing_user = None
    try:
        existing_user = await get_user_by_id(auth_user_id)
    except ValueError:
        existing_user = None
    except Exception:
        logger.exception("Error on get user by id during tally enrichment process")
        raise

    if existing_user is not None:
        return existing_user

    if not _settings.config.enable_invite_gate or is_internal_email(normalized_email):
        return await _open_signup_create_user(
            auth_user_id, normalized_email, metadata_name
        )

    invited_user = await prisma.models.InvitedUser.prisma().find_unique(
        where={"email": normalized_email}
    )
    if invited_user is None:
        raise NotAuthorizedError("Your email is not allowed to access the platform")

    if invited_user.status != prisma.enums.InvitedUserStatus.INVITED:
        raise NotAuthorizedError("Your invitation is no longer active")

    try:
        async with transaction() as tx:
            current_user = await prisma.models.User.prisma(tx).find_unique(
                where={"id": auth_user_id}
            )
            if current_user is not None:
                return User.from_db(current_user)

            current_invited_user = await prisma.models.InvitedUser.prisma(
                tx
            ).find_unique(where={"email": normalized_email})
            if current_invited_user is None:
                raise NotAuthorizedError(
                    "Your email is not allowed to access the platform"
                )

            if current_invited_user.status != prisma.enums.InvitedUserStatus.INVITED:
                raise NotAuthorizedError("Your invitation is no longer active")

            if current_invited_user.authUserId not in (None, auth_user_id):
                raise NotAuthorizedError("Your invitation has already been claimed")

            preferred_name = current_invited_user.name or _normalize_name(metadata_name)
            await prisma.models.User.prisma(tx).create(
                data=prisma.types.UserCreateInput(
                    id=auth_user_id,
                    email=normalized_email,
                    name=preferred_name,
                )
            )

            await prisma.models.InvitedUser.prisma(tx).update(
                where={"id": current_invited_user.id},
                data={
                    "status": prisma.enums.InvitedUserStatus.CLAIMED,
                    "authUserId": auth_user_id,
                },
            )

            await _ensure_default_profile(
                auth_user_id,
                normalized_email,
                preferred_name,
                tx,
            )
            await _ensure_default_onboarding(auth_user_id, tx)
            await _apply_tally_understanding(auth_user_id, current_invited_user, tx)
    except UniqueViolationError:
        logger.info("Concurrent activation for user %s; re-fetching", auth_user_id)
        already_created = await prisma.models.User.prisma().find_unique(
            where={"id": auth_user_id}
        )
        if already_created is not None:
            return User.from_db(already_created)
        raise RuntimeError(
            f"UniqueViolationError during activation but user {auth_user_id} not found"
        )

    get_user_by_id.cache_delete(auth_user_id)
    get_user_by_email.cache_delete(normalized_email)

    activated_user = await prisma.models.User.prisma().find_unique(
        where={"id": auth_user_id}
    )
    if activated_user is None:
        raise RuntimeError(
            f"Activated user {auth_user_id} was not found after creation"
        )

    return User.from_db(activated_user)
