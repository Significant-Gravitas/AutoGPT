import asyncio
import csv
import io
import logging
import os
import re
import socket
from datetime import datetime, timezone
from typing import Any, Literal, Optional
from uuid import uuid4

import prisma.enums
import prisma.models
import prisma.types
from prisma.errors import UniqueViolationError
from pydantic import BaseModel, ConfigDict, EmailStr, TypeAdapter, ValidationError

from backend.data.db import transaction
from backend.data.model import User
from backend.data.redis_client import get_redis_async
from backend.data.tally import (
    TallyExtractionTimeoutError,
    extract_business_understanding_from_tally,
    format_submission_for_llm,
    get_tally_submission_by_email,
    mask_email,
)
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
TallyMode = Literal["default", "bulk"]

_tally_seed_tasks: dict[str, asyncio.Task] = {}
_tally_lookup_semaphore = asyncio.Semaphore(1)
_bulk_tally_extraction_semaphore = asyncio.Semaphore(3)
_TALLY_RATE_DELAY = 0.6  # ~100 req/min, matching the Tally rate limit
_TALLY_STALE_SECONDS = 300
_MAX_TALLY_ERROR_LENGTH = 200
_email_adapter = TypeAdapter(EmailStr)

MAX_BULK_INVITE_FILE_BYTES = 1024 * 1024
MAX_BULK_INVITE_ROWS = 5000
_FETCH_EXISTING_EMAILS_CHUNK_SIZE = 500
_BULK_INVITE_CREATE_CONCURRENCY = 50
# Bulk uploads use a much shorter extraction budget so a few slow timeouts
# cannot block the queued enrichment backlog for hours.
_BULK_TALLY_LLM_TIMEOUT = 30
_BULK_TALLY_LLM_MAX_ATTEMPTS = 3


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


class _ParsedInviteRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    row_number: int
    email: str
    name: Optional[str]


_EmailValidatedInviteRow = tuple[_ParsedInviteRow, str, Optional[str]]


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
    search: Optional[str] = None,
) -> tuple[list[InvitedUserRecord], int]:
    where: prisma.types.InvitedUserWhereInput = {}
    if search:
        search = search.strip()
        where["OR"] = [
            {"email": {"contains": search, "mode": "insensitive"}},
            {"name": {"contains": search, "mode": "insensitive"}},
        ]

    total = await prisma.models.InvitedUser.prisma().count(where=where)
    invited_users = await prisma.models.InvitedUser.prisma().find_many(
        where=where,
        order={"createdAt": "desc"},
        skip=(page - 1) * page_size,
        take=page_size,
    )
    return [InvitedUserRecord.from_db(iu) for iu in invited_users], total


async def create_invited_user(
    email: str,
    name: Optional[str] = None,
    *,
    tally_mode: TallyMode = "default",
) -> InvitedUserRecord:
    normalized_email = normalize_email(email)
    normalized_name = _normalize_name(name)

    existing_user, existing_invited_user = await asyncio.gather(
        prisma.models.User.prisma().find_unique(where={"email": normalized_email}),
        prisma.models.InvitedUser.prisma().find_unique(
            where={"email": normalized_email}
        ),
    )
    if existing_user is not None:
        raise PreconditionFailed("An active user with this email already exists")

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
    schedule_invited_user_tally_precompute(invited_user.id, tally_mode=tally_mode)
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
    schedule_invited_user_tally_precompute(invited_user_id, tally_mode="default")
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

    if len(parsed_rows) > MAX_BULK_INVITE_ROWS:
        raise ValueError(
            f"Invite file exceeds the maximum of {MAX_BULK_INVITE_ROWS} users"
        )

    return parsed_rows


async def _fetch_existing_emails(emails: list[str]) -> tuple[set[str], set[str]]:
    """Batch-fetch emails that already exist as active users or invited users."""
    if not emails:
        return set(), set()

    active_emails: set[str] = set()
    invited_emails: set[str] = set()

    for i in range(0, len(emails), _FETCH_EXISTING_EMAILS_CHUNK_SIZE):
        email_chunk = emails[i : i + _FETCH_EXISTING_EMAILS_CHUNK_SIZE]
        existing_users, existing_invited = await asyncio.gather(
            prisma.models.User.prisma().find_many(
                where={"email": {"in": email_chunk}},
            ),
            prisma.models.InvitedUser.prisma().find_many(
                where={"email": {"in": email_chunk}},
            ),
        )
        active_emails.update(u.email for u in existing_users)
        invited_emails.update(iu.email for iu in existing_invited)

    return active_emails, invited_emails


async def _create_bulk_invited_user(
    semaphore: asyncio.Semaphore,
    email: str,
    name: Optional[str],
) -> InvitedUserRecord:
    async with semaphore:
        return await create_invited_user(email, name, tally_mode="bulk")


def _validate_bulk_invite_rows(
    parsed_rows: list[_ParsedInviteRow],
) -> tuple[list[_EmailValidatedInviteRow], list[BulkInvitedUserRowResult], int, int]:
    email_validated_rows: list[_EmailValidatedInviteRow] = []
    results: list[BulkInvitedUserRowResult] = []
    skipped_count = 0
    error_count = 0
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
        email_validated_rows.append((row, normalized_email, row_name))

    return email_validated_rows, results, skipped_count, error_count


async def _filter_preexisting_bulk_invite_rows(
    email_validated_rows: list[_EmailValidatedInviteRow],
) -> tuple[list[_EmailValidatedInviteRow], list[BulkInvitedUserRowResult], int]:
    all_emails = [email for _, email, _ in email_validated_rows]
    active_emails, invited_emails = await _fetch_existing_emails(all_emails)

    precheck_passed_rows: list[_EmailValidatedInviteRow] = []
    results: list[BulkInvitedUserRowResult] = []
    skipped_count = 0

    for row, normalized_email, row_name in email_validated_rows:
        if normalized_email in active_emails:
            skipped_count += 1
            results.append(
                BulkInvitedUserRowResult(
                    row_number=row.row_number,
                    email=normalized_email,
                    name=row_name,
                    status="SKIPPED",
                    message="An active user with this email already exists",
                )
            )
            continue

        if normalized_email in invited_emails:
            skipped_count += 1
            results.append(
                BulkInvitedUserRowResult(
                    row_number=row.row_number,
                    email=normalized_email,
                    name=row_name,
                    status="SKIPPED",
                    message="An invited user with this email already exists",
                )
            )
            continue

        precheck_passed_rows.append((row, normalized_email, row_name))

    return precheck_passed_rows, results, skipped_count


async def _create_bulk_invite_results(
    precheck_passed_rows: list[_EmailValidatedInviteRow],
) -> tuple[list[BulkInvitedUserRowResult], int, int, int]:
    create_semaphore = asyncio.Semaphore(_BULK_INVITE_CREATE_CONCURRENCY)
    outcomes = await asyncio.gather(
        *(
            _create_bulk_invited_user(create_semaphore, email, name)
            for _, email, name in precheck_passed_rows
        ),
        return_exceptions=True,
    )

    results: list[BulkInvitedUserRowResult] = []
    created_count = 0
    skipped_count = 0
    error_count = 0

    for (row, normalized_email, row_name), outcome in zip(
        precheck_passed_rows, outcomes
    ):
        if isinstance(outcome, PreconditionFailed):
            skipped_count += 1
            results.append(
                BulkInvitedUserRowResult(
                    row_number=row.row_number,
                    email=normalized_email,
                    name=row_name,
                    status="SKIPPED",
                    message=str(outcome),
                )
            )
            continue

        if isinstance(outcome, BaseException):
            if not isinstance(outcome, Exception):
                raise outcome
            masked = mask_email(normalized_email)
            logger.exception(
                "Failed to create bulk invite for row %s (%s)",
                row.row_number,
                masked,
                exc_info=outcome,
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
            continue

        created_count += 1
        results.append(
            BulkInvitedUserRowResult(
                row_number=row.row_number,
                email=normalized_email,
                name=row_name,
                status="CREATED",
                message="Invite created",
                invited_user=outcome,
            )
        )

    return results, created_count, skipped_count, error_count


async def bulk_create_invited_users_from_file(
    filename: Optional[str],
    content: bytes,
) -> BulkInvitedUsersResult:
    parsed_rows = _parse_bulk_invite_file(filename, content)
    (
        email_validated_rows,
        validation_results,
        validation_skipped_count,
        validation_error_count,
    ) = _validate_bulk_invite_rows(parsed_rows)
    (
        precheck_passed_rows,
        precheck_results,
        precheck_skipped_count,
    ) = await _filter_preexisting_bulk_invite_rows(email_validated_rows)
    (
        creation_results,
        created_count,
        creation_skipped_count,
        creation_error_count,
    ) = await _create_bulk_invite_results(precheck_passed_rows)

    results = [
        *validation_results,
        *precheck_results,
        *creation_results,
    ]

    results.sort(key=lambda r: r.row_number)
    return BulkInvitedUsersResult(
        created_count=created_count,
        skipped_count=(
            validation_skipped_count + precheck_skipped_count + creation_skipped_count
        ),
        error_count=validation_error_count + creation_error_count,
        results=results,
    )


async def _fetch_tally_submission_with_rate_limit(
    email: str,
    *,
    require_api_key: bool = False,
) -> Optional[tuple[dict, list]]:
    async with _tally_lookup_semaphore:
        result = await get_tally_submission_by_email(
            email,
            require_api_key=require_api_key,
        )
        if result is not None:
            await asyncio.sleep(_TALLY_RATE_DELAY)
        return result


async def _get_invited_user_tally_understanding(
    email: str,
    *,
    tally_mode: TallyMode = "default",
) -> Optional[BusinessUnderstandingInput]:
    result = await _fetch_tally_submission_with_rate_limit(
        email,
        require_api_key=True,
    )
    if result is None:
        return None

    submission, questions = result
    formatted = format_submission_for_llm(submission, questions)
    if not formatted.strip():
        logger.warning("Tally: formatted submission was empty, skipping")
        return None

    if tally_mode == "bulk":
        async with _bulk_tally_extraction_semaphore:
            return await extract_business_understanding_from_tally(
                formatted,
                timeout_seconds=_BULK_TALLY_LLM_TIMEOUT,
                max_attempts=_BULK_TALLY_LLM_MAX_ATTEMPTS,
            )

    return await extract_business_understanding_from_tally(formatted)


async def _fail_tally_seed(invited_user_id: str, exc: Exception) -> None:
    sanitized_error = re.sub(r"https?://\S+", "<url>", f"{type(exc).__name__}: {exc}")[
        :_MAX_TALLY_ERROR_LENGTH
    ]
    await prisma.models.InvitedUser.prisma().update(
        where={"id": invited_user_id},
        data={
            "tallyStatus": prisma.enums.TallyComputationStatus.FAILED,
            "tallyError": sanitized_error,
        },
    )


async def _should_skip_tally_seed_for_lock(invited_user_id: str) -> bool:
    try:
        redis = await get_redis_async()
    except Exception:
        return False

    if redis is None:
        return False

    lock = AsyncClusterLock(
        redis=redis,
        key=f"tally_seed:{invited_user_id}",
        owner_id=_WORKER_ID,
        timeout=_TALLY_STALE_SECONDS,
    )
    current_owner = await lock.try_acquire()

    if current_owner is None:
        logger.warning("Redis unavailable for tally lock - skipping tally enrichment")
        return True

    if current_owner != _WORKER_ID:
        logger.debug(
            "Tally seed for %s already locked by %s, skipping",
            invited_user_id,
            current_owner,
        )
        return True

    return False


def _should_skip_running_tally_seed(
    invited_user: "prisma.models.InvitedUser",
    invited_user_id: str,
) -> bool:
    if (
        invited_user.tallyStatus != prisma.enums.TallyComputationStatus.RUNNING
        or invited_user.updatedAt is None
    ):
        return False

    age = (datetime.now(timezone.utc) - invited_user.updatedAt).total_seconds()
    if age < _TALLY_STALE_SECONDS:
        logger.debug(
            "Tally task for %s still RUNNING (age=%ds), skipping",
            invited_user_id,
            int(age),
        )
        return True

    logger.info(
        "Tally task for %s is stale (age=%ds), re-running",
        invited_user_id,
        int(age),
    )
    return False


async def _mark_tally_seed_running(invited_user_id: str) -> None:
    await prisma.models.InvitedUser.prisma().update(
        where={"id": invited_user_id},
        data={
            "tallyStatus": prisma.enums.TallyComputationStatus.RUNNING,
            "tallyError": None,
        },
    )


async def _mark_tally_seed_ready(
    invited_user_id: str,
    input_data: Optional[BusinessUnderstandingInput],
) -> None:
    update_data: prisma.types.InvitedUserUpdateInput = {
        "tallyStatus": prisma.enums.TallyComputationStatus.READY,
        "tallyComputedAt": datetime.now(timezone.utc),
        "tallyError": None,
    }
    if input_data is not None:
        update_data["tallyUnderstanding"] = SafeJson(
            input_data.model_dump(exclude_none=True)
        )

    await prisma.models.InvitedUser.prisma().update(
        where={"id": invited_user_id},
        data=update_data,
    )


async def _compute_invited_user_tally_seed(
    invited_user_id: str,
    *,
    tally_mode: TallyMode = "default",
) -> None:
    invited_user = await prisma.models.InvitedUser.prisma().find_unique(
        where={"id": invited_user_id}
    )
    if invited_user is None:
        return

    if invited_user.status == prisma.enums.InvitedUserStatus.REVOKED:
        return

    if await _should_skip_tally_seed_for_lock(invited_user_id):
        return

    if _should_skip_running_tally_seed(invited_user, invited_user_id):
        return

    await _mark_tally_seed_running(invited_user_id)

    try:
        input_data = await _get_invited_user_tally_understanding(
            invited_user.email,
            tally_mode=tally_mode,
        )
        await _mark_tally_seed_ready(invited_user_id, input_data)
    except TallyExtractionTimeoutError as exc:
        logger.warning(
            "Timed out computing Tally understanding for invited user %s after %s attempts",
            invited_user_id,
            exc.attempts,
        )
        await _fail_tally_seed(invited_user_id, exc)
    except Exception as exc:
        logger.exception(
            "Failed to compute Tally understanding for invited user %s",
            invited_user_id,
        )
        await _fail_tally_seed(invited_user_id, exc)


def schedule_invited_user_tally_precompute(
    invited_user_id: str,
    *,
    tally_mode: TallyMode = "default",
) -> None:
    existing = _tally_seed_tasks.get(invited_user_id)
    if existing is not None and not existing.done():
        logger.debug("Tally task already running for %s, skipping", invited_user_id)
        return

    task = asyncio.create_task(
        _compute_invited_user_tally_seed(
            invited_user_id,
            tally_mode=tally_mode,
        )
    )
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
