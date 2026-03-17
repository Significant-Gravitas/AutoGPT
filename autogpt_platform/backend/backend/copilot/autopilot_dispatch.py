from __future__ import annotations

import logging
from datetime import UTC, date, datetime, time, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4
from zoneinfo import ZoneInfo

import prisma.enums

from backend.copilot import stream_registry
from backend.copilot.autopilot_prompts import (
    AUTOPILOT_CALLBACK_TAG,
    AUTOPILOT_DISABLED_TOOLS,
    AUTOPILOT_INVITE_CTA_TAG,
    AUTOPILOT_NIGHTLY_TAG_PREFIX,
    _build_autopilot_system_prompt,
    _render_initial_message,
)
from backend.copilot.constants import COPILOT_SESSION_PREFIX
from backend.copilot.executor.utils import enqueue_copilot_turn
from backend.copilot.model import ChatMessage, ChatSession, create_chat_session
from backend.copilot.session_types import ChatSessionConfig, ChatSessionStartType
from backend.data.db_accessors import chat_db, invited_user_db, user_db
from backend.data.model import User
from backend.util.feature_flag import Flag, is_feature_enabled
from backend.util.settings import Settings
from backend.util.timezone_utils import get_user_timezone_or_utc

if TYPE_CHECKING:
    from backend.data.invited_user import InvitedUserRecord

logger = logging.getLogger(__name__)
settings = Settings()

DISPATCH_BATCH_SIZE = 500


# --------------- tag helpers --------------- #


def get_graph_exec_id_for_session(session_id: str) -> str:
    return f"{COPILOT_SESSION_PREFIX}{session_id}"


def get_nightly_execution_tag(target_local_date: date) -> str:
    return f"{AUTOPILOT_NIGHTLY_TAG_PREFIX}{target_local_date.isoformat()}"


def get_callback_execution_tag() -> str:
    return AUTOPILOT_CALLBACK_TAG


def get_invite_cta_execution_tag() -> str:
    return AUTOPILOT_INVITE_CTA_TAG


def _get_manual_trigger_execution_tag(start_type: ChatSessionStartType) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"admin-autopilot:{start_type.value}:{timestamp}:{uuid4()}"


# --------------- timezone helpers --------------- #


def _bucket_end_for_now(now_utc: datetime) -> datetime:
    minute = 30 if now_utc.minute >= 30 else 0
    return now_utc.replace(minute=minute, second=0, microsecond=0)


def _resolve_timezone_name(raw_timezone: str | None) -> str:
    return get_user_timezone_or_utc(raw_timezone)


def _crosses_local_midnight(
    bucket_start_utc: datetime,
    bucket_end_utc: datetime,
    timezone_name: str,
) -> date | None:
    """Return the new local date if *bucket_end_utc* falls on a different local
    date than *bucket_start_utc*, taking DST transitions into account.

    During a DST spring-forward the wall clock jumps forward (e.g. 01:59 →
    03:00).  We use ``fold=0`` on the end instant so that ambiguous/missing
    times are resolved consistently and a single 30-min UTC bucket can never
    produce midnight on *two* consecutive calls.
    """
    try:
        tz = ZoneInfo(timezone_name)
    except (KeyError, Exception):
        logger.warning("Unknown timezone %s, falling back to UTC", timezone_name)
        tz = ZoneInfo("UTC")
    start_local = bucket_start_utc.astimezone(tz)
    end_local = bucket_end_utc.astimezone(tz)
    # Resolve ambiguous wall-clock times consistently (spring-forward / fall-back)
    end_local = end_local.replace(fold=0)
    if start_local.date() == end_local.date():
        return None
    return end_local.date()


# --------------- thin DB wrappers --------------- #


async def _user_has_recent_manual_message(user_id: str, since: datetime) -> bool:
    return await chat_db().has_recent_manual_message(user_id, since)


async def _user_has_session_since(user_id: str, since: datetime) -> bool:
    return await chat_db().has_session_since(user_id, since)


async def _session_exists_for_execution_tag(user_id: str, execution_tag: str) -> bool:
    return await chat_db().session_exists_for_execution_tag(user_id, execution_tag)


# --------------- session creation --------------- #


async def _enqueue_session_turn(
    session: ChatSession,
    *,
    message: str,
    tool_name: str,
) -> None:
    turn_id = str(uuid4())
    await stream_registry.create_session(
        session_id=session.session_id,
        user_id=session.user_id,
        tool_call_id=tool_name,
        tool_name=tool_name,
        turn_id=turn_id,
        blocking=False,
    )
    await enqueue_copilot_turn(
        session_id=session.session_id,
        user_id=session.user_id,
        message=message,
        turn_id=turn_id,
        is_user_message=True,
    )


async def _create_autopilot_session(
    user: User,
    *,
    start_type: ChatSessionStartType,
    execution_tag: str,
    timezone_name: str,
    target_local_date: date | None = None,
    invited_user: InvitedUserRecord | None = None,
) -> ChatSession | None:
    if await _session_exists_for_execution_tag(user.id, execution_tag):
        return None

    system_prompt = await _build_autopilot_system_prompt(
        user,
        start_type=start_type,
        timezone_name=timezone_name,
        target_local_date=target_local_date,
        invited_user=invited_user,
    )
    initial_message = _render_initial_message(
        start_type,
        user_name=user.name,
        invited_user=invited_user,
    )
    session_config = ChatSessionConfig(
        system_prompt_override=system_prompt,
        initial_user_message=initial_message,
        extra_tools=["completion_report"],
        disabled_tools=AUTOPILOT_DISABLED_TOOLS,
    )

    session = await create_chat_session(
        user.id,
        start_type=start_type,
        execution_tag=execution_tag,
        session_config=session_config,
        initial_messages=[ChatMessage(role="user", content=initial_message)],
    )
    await _enqueue_session_turn(
        session,
        message=initial_message,
        tool_name="autopilot_dispatch",
    )
    return session


# --------------- cohort helpers --------------- #


async def _try_create_invite_cta_session(
    user: User,
    *,
    invited_user: InvitedUserRecord | None,
    now_utc: datetime,
    timezone_name: str,
    invite_cta_start: date,
    invite_cta_delay: timedelta,
) -> bool:
    if invited_user is None:
        return False
    if invited_user.status != prisma.enums.InvitedUserStatus.INVITED:
        return False
    if invited_user.created_at.date() < invite_cta_start:
        return False
    if invited_user.created_at > now_utc - invite_cta_delay:
        return False
    if await _session_exists_for_execution_tag(user.id, get_invite_cta_execution_tag()):
        return False

    created = await _create_autopilot_session(
        user,
        start_type=ChatSessionStartType.AUTOPILOT_INVITE_CTA,
        execution_tag=get_invite_cta_execution_tag(),
        timezone_name=timezone_name,
        invited_user=invited_user,
    )
    return created is not None


async def _try_create_nightly_session(
    user: User,
    *,
    now_utc: datetime,
    timezone_name: str,
    target_local_date: date,
) -> bool:
    if not await _user_has_recent_manual_message(
        user.id,
        now_utc - timedelta(hours=24),
    ):
        return False

    created = await _create_autopilot_session(
        user,
        start_type=ChatSessionStartType.AUTOPILOT_NIGHTLY,
        execution_tag=get_nightly_execution_tag(target_local_date),
        timezone_name=timezone_name,
        target_local_date=target_local_date,
    )
    return created is not None


async def _try_create_callback_session(
    user: User,
    *,
    callback_start: datetime,
    timezone_name: str,
) -> bool:
    if not await _user_has_session_since(user.id, callback_start):
        return False
    if await _session_exists_for_execution_tag(user.id, get_callback_execution_tag()):
        return False

    created = await _create_autopilot_session(
        user,
        start_type=ChatSessionStartType.AUTOPILOT_CALLBACK,
        execution_tag=get_callback_execution_tag(),
        timezone_name=timezone_name,
    )
    return created is not None


# --------------- dispatch --------------- #


async def _dispatch_nightly_copilot() -> int:
    now_utc = datetime.now(UTC)
    bucket_end = _bucket_end_for_now(now_utc)
    bucket_start = bucket_end - timedelta(minutes=30)
    callback_start = datetime.combine(
        settings.config.nightly_copilot_callback_start_date,
        time.min,
        tzinfo=UTC,
    )
    invite_cta_start = settings.config.nightly_copilot_invite_cta_start_date
    invite_cta_delay = timedelta(
        hours=settings.config.nightly_copilot_invite_cta_delay_hours
    )

    # Paginate user list to avoid loading the entire table into memory.
    created_count = 0
    cursor: str | None = None
    while True:
        batch = await user_db().list_users(
            limit=DISPATCH_BATCH_SIZE,
            cursor=cursor,
        )
        if not batch:
            break

        user_ids = [user.id for user in batch]
        invites = await invited_user_db().list_invited_users_for_auth_users(user_ids)
        invites_by_user_id = {
            invite.auth_user_id: invite for invite in invites if invite.auth_user_id
        }

        for user in batch:
            if not await is_feature_enabled(
                Flag.NIGHTLY_COPILOT, user.id, default=False
            ):
                continue

            timezone_name = _resolve_timezone_name(user.timezone)
            target_local_date = _crosses_local_midnight(
                bucket_start,
                bucket_end,
                timezone_name,
            )
            if target_local_date is None:
                continue

            invited_user = invites_by_user_id.get(user.id)
            if await _try_create_invite_cta_session(
                user,
                invited_user=invited_user,
                now_utc=now_utc,
                timezone_name=timezone_name,
                invite_cta_start=invite_cta_start,
                invite_cta_delay=invite_cta_delay,
            ):
                created_count += 1
                continue

            if await _try_create_nightly_session(
                user,
                now_utc=now_utc,
                timezone_name=timezone_name,
                target_local_date=target_local_date,
            ):
                created_count += 1
                continue

            if await _try_create_callback_session(
                user,
                callback_start=callback_start,
                timezone_name=timezone_name,
            ):
                created_count += 1

        cursor = batch[-1].id if len(batch) == DISPATCH_BATCH_SIZE else None
        if cursor is None:
            break

    return created_count


async def dispatch_nightly_copilot() -> int:
    return await _dispatch_nightly_copilot()


async def trigger_autopilot_session_for_user(
    user_id: str,
    *,
    start_type: ChatSessionStartType,
) -> ChatSession:
    allowed_start_types = {
        ChatSessionStartType.AUTOPILOT_INVITE_CTA,
        ChatSessionStartType.AUTOPILOT_NIGHTLY,
        ChatSessionStartType.AUTOPILOT_CALLBACK,
    }
    if start_type not in allowed_start_types:
        raise ValueError(f"Unsupported autopilot start type: {start_type}")

    try:
        user = await user_db().get_user_by_id(user_id)
    except ValueError as exc:
        raise LookupError(str(exc)) from exc

    invites = await invited_user_db().list_invited_users_for_auth_users([user_id])
    invited_user = invites[0] if invites else None
    timezone_name = _resolve_timezone_name(user.timezone)
    target_local_date = None
    if start_type == ChatSessionStartType.AUTOPILOT_NIGHTLY:
        try:
            tz = ZoneInfo(timezone_name)
        except (KeyError, Exception):
            tz = ZoneInfo("UTC")
        target_local_date = datetime.now(UTC).astimezone(tz).date()

    session = await _create_autopilot_session(
        user,
        start_type=start_type,
        execution_tag=_get_manual_trigger_execution_tag(start_type),
        timezone_name=timezone_name,
        target_local_date=target_local_date,
        invited_user=invited_user,
    )
    if session is None:
        raise ValueError("Failed to create autopilot session")

    return session
