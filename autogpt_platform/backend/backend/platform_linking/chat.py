"""Chat-turn orchestration for the platform bot bridge."""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Awaitable, TypeVar
from uuid import uuid4

from backend.api.features.store.exceptions import VirusDetectedError, VirusScanError
from backend.copilot.active_turns import (
    ConcurrentTurnLimitError,
    TurnSlot,
    acquire_turn_slot,
    get_inflight_turn_limit,
)
from backend.copilot.config import ChatConfig, CopilotLLMModel
from backend.copilot.executor.utils import dispatch_turn
from backend.copilot.model import (
    ChatMessage,
    ChatSession,
    append_and_save_message,
    create_chat_session,
    get_chat_session,
    get_user_sessions,
)
from backend.copilot.rate_limit import RateLimitExceeded, RateLimitUnavailable
from backend.copilot.transports import resolve_default_chat_route
from backend.copilot.tree import root_envelope
from backend.copilot.turn_queue import count_inflight_turns
from backend.data.db_accessors import orgs_db, platform_linking_db, workspace_db
from backend.data.entitlement import is_user_paywalled
from backend.data.entitlement_limits import check_rate_limit, get_global_rate_limits
from backend.data.tenancy import has_live_resource_access, live_resource_access_barrier
from backend.data.user import is_user_banned
from backend.util.exceptions import DuplicateChatMessageError, NotFoundError
from backend.util.settings import Settings
from backend.util.workspace import WorkspaceManager

from .chat_experts import (
    list_owner_experts,
    require_turn_expert,
    resolve_turn_model,
    session_reusable,
)
from .models import (
    BotChatRequest,
    BotExpert,
    ChatSessionSummary,
    ChatTurnHandle,
    EnsureSessionResult,
    ListUserChatsResponse,
    Platform,
    TurnDenial,
    WorkspaceUploadRequest,
    WorkspaceUploadResult,
)

logger = logging.getLogger(__name__)

CHAT_TOOL_CALL_ID = "chat_stream"
CHAT_TOOL_NAME = "chat"
T = TypeVar("T")


class PlatformLinkLeaseGuard:
    def __init__(self, link_db, lease_id: str) -> None:
        self._link_db = link_db
        self.lease_id = lease_id

    async def validate(self) -> None:
        try:
            active = await self._link_db.is_platform_link_lease_active(self.lease_id)
        except Exception as error:
            raise NotFoundError("The platform link is no longer active.") from error
        if not active:
            raise NotFoundError("The platform link is no longer active.")

    async def _wait_for_loss(self) -> None:
        while True:
            await asyncio.sleep(0.1)
            await self.validate()

    async def run(self, action: Awaitable[T]) -> T:
        action_task = asyncio.ensure_future(action)
        loss_task = asyncio.create_task(self._wait_for_loss())
        try:
            done, _ = await asyncio.wait(
                (action_task, loss_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if loss_task in done:
                action_task.cancel()
                await asyncio.gather(action_task, return_exceptions=True)
                await loss_task
            result = await action_task
            await self.validate()
            return result
        finally:
            if not loss_task.done():
                loss_task.cancel()
            await asyncio.gather(loss_task, return_exceptions=True)


def _session_matches_scope(
    session: ChatSession | None,
    organization_id: str | None,
    team_id: str | None,
) -> bool:
    return session is not None and (
        session.organization_id,
        session.team_id,
    ) == (organization_id, team_id)


async def _default_scope(user_id: str) -> tuple[str | None, str | None]:
    return await orgs_db().get_user_default_team(user_id)


def _billing_url() -> str | None:
    """The platform billing page, or None when no frontend URL is configured
    (Discord rejects relative URLs on link buttons, so callers fall back to a
    plain-text message in that case)."""
    base = Settings().config.frontend_base_url
    return f"{base.rstrip('/')}/settings/billing" if base else None


def _unavailable_denial() -> TurnDenial:
    """The gate couldn't be evaluated (tier lookup or Redis unreachable). Fail
    closed with a retry nudge rather than letting an unmetered turn through —
    mirrors the web route's 503-on-lookup-failure behaviour."""
    return TurnDenial(
        reason="unavailable",
        message="Chat is temporarily unavailable — please try again in a moment.",
    )


async def _check_ban(user_id: str) -> TurnDenial | None:
    """Fail closed on lookup failure, as the paywall check does."""
    try:
        banned = await is_user_banned(user_id)
    except Exception:
        logger.warning(
            f"evaluate_turn_gate: ban lookup failed for ...{user_id[-8:]}",
            exc_info=True,
        )
        return _unavailable_denial()
    if not banned:
        return None
    return TurnDenial(reason="banned", message="This account has been banned.")


async def _check_paywall(user_id: str, org_id: str | None) -> TurnDenial | None:
    """Subscription paywall check — fail closed on lookup failure."""
    try:
        paywalled = await is_user_paywalled(user_id, org_id)
    except Exception:
        # Tier lookup blip (Supabase / LD). The web maps this to a 503; fail
        # closed here too so a transient error can't bypass the paywall.
        logger.warning(
            "evaluate_turn_gate: paywall lookup failed for ...%s",
            user_id[-8:],
            exc_info=True,
        )
        return _unavailable_denial()
    if not paywalled:
        return None
    billing = _billing_url()
    return TurnDenial(
        reason="paywalled",
        message=(
            "Chatting with experts requires an active subscription. "
            "Upgrade your plan to start chatting."
        ),
        button_label="Subscribe" if billing else None,
        button_url=billing,
    )


async def _check_usage_limits(user_id: str, org_id: str | None) -> TurnDenial | None:
    """Daily/weekly USD-cap check — fail closed on any lookup failure."""
    cfg = ChatConfig()
    try:
        daily, weekly, _tier = await get_global_rate_limits(
            user_id,
            cfg.daily_cost_limit_microdollars,
            cfg.weekly_cost_limit_microdollars,
            org_id,
        )
        await check_rate_limit(
            user_id=user_id,
            daily_cost_limit=daily,
            weekly_cost_limit=weekly,
            expected_tier=_tier,
            org_id=org_id,
        )
    except RateLimitExceeded as exc:
        # str(exc) already reads e.g. "You've reached your daily usage limit.
        # Resets in 5h 30m." — exactly the window + countdown we want to show.
        billing = _billing_url()
        return TurnDenial(
            reason="rate_limited",
            message=str(exc),
            button_label="Upgrade for higher limits" if billing else None,
            button_url=billing,
        )
    except RateLimitUnavailable:
        return _unavailable_denial()
    except Exception:
        # Same fail-closed treatment as the paywall lookup: an unexpected
        # error (LD, config, Redis client) must not bypass the USD caps.
        logger.warning(
            "evaluate_turn_gate: rate limit check failed for ...%s",
            user_id[-8:],
            exc_info=True,
        )
        return _unavailable_denial()
    return None


async def evaluate_turn_gate(user_id: str) -> TurnDenial | None:
    """Apply the same ban, subscription paywall and usage limits as the web UI.

    The bot enqueues turns directly into the executor, bypassing the
    route-level paywall dependency and the dispatcher's rate-limit check, so
    without this a NO_TIER / over-cap user would get unmetered free usage.
    Returns a :class:`TurnDenial` to surface, or None when the turn may run.
    A bot conversation runs in the owner's personal organization (its default
    team, or a private expert's), so the owner's own plan decides.
    """
    return (
        await _check_ban(user_id)
        or await _check_paywall(user_id, None)
        or await _check_usage_limits(user_id, None)
    )


async def resolve_owner(
    platform: str, platform_server_id: str | None, platform_user_id: str
) -> str:
    """Resolve the AutoGPT user that owns a platform conversation.

    Server context → server owner. DM context → the DM-linked user.
    """
    db = platform_linking_db()

    if platform_server_id:
        owner = await db.find_server_link_owner_for_sender(
            platform, platform_server_id, platform_user_id
        )
        if owner is None:
            raise NotFoundError(
                "Only the account owner who linked this server can use AutoGPT here."
            )
        return owner

    owner = await db.find_user_link_owner(platform, platform_user_id)
    if owner is None:
        raise NotFoundError("Your DMs are not linked to an AutoGPT account.")
    return owner


async def resolve_chat_owner(request: BotChatRequest) -> str:
    """Return the AutoGPT user ID that owns the platform conversation."""
    return await resolve_owner(
        request.platform.value,
        request.platform_server_id,
        request.platform_user_id,
    )


@asynccontextmanager
async def _live_platform_link(
    platform: str,
    platform_server_id: str | None,
    platform_user_id: str,
    expected_owner_user_id: str,
) -> AsyncIterator[PlatformLinkLeaseGuard]:
    db = platform_linking_db()
    lease_id = await db.acquire_platform_link_lease(
        platform,
        platform_server_id,
        platform_user_id,
    )
    try:
        current_owner = await resolve_owner(
            platform,
            platform_server_id,
            platform_user_id,
        )
        if current_owner != expected_owner_user_id:
            raise NotFoundError("The platform link is no longer active.")
        yield PlatformLinkLeaseGuard(db, lease_id)
    finally:
        released = await db.release_platform_link_lease(lease_id)
        if not released:
            raise NotFoundError("The platform link is no longer active.")


async def upload_workspace_file(
    request: WorkspaceUploadRequest,
) -> WorkspaceUploadResult:
    """Store a user-attached file in the conversation owner's workspace.

    Runs the same machinery as the web upload endpoint
    (``WorkspaceManager.write_file`` → ClamAV scan → storage), so Otto can
    read the file during the turn. Failures map to a stable ``error`` code
    rather than raising, so one bad file doesn't sink the whole message.
    """
    owner_user_id = await resolve_owner(
        request.platform.value,
        request.platform_server_id,
        request.platform_user_id,
    )
    org_id, team_id = await _default_scope(owner_user_id)
    if request.session_id:
        # Same guard as ensure_chat_session / start_chat_turn: the session
        # must exist and belong to the owner, and a shared-server upload must
        # never land inside an expert-scoped session's folder.
        session = await get_chat_session(request.session_id, owner_user_id)
        if not _session_matches_scope(session, org_id, team_id) or (
            request.platform_server_id is not None
            and session is not None
            and session.expert_id is not None
        ):
            raise NotFoundError("The session for the uploaded files no longer exists.")
    # Reduce the filename to its basename so a (possibly hostile) client can't
    # traverse the workspace path or leak ".."/separators into the storage
    # backend. Workspace paths are POSIX, so split on "/" (after normalising
    # backslashes) rather than os.path. Reject "."/".." which survive that.
    safe_name = request.filename.replace("\\", "/").rsplit("/", 1)[-1]
    if safe_name in {"", ".", ".."}:
        safe_name = "file"
    try:

        async def write_upload():
            async with live_resource_access_barrier(
                owner_user_id, org_id, team_id, "create"
            ) as allowed:
                if not allowed:
                    raise NotFoundError(
                        "The session for the uploaded files no longer exists."
                    )
                workspace = await workspace_db().get_or_create_workspace(owner_user_id)
                manager = WorkspaceManager(
                    owner_user_id,
                    workspace.id,
                    request.session_id,
                    organization_id=org_id,
                    team_id=team_id,
                    access="create",
                )
                return await manager.write_file(
                    content=request.content,
                    filename=safe_name,
                    mime_type=request.mime_type,
                    metadata={"origin": "user-upload"},
                    overwrite=True,
                )

        async with _live_platform_link(
            request.platform.value,
            request.platform_server_id,
            request.platform_user_id,
            owner_user_id,
        ) as link_guard:
            stored = await link_guard.run(write_upload())
    except VirusDetectedError:
        return WorkspaceUploadResult(filename=request.filename, error="virus_detected")
    except VirusScanError:
        return WorkspaceUploadResult(
            filename=request.filename, error="scan_unavailable"
        )
    except NotFoundError:
        # NotFoundError subclasses ValueError; let a missing user/workspace
        # propagate as a linking error instead of being mislabelled "rejected".
        raise
    except ValueError:
        # write_file raises ValueError for size / storage-quota limits.
        return WorkspaceUploadResult(filename=request.filename, error="rejected")
    except Exception:
        logger.exception("Workspace upload failed for %s", request.filename)
        return WorkspaceUploadResult(filename=request.filename, error="upload_failed")
    return WorkspaceUploadResult(filename=request.filename, file_id=stored.id)


async def _resolve_or_create_session(
    owner_user_id: str,
    session_id: str | None,
    source_platform: str,
    *,
    allow_expert_session: bool = True,
    expert_id: str | None = None,
) -> ChatSession:
    """Reuse the bot's cached session, or start a fresh one.

    The bot caches session IDs across turns and a cached session can be
    deleted from the web app in between — fall back to a new one so the
    conversation self-heals instead of erroring. A session bound to another
    expert than ``expert_id`` (or one this context may not reach) is replaced,
    and so is one outside the owner's default team (see ``_session_in_scope``).
    """
    org_id, team_id = await _default_scope(owner_user_id)
    session = None
    if session_id:
        session = await get_chat_session(session_id, owner_user_id)
        if session is not None and not (
            _session_in_scope(session, org_id, team_id)
            and await session_reusable(
                session, owner_user_id, expert_id, allow_expert_session
            )
        ):
            session = None
    if session is None:
        async with live_resource_access_barrier(
            owner_user_id, org_id, team_id, "create"
        ) as allowed:
            if not allowed:
                raise NotFoundError("The selected workspace is no longer available.")
            llm_auth_provider, llm_credential_id = await resolve_default_chat_route(
                owner_user_id
            )
            session = await create_chat_session(
                owner_user_id,
                dry_run=False,
                organization_id=org_id,
                team_id=team_id,
                source_platform=source_platform,
                llm_auth_provider=llm_auth_provider,
                llm_credential_id=llm_credential_id,
                expert_id=expert_id,
            )
            if not _session_in_scope(session, org_id, team_id):
                raise NotFoundError("The selected workspace is no longer available.")
    elif not await has_live_resource_access(
        owner_user_id, session.organization_id, session.team_id, "execute"
    ):
        raise NotFoundError("The selected workspace is no longer available.")
    return session


def _session_in_scope(
    session: ChatSession | None,
    organization_id: str | None,
    team_id: str | None,
) -> bool:
    """Whether a bot conversation in the owner's default team may use this
    session: one of that team's own, or an expert session, which lives where
    its expert was hired (see ``_turn_tenancy``) and is checked for live
    access there."""
    return session is not None and (
        session.expert_id is not None
        or _session_matches_scope(session, organization_id, team_id)
    )


async def ensure_chat_session(
    platform: Platform,
    platform_user_id: str,
    platform_server_id: str | None,
    session_id: str | None,
    expert_id: str | None = None,
) -> EnsureSessionResult:
    """Resolve (or create) the session for a bot conversation.

    Called before uploading attachments so they can be written into the
    session folder — mirroring the web UI, which uploads into an already-open
    session so files land at /sessions/<id>/ where Otto reads them.

    Evaluates the turn gate first: a capped/paywalled user gets the denial
    back *before* any file is scanned or stored (the caller renders it and
    skips the upload + turn). ``start_chat_turn`` re-checks the gate as the
    authoritative enforcement point; this early check exists purely to spare
    the wasted upload.
    """
    owner_user_id = await resolve_owner(
        platform.value, platform_server_id, platform_user_id
    )

    async def ensure_linked_session() -> EnsureSessionResult:
        denial = await evaluate_turn_gate(owner_user_id)
        if denial is not None:
            logger.info(
                "Bot session request denied (%s) for user ...%s",
                denial.reason,
                owner_user_id[-8:],
            )
            return EnsureSessionResult(denial=denial)
        await require_turn_expert(owner_user_id, expert_id, platform_server_id)
        session = await _resolve_or_create_session(
            owner_user_id,
            session_id,
            platform.value.lower(),
            allow_expert_session=platform_server_id is None,
            expert_id=expert_id,
        )
        return EnsureSessionResult(session_id=session.session_id)

    async with _live_platform_link(
        platform.value,
        platform_server_id,
        platform_user_id,
        owner_user_id,
    ) as link_guard:
        return await link_guard.run(ensure_linked_session())


async def start_chat_turn(request: BotChatRequest) -> ChatTurnHandle:
    """Prepare a copilot turn; caller subscribes via the returned handle.

    ``subscribe_from="0-0"`` on the handle means a late subscriber replays
    the full stream (Redis Streams, not pub/sub).
    """
    owner_user_id = await resolve_chat_owner(request)

    async def start_linked_turn() -> ChatTurnHandle:
        # Gate before touching the session / enqueue so a refused turn neither
        # persists a message nor runs the LLM.
        denial = await evaluate_turn_gate(owner_user_id)
        if denial is not None:
            logger.info(
                "Bot chat turn denied (%s) for user ...%s",
                denial.reason,
                owner_user_id[-8:],
            )
            return ChatTurnHandle(
                session_id="", turn_id="", user_id=owner_user_id, denial=denial
            )

        # Bot turns share the owner's in-flight cap with every other turn source.
        inflight_cap = get_inflight_turn_limit()
        if await count_inflight_turns(owner_user_id) >= inflight_cap:
            return _busy_handle(owner_user_id, inflight_cap)

        await require_turn_expert(
            owner_user_id, request.expert_id, request.platform_server_id
        )
        session = await _turn_session(request, owner_user_id)
        session_id = session.session_id
        model, notice = await resolve_turn_model(owner_user_id, request.model, session)

        async with live_resource_access_barrier(
            owner_user_id, session.organization_id, session.team_id, "execute"
        ) as allowed:
            if not allowed:
                raise NotFoundError("The selected workspace is no longer available.")
            try:
                async with acquire_turn_slot(
                    owner_user_id, session_id, capacity=inflight_cap
                ) as slot:
                    turn_id = await _persist_and_dispatch(
                        request, slot, session, owner_user_id, model
                    )
            except ConcurrentTurnLimitError:
                return _busy_handle(owner_user_id, inflight_cap)

        logger.info(
            f"Bot chat turn started: {request.platform.value} "
            f"(server {request.platform_server_id or 'DM'}, session {session_id}, "
            f"turn {turn_id}, owner ...{owner_user_id[-8:]})"
        )

        return ChatTurnHandle(
            session_id=session_id,
            turn_id=turn_id,
            user_id=owner_user_id,
            notice=notice,
        )

    async with _live_platform_link(
        request.platform.value,
        request.platform_server_id,
        request.platform_user_id,
        owner_user_id,
    ) as link_guard:
        return await link_guard.run(start_linked_turn())


async def _turn_session(request: BotChatRequest, owner_user_id: str) -> ChatSession:
    """The session a bot turn runs in."""
    if not (request.file_ids and request.session_id):
        return await _resolve_or_create_session(
            owner_user_id,
            request.session_id,
            request.platform.value.lower(),
            allow_expert_session=request.platform_server_id is None,
            expert_id=request.expert_id,
        )
    # Attachment turn: the files were already uploaded into this session
    # (ensure_chat_session ran first). It must still exist — silently
    # switching to a new session would orphan those files. Fail instead.
    session = await get_chat_session(request.session_id, owner_user_id)
    org_id, team_id = await _default_scope(owner_user_id)
    if (
        session is None
        or not _session_in_scope(session, org_id, team_id)
        or not await session_reusable(
            session,
            owner_user_id,
            request.expert_id,
            allow_expert_session=request.platform_server_id is None,
        )
    ):
        raise NotFoundError("The session for the uploaded files no longer exists.")
    return session


async def _persist_and_dispatch(
    request: BotChatRequest,
    slot: TurnSlot,
    session: ChatSession,
    owner_user_id: str,
    model: CopilotLLMModel | None,
) -> str:
    """Save the message and dispatch its turn inside the held slot; returns the
    turn id."""
    session_id = session.session_id
    # Persist the user message before enqueueing, mirroring the REST chat
    # endpoint — otherwise the executor runs against empty history.
    is_duplicate = (
        await append_and_save_message(
            session_id, ChatMessage(role="user", content=request.message)
        )
    ) is None
    if is_duplicate:
        # Matches REST chat behaviour: skip create_session + enqueue so we
        # don't create an orphan stream with no producer. Caller subscribes
        # to the in-flight turn via its own retry logic, or drops.
        logger.info(
            "Duplicate bot message for session %s (platform %s, user ...%s)",
            session_id,
            request.platform.value,
            owner_user_id[-8:],
        )
        raise DuplicateChatMessageError("Message already in flight.")

    turn_id = str(uuid4())
    org_id, team_id = await _turn_tenancy(owner_user_id, session)
    await dispatch_turn(
        slot,
        session_id=session_id,
        user_id=owner_user_id,
        turn_id=turn_id,
        message=request.message,
        tool_call_id=CHAT_TOOL_CALL_ID,
        tool_name=CHAT_TOOL_NAME,
        is_user_message=True,
        organization_id=org_id,
        team_id=team_id,
        file_ids=request.file_ids or None,
        model=model,
        llm_auth_provider=session.metadata.llm_auth_provider,
        llm_credential_id=session.metadata.llm_credential_id,
        # Roots its own tree, and born tainted: the message was written on a
        # chat platform by someone who need not be the account owner, so
        # anything this turn spawns inherits the bit.
        envelope=root_envelope(turn_id, tainted=True, session_id=session_id),
    )
    return turn_id


def _busy_handle(owner_user_id: str, inflight_cap: int) -> ChatTurnHandle:
    return ChatTurnHandle(
        session_id="",
        turn_id="",
        user_id=owner_user_id,
        denial=TurnDenial(
            reason="rate_limited",
            message=(
                f"The linked account is already running its limit of {inflight_cap} "
                "tasks at once. Try again once one of them finishes."
            ),
        ),
    )


async def _turn_tenancy(
    owner_user_id: str, session: ChatSession
) -> tuple[str | None, str | None]:
    """A turn runs in its session's org/team, the scope the turn was checked
    in: the one an expert session is pinned to (the owner's personal one), or
    the owner's default team any other session was matched against. A session
    from before org tagging runs in the owner's default team."""
    if session.organization_id:
        return session.organization_id, session.team_id
    return await orgs_db().get_user_default_team(owner_user_id)


async def list_user_experts(
    platform: Platform, platform_user_id: str, platform_server_id: str | None
) -> list[BotExpert]:
    """The experts a bot conversation can route a turn to.

    The owner resolves exactly like a chat turn. A server gets none: expert
    sessions are DM-only, and listing the server owner's hires to every
    member would leak them for nothing.
    """
    owner_user_id = await resolve_owner(
        platform.value, platform_server_id, platform_user_id
    )
    if platform_server_id is not None:
        return []
    return await list_owner_experts(owner_user_id)


LIST_USER_CHATS_MAX_LIMIT = 25


async def list_user_chats(
    platform: Platform,
    platform_user_id: str,
    limit: int = 25,
    offset: int = 0,
) -> ListUserChatsResponse:
    """List a DM-linked user's own copilot chats, most recent first.

    Ownership is resolved server-side from the user's own DM link — never a
    server — so a caller can only ever see their own conversations.
    """
    owner_user_id = await platform_linking_db().find_user_link_owner(
        platform.value, platform_user_id
    )
    if owner_user_id is None:
        raise NotFoundError("Your DMs are not linked to an AutoGPT account.")

    # Clamp pagination — negative values would crash the DB driver, and an
    # unbounded `limit` would fan out into a giant query. The cap also lines
    # up with Discord's 25-option select-menu limit used by /resume.
    safe_limit = max(0, min(limit, LIST_USER_CHATS_MAX_LIMIT))
    safe_offset = max(0, offset)

    async def list_linked_chats() -> ListUserChatsResponse:
        org_id, team_id = await _default_scope(owner_user_id)
        async with live_resource_access_barrier(
            owner_user_id, org_id, team_id, "view"
        ) as allowed:
            if not allowed:
                raise NotFoundError("The selected workspace is no longer available.")
            sessions, total = await get_user_sessions(
                owner_user_id,
                limit=safe_limit,
                offset=safe_offset,
                organization_id=org_id,
                team_id=team_id,
                team_ids=[team_id] if team_id is not None else [],
            )
            visible_sessions = [
                session
                for session in sessions
                if (session.organization_id, session.team_id) == (org_id, team_id)
            ]
            if len(visible_sessions) != len(sessions):
                total = len(visible_sessions)
            return ListUserChatsResponse(
                sessions=[
                    ChatSessionSummary(
                        session_id=s.session_id,
                        title=s.title,
                        updated_at=s.updated_at,
                    )
                    for s in visible_sessions
                ],
                total=total,
            )

    async with _live_platform_link(
        platform.value,
        None,
        platform_user_id,
        owner_user_id,
    ) as link_guard:
        return await link_guard.run(list_linked_chats())
