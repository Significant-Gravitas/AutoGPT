"""Chat-turn orchestration for the platform bot bridge."""

import logging
from uuid import uuid4

from backend.api.features.store.exceptions import VirusDetectedError, VirusScanError
from backend.copilot import stream_registry
from backend.copilot.executor.utils import enqueue_copilot_turn
from backend.copilot.model import (
    ChatMessage,
    ChatSession,
    append_and_save_message,
    create_chat_session,
    get_chat_session,
    get_user_sessions,
)
from backend.data.db_accessors import platform_linking_db, workspace_db
from backend.util.exceptions import DuplicateChatMessageError, NotFoundError
from backend.util.workspace import WorkspaceManager

from .models import (
    BotChatRequest,
    ChatSessionSummary,
    ChatTurnHandle,
    ListUserChatsResponse,
    Platform,
    WorkspaceUploadRequest,
    WorkspaceUploadResult,
)

logger = logging.getLogger(__name__)

CHAT_TOOL_CALL_ID = "chat_stream"
CHAT_TOOL_NAME = "chat"


async def _resolve_owner(
    platform: str, platform_server_id: str | None, platform_user_id: str
) -> str:
    """Resolve the AutoGPT user that owns a platform conversation.

    Server context → server owner. DM context → the DM-linked user.
    """
    db = platform_linking_db()

    if platform_server_id:
        owner = await db.find_server_link_owner(platform, platform_server_id)
        if owner is None:
            raise NotFoundError("This server is not linked to an AutoGPT account.")
        return owner

    owner = await db.find_user_link_owner(platform, platform_user_id)
    if owner is None:
        raise NotFoundError("Your DMs are not linked to an AutoGPT account.")
    return owner


async def resolve_chat_owner(request: BotChatRequest) -> str:
    """Return the AutoGPT user ID that owns the platform conversation."""
    return await _resolve_owner(
        request.platform.value,
        request.platform_server_id,
        request.platform_user_id,
    )


async def upload_workspace_file(
    request: WorkspaceUploadRequest,
) -> WorkspaceUploadResult:
    """Store a user-attached file in the conversation owner's workspace.

    Runs the same machinery as the web upload endpoint
    (``WorkspaceManager.write_file`` → ClamAV scan → storage), so AutoPilot can
    read the file during the turn. Failures map to a stable ``error`` code
    rather than raising, so one bad file doesn't sink the whole message.
    """
    owner_user_id = await _resolve_owner(
        request.platform.value,
        request.platform_server_id,
        request.platform_user_id,
    )
    # Reduce the filename to its basename so a (possibly hostile) client can't
    # traverse the workspace path or leak ".."/separators into the storage
    # backend. Workspace paths are POSIX, so split on "/" (after normalising
    # backslashes) rather than os.path. Reject "."/".." which survive that.
    safe_name = request.filename.replace("\\", "/").rsplit("/", 1)[-1]
    if safe_name in {"", ".", ".."}:
        safe_name = "file"
    try:
        workspace = await workspace_db().get_or_create_workspace(owner_user_id)
        # Session-scoped, exactly like the web upload endpoint: the file lands
        # at /sessions/<session_id>/<name> so AutoPilot reads it during the
        # turn. The caller resolves the session before uploading (see
        # ensure_chat_session).
        manager = WorkspaceManager(owner_user_id, workspace.id, request.session_id)
        stored = await manager.write_file(
            content=request.content,
            filename=safe_name,
            mime_type=request.mime_type,
            metadata={"origin": "user-upload"},
        )
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
    owner_user_id: str, session_id: str | None, source_platform: str
) -> ChatSession:
    """Reuse the bot's cached session, or start a fresh one.

    The bot caches session IDs across turns and a cached session can be
    deleted from the web app in between — fall back to a new one so the
    conversation self-heals instead of erroring.
    """
    session = None
    if session_id:
        session = await get_chat_session(session_id, owner_user_id)
    if session is None:
        session = await create_chat_session(
            owner_user_id, dry_run=False, source_platform=source_platform
        )
    return session


async def ensure_chat_session(
    platform: Platform,
    platform_user_id: str,
    platform_server_id: str | None,
    session_id: str | None,
) -> str:
    """Resolve (or create) the session for a bot conversation, return its ID.

    Called before uploading attachments so they can be written into the
    session folder — mirroring the web UI, which uploads into an already-open
    session so files land at /sessions/<id>/ where AutoPilot reads them.
    """
    owner_user_id = await _resolve_owner(
        platform.value, platform_server_id, platform_user_id
    )
    session = await _resolve_or_create_session(
        owner_user_id, session_id, platform.value.lower()
    )
    return session.session_id


async def start_chat_turn(request: BotChatRequest) -> ChatTurnHandle:
    """Prepare a copilot turn; caller subscribes via the returned handle.

    ``subscribe_from="0-0"`` on the handle means a late subscriber replays
    the full stream (Redis Streams, not pub/sub).
    """
    owner_user_id = await resolve_chat_owner(request)
    session = await _resolve_or_create_session(
        owner_user_id, request.session_id, request.platform.value.lower()
    )
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

    await stream_registry.create_session(
        session_id=session_id,
        user_id=owner_user_id,
        tool_call_id=CHAT_TOOL_CALL_ID,
        tool_name=CHAT_TOOL_NAME,
        turn_id=turn_id,
    )

    await enqueue_copilot_turn(
        session_id=session_id,
        user_id=owner_user_id,
        message=request.message,
        turn_id=turn_id,
        is_user_message=True,
        file_ids=request.file_ids or None,
    )

    logger.info(
        "Bot chat turn started: %s (server %s, session %s, turn %s, owner ...%s)",
        request.platform.value,
        request.platform_server_id or "DM",
        session_id,
        turn_id,
        owner_user_id[-8:],
    )

    return ChatTurnHandle(
        session_id=session_id,
        turn_id=turn_id,
        user_id=owner_user_id,
    )


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

    sessions, total = await get_user_sessions(
        owner_user_id, limit=safe_limit, offset=safe_offset
    )
    return ListUserChatsResponse(
        sessions=[
            ChatSessionSummary(
                session_id=s.session_id,
                title=s.title,
                updated_at=s.updated_at,
            )
            for s in sessions
        ],
        total=total,
    )
