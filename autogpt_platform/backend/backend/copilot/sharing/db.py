"""Data layer for chat session sharing.

Mirrors :mod:`backend.data.execution`'s share helpers but for
:class:`prisma.models.ChatSession`.  Sharing a chat also opts-in a
caller-selected set of :class:`AgentGraphExecution` rows so the public
viewer can drill into the underlying agent run; the cascade rules are
encoded here (see :func:`disable_chat_session_share`).
"""

import json
import logging
from datetime import UTC, datetime
from typing import Any

import sentry_sdk
from prisma.enums import SharedVia
from prisma.errors import ForeignKeyViolationError, UniqueViolationError
from prisma.models import AgentGraph, AgentGraphExecution, ChatLinkedShare
from prisma.models import ChatMessage as PrismaChatMessage
from prisma.models import ChatSession as PrismaChatSession
from prisma.models import (
    SharedChatFile,
    SharedExecutionFile,
    UserWorkspace,
    UserWorkspaceFile,
)

from backend.blocks import get_block
from backend.blocks._base import BlockType
from backend.copilot.db import get_chat_messages_paginated
from backend.copilot.model import ChatSessionInfo
from backend.copilot.sharing.models import (
    ChatShareState,
    SharedChatLinkedExecution,
    SharedChatMessagesPage,
    SharedChatSession,
    sanitize_chat_message,
    sanitize_chat_session,
)
from backend.data.block import BlockInput, CompletedBlockOutput
from backend.data.db import transaction
from backend.data.sharing.tokens import generate_share_token
from backend.data.sharing.workspace_refs import extract_workspace_file_ids
from backend.util import type as type_utils

logger = logging.getLogger(__name__)

# Linked-execution discovery scans assistant tool responses (role="tool"
# rows) for run_agent tool payloads that reference an execution.
#
# Two response shapes can result from a run_agent invocation:
#   * ExecutionStartedResponse — async / still-running / REVIEW;
#     ``execution_id`` lives at the top level of the JSON payload.
#   * AgentOutputResponse — sync-complete (``wait_for_result``); the
#     execution_id lives nested under ``execution.execution_id``.
#   * ErrorResponse — failed / terminated actual runs; execution_id is
#     optional because generic tool errors do not have a run to link.
# These shapes are emitted by ``backend.copilot.tools.models``.
_EXECUTION_STARTED_TYPE = "execution_started"
_AGENT_OUTPUT_TYPE = "agent_output"
_ERROR_TYPE = "error"
_SCHEDULED_STATUS = "SCHEDULED"


# ---------- Enable / disable -------------------------------------------------


async def enable_chat_session_share(
    session_id: str,
    user_id: str,
    auto_share_executions: bool,
) -> str:
    """Enable sharing on a chat session.

    Generates a fresh token, refreshes the file allowlist from the
    session's messages, and — when ``auto_share_executions`` is True —
    auto-links every ``run_agent`` execution referenced by the chat's
    tool messages.  The flag is persisted on the session so subsequent
    ``run_agent`` invocations (handled by
    :func:`link_new_execution_to_chat_share`) keep the chat share in
    sync without requiring the owner to revisit the dialog.

    All writes happen inside a single Prisma transaction so a crash
    mid-flow cannot leave the chat publicly readable with a missing
    file allowlist or an execution flipped to ``CHAT_LINK`` shared
    state with no parent chat linkage.

    Returns the share token.

    Raises ValueError if the session does not belong to *user_id*.
    """
    session = await PrismaChatSession.prisma().find_first(
        where={"id": session_id, "userId": user_id},
    )
    if not session:
        raise ValueError(f"Chat session {session_id} not found for user")

    if auto_share_executions:
        execution_ids = list(
            await _collect_execution_ids_from_messages(session_id=session_id)
        )
        # Only auto-link executions the caller still owns.  Runs the
        # caller doesn't own (deleted, cross-user) are silently skipped
        # rather than failing the whole share — they shouldn't have
        # been referenced in the chat in the first place, but defending
        # against that here keeps share-enable robust.
        executions_to_link = (
            await AgentGraphExecution.prisma().find_many(
                where={
                    "id": {"in": execution_ids},
                    "userId": user_id,
                    "isDeleted": False,
                },
            )
            if execution_ids
            else []
        )
    else:
        executions_to_link = []

    share_token = generate_share_token()
    now = datetime.now(UTC)

    async with transaction() as tx:
        # Cascade-revoke previously-linked CHAT_LINK executions BEFORE
        # wiping the join table.  Otherwise a re-share with a different
        # ``autoShareExecutions`` value (or just a re-share to mint a
        # fresh token) leaves the old per-execution tokens publicly
        # active with no linkage row to reach them via — orphan public
        # shares.  Same guard ``disable_chat_session_share`` runs.
        await _cascade_revoke_chat_linked_executions(session_id=session_id, tx=tx)

        # Clear stale allowlist + linkage rows before re-keying the token,
        # so an old token + new linkage never coexist.
        await SharedChatFile.prisma(tx).delete_many(where={"sessionId": session_id})
        await ChatLinkedShare.prisma(tx).delete_many(where={"sessionId": session_id})

        await _build_shared_chat_files(
            session_id=session_id,
            share_token=share_token,
            user_id=user_id,
            tx=tx,
        )

        await _link_executions_to_share(
            session_id=session_id,
            executions=executions_to_link,
            shared_at=now,
            tx=tx,
        )

        # Session update goes LAST inside the transaction so the chat
        # only becomes publicly readable after the allowlist + linkage
        # are committed.
        await PrismaChatSession.prisma(tx).update(
            where={"id": session_id},
            data={
                "isShared": True,
                "shareToken": share_token,
                "sharedAt": now,
                "autoShareExecutions": auto_share_executions,
            },
        )

    return share_token


async def link_new_execution_to_chat_share(
    session_id: str,
    execution_id: str,
) -> None:
    """Auto-link a freshly-completed execution into an already-shared chat.

    Hook called by the ``run_agent`` tool after the tool result is
    persisted.  No-op when the chat is not shared, or shared without
    ``autoShareExecutions=True``.  Idempotent — repeated calls for the
    same (session, execution) pair don't double-link.

    ``run_agent`` runs inside the CoPilotExecutor worker process, which
    has a different Prisma client lifecycle than the API server.  Any
    Prisma failure (e.g. the worker's client not yet connected) is
    caught and logged: a sharing hook must never crash the tool itself
    — a failure here surfaces as ``ClientNotConnectedError`` inside the
    Claude SDK as an orphan tool_use, which the SDK turns into "The
    model returned an empty response."  Skipped links are recovered the
    next time the owner re-shares the chat (enable-time backfill walks
    every existing run via ``_collect_execution_ids_from_messages``).
    """
    try:
        session = await PrismaChatSession.prisma().find_unique(where={"id": session_id})
        if not session or not session.isShared or not session.autoShareExecutions:
            return

        now = datetime.now(UTC)
        async with transaction() as tx:
            # Re-check share state inside the transaction.  A concurrent
            # ``disable_chat_session_share`` could have flipped
            # ``isShared`` between the outer check above and the tx
            # start; without this guard we'd create a ChatLinkedShare
            # row pointing at a no-longer-shared session and possibly
            # flip the execution to ``CHAT_LINK`` shared — orphan
            # public state.  The pre-fetched ``session`` snapshot is
            # only used for the early-return fast path; the
            # transactional read is the authoritative one.
            session_tx = await PrismaChatSession.prisma(tx).find_unique(
                where={"id": session_id}
            )
            if (
                not session_tx
                or not session_tx.isShared
                or not session_tx.autoShareExecutions
            ):
                return

            execution = await AgentGraphExecution.prisma(tx).find_first(
                where={
                    "id": execution_id,
                    "userId": session_tx.userId,
                    "isDeleted": False,
                },
            )
            if execution is None:
                return

            existing = await ChatLinkedShare.prisma(tx).find_first(
                where={"sessionId": session_id, "executionId": execution_id},
            )
            if existing is not None:
                return

            await _link_executions_to_share(
                session_id=session_id,
                executions=[execution],
                shared_at=now,
                tx=tx,
            )
    except Exception as exc:
        # Best-effort hook: a sharing failure must never break the
        # underlying run_agent tool.  See note above on the empty-
        # completion failure mode this guards against.  Pair the log
        # with a Sentry capture so the no-crash guarantee doesn't
        # hide real operational regressions (DB outage, schema drift)
        # behind a warn-only signal.
        logger.warning(
            "link_new_execution_to_chat_share failed for session=%s execution=%s; "
            "owner can recover via stop+reshare or an explicit re-enable",
            session_id,
            execution_id,
            exc_info=True,
        )
        sentry_sdk.capture_exception(exc)


async def disable_chat_session_share(session_id: str, user_id: str) -> None:
    """Revoke sharing on a chat session and cascade-revoke linked executions.

    Cascade rule: an execution share is cleared only when (a) it was
    enabled via a chat share (``sharedVia == CHAT_LINK``) AND (b) no
    other chat session still has a ``ChatLinkedShare`` row referencing
    it.  User-initiated execution shares survive untouched, and an
    execution opted-in by multiple chat shares stays shared as long as
    at least one chat share still references it.
    """
    session = await PrismaChatSession.prisma().find_first(
        where={"id": session_id, "userId": user_id},
    )
    if not session:
        raise ValueError(f"Chat session {session_id} not found for user")

    async with transaction() as tx:
        # Flip the session to ``isShared=False`` FIRST so any concurrent
        # ``link_new_execution_to_chat_share`` racing this transaction
        # sees the new state on its own in-tx re-read and bails before
        # creating an orphan ``ChatLinkedShare`` row.  The cascade
        # bookkeeping (read linkages → revoke executions → delete
        # rows) all runs inside the same transaction so the window for
        # a phantom row is reduced to the gap between two committed
        # transactions — which the hook's in-tx re-check covers.
        await PrismaChatSession.prisma(tx).update(
            where={"id": session_id},
            data={
                "isShared": False,
                "shareToken": None,
                "sharedAt": None,
                # Reset the auto-share preference so the next time the
                # owner re-shares this chat the modal opens in its
                # default state (toggle off until they flip it).
                "autoShareExecutions": False,
            },
        )

        # Walk the linkage table and disable only the chat-derived shares
        # that aren't still referenced by another chat session's linkage.
        # Reading INSIDE the tx ensures the pre-flip session update is
        # visible and any concurrent linkage commit lands either fully
        # before this read (and gets cascaded) or fully after (and is
        # blocked by the in-tx session re-check).
        await _cascade_revoke_chat_linked_executions(session_id=session_id, tx=tx)

        await ChatLinkedShare.prisma(tx).delete_many(where={"sessionId": session_id})
        await SharedChatFile.prisma(tx).delete_many(where={"sessionId": session_id})


# ---------- Public read path -------------------------------------------------


async def get_chat_session_by_share_token(
    share_token: str,
) -> SharedChatSession | None:
    """Look up a shared session header (no messages, no auth)."""
    session = await PrismaChatSession.prisma().find_first(
        where={"shareToken": share_token, "isShared": True},
    )
    if not session:
        return None

    linked = await _resolve_linked_executions(session_id=session.id)
    return sanitize_chat_session(
        ChatSessionInfo.from_db(session),
        linked_executions=linked,
        shared_at=session.sharedAt,
    )


async def get_shared_chat_messages_paginated(
    share_token: str,
    *,
    limit: int = 50,
    before_sequence: int | None = None,
) -> SharedChatMessagesPage | None:
    """Paginate messages for a shared session.

    Returns ``None`` when the token is unknown or sharing has been
    disabled (uniform with :func:`get_chat_session_by_share_token`).
    """
    session = await PrismaChatSession.prisma().find_first(
        where={"shareToken": share_token, "isShared": True},
    )
    if not session:
        return None

    page = await get_chat_messages_paginated(
        session_id=session.id,
        limit=limit,
        before_sequence=before_sequence,
        # No user scoping — the share token already authorises the read.
    )
    if page is None:
        return None

    return SharedChatMessagesPage(
        messages=[sanitize_chat_message(m) for m in page.messages],
        has_more=page.has_more,
        oldest_sequence=page.oldest_sequence,
    )


async def get_shared_chat_file(share_token: str, file_id: str) -> str | None:
    """Allowlist lookup for the public file-download endpoint.

    Returns the owning ``session_id`` if the file is allowlisted,
    ``None`` otherwise.  Uniform None for every failure mode prevents
    timing-based enumeration of valid file IDs.

    Falls back to a live re-scan of the chat's messages when the
    static allowlist misses — files uploaded AFTER the share was
    enabled (or referenced by later assistant messages) would
    otherwise 404 forever, even though the file is clearly part of
    the shared conversation.  The re-scan re-uses the same workspace-
    ownership guard ``_build_shared_chat_files`` enforces at enable
    time, so a malicious tool output trying to reference a foreign
    workspace file still gets rejected.  On hit we backfill a
    SharedChatFile row so subsequent downloads short-circuit.
    """
    record = await SharedChatFile.prisma().find_first(
        where={"shareToken": share_token, "fileId": file_id}
    )
    if record is not None:
        return record.sessionId

    # Slow path: chat-share allowlists are built once at enable time,
    # but the chat keeps accumulating messages.  Re-derive ownership
    # from the live messages for this token before giving up.
    session = await PrismaChatSession.prisma().find_first(
        where={"shareToken": share_token, "isShared": True},
    )
    if session is None:
        return None
    if not await _file_referenced_in_session(session_id=session.id, file_id=file_id):
        return None
    workspace = await UserWorkspace.prisma().find_unique(
        where={"userId": session.userId}
    )
    if workspace is None:
        return None
    file = await UserWorkspaceFile.prisma().find_first(
        where={"id": file_id, "workspaceId": workspace.id, "isDeleted": False}
    )
    if file is None:
        return None
    # Backfill the allowlist so repeat downloads skip the re-scan.
    try:
        await SharedChatFile.prisma().create(
            data={
                "sessionId": session.id,
                "fileId": file_id,
                "shareToken": share_token,
            }
        )
    except (UniqueViolationError, ForeignKeyViolationError):
        # Concurrent backfill or file deleted between checks — fall
        # through; the file is still ownership-validated above so
        # the download is safe to authorise this once.
        pass
    return session.id


async def _file_referenced_in_session(*, session_id: str, file_id: str) -> bool:
    """True iff *file_id* appears in any of *session_id*'s messages.

    Scans ``content``, ``toolCalls``, and ``functionCall`` of every
    message using :func:`extract_workspace_file_ids` — same shape the
    enable-time allowlist build uses, so a message that references the
    file via ``workspace://``, ``[Attached files] file_id=``, or JSON
    tool output all qualify.
    """
    candidate_rows = await PrismaChatMessage.prisma().find_many(
        where={"sessionId": session_id, "content": {"contains": file_id}},
    )
    for row in candidate_rows:
        if row.content is not None and file_id in extract_workspace_file_ids(
            row.content
        ):
            return True

    rows = await PrismaChatMessage.prisma().find_many(
        where={"sessionId": session_id},
    )
    for row in rows:
        if row.content is not None:
            if file_id in extract_workspace_file_ids(row.content):
                return True
        if row.toolCalls is not None:
            if file_id in extract_workspace_file_ids(_json_or_none(row.toolCalls)):
                return True
        if row.functionCall is not None:
            if file_id in extract_workspace_file_ids(_json_or_none(row.functionCall)):
                return True
    return False


# ---------- Linked execution discovery (share-modal helper) ------------------


async def get_chat_share_state(session_id: str, user_id: str) -> ChatShareState:
    """Return the current share state for *session_id* scoped to *user_id*.

    Returns ``ChatShareState(is_shared=False, ...)`` when the session is
    missing or not owned by the user — the same shape callers see for
    never-shared sessions, so the share modal renders the "enable" path
    without an extra existence check.
    """
    row = await PrismaChatSession.prisma().find_first(
        where={"id": session_id, "userId": user_id},
    )
    if not row:
        return ChatShareState(
            is_shared=False, share_token=None, auto_share_executions=False
        )

    # Counts the modal renders as informed-consent disclosure.  Computed
    # at modal-open time (and again on each refetch) so they reflect the
    # live state of the chat — sharing is live, not snapshot-at-enable
    # (see the share-modal warning text), so the numbers under the
    # toggle ARE the numbers that will be visible the moment the
    # owner clicks "Enable sharing".
    message_count = await PrismaChatMessage.prisma().count(
        where={"sessionId": session_id},
    )
    linked_run_count = len(
        await _collect_execution_ids_from_messages(session_id=session_id)
    )
    file_count = await _count_referenced_workspace_files(
        session_id=session_id, user_id=user_id
    )

    return ChatShareState(
        is_shared=row.isShared,
        share_token=row.shareToken,
        auto_share_executions=row.autoShareExecutions,
        message_count=message_count,
        linked_run_count=linked_run_count,
        file_count=file_count,
    )


async def _count_referenced_workspace_files(*, session_id: str, user_id: str) -> int:
    """How many of the workspace files referenced by this chat's messages
    the owner actually owns (and would land in the public allowlist if
    the share were enabled right now).  Mirrors the ownership filter
    ``_build_shared_chat_files`` applies at enable time so the count
    matches what the public viewer can actually download.
    """
    rows = await PrismaChatMessage.prisma().find_many(
        where={"sessionId": session_id},
    )
    file_ids: set[str] = set()
    for row in rows:
        if row.content is not None:
            file_ids |= extract_workspace_file_ids(row.content)
        if row.toolCalls is not None:
            file_ids |= extract_workspace_file_ids(_json_or_none(row.toolCalls))
        if row.functionCall is not None:
            file_ids |= extract_workspace_file_ids(_json_or_none(row.functionCall))
    if not file_ids:
        return 0
    workspace = await UserWorkspace.prisma().find_unique(where={"userId": user_id})
    if not workspace:
        return 0
    owned = await UserWorkspaceFile.prisma().find_many(
        where={
            "id": {"in": list(file_ids)},
            "workspaceId": workspace.id,
            "isDeleted": False,
        }
    )
    return len(owned)


# ---------- Helpers ----------------------------------------------------------


async def _build_shared_chat_files(
    session_id: str,
    share_token: str,
    user_id: str,
    *,
    tx: Any | None = None,
) -> int:
    """Scan all messages in the session for workspace://<id> refs + persist allowlist."""
    rows = await PrismaChatMessage.prisma(tx).find_many(
        where={"sessionId": session_id},
    )

    file_ids: set[str] = set()
    for row in rows:
        if row.content is not None:
            file_ids |= extract_workspace_file_ids(row.content)
        if row.toolCalls is not None:
            file_ids |= extract_workspace_file_ids(_json_or_none(row.toolCalls))
        if row.functionCall is not None:
            file_ids |= extract_workspace_file_ids(_json_or_none(row.functionCall))

    if not file_ids:
        return 0

    workspace = await UserWorkspace.prisma(tx).find_unique(where={"userId": user_id})
    if not workspace:
        return 0

    owned_files = await UserWorkspaceFile.prisma(tx).find_many(
        where={
            "id": {"in": list(file_ids)},
            "workspaceId": workspace.id,
            "isDeleted": False,
        }
    )

    created = 0
    for file in owned_files:
        try:
            await SharedChatFile.prisma(tx).create(
                data={
                    "sessionId": session_id,
                    "fileId": file.id,
                    "shareToken": share_token,
                }
            )
            created += 1
        except UniqueViolationError:
            logger.debug("SharedChatFile already exists: %s/%s", session_id, file.id)
        except ForeignKeyViolationError:
            logger.debug("SharedChatFile FK violation for file %s", file.id)
    return created


async def _cascade_revoke_chat_linked_executions(
    *,
    session_id: str,
    tx: Any | None = None,
) -> None:
    """Revoke each CHAT_LINK execution previously linked to *session_id*
    that isn't still referenced by another chat session's linkage.

    Used by both ``disable_chat_session_share`` and the
    ``enable_chat_session_share`` re-share path — the latter calls this
    BEFORE wiping the join table so the previous generation of linkages
    doesn't strand orphan public per-execution shares.

    User-initiated execution shares (``sharedVia=USER``) are left
    untouched.  An execution still referenced by another chat session's
    linkage is left shared so revoking chat A doesn't break chat B's
    drill-in.
    """
    linked = await ChatLinkedShare.prisma(tx).find_many(
        where={"sessionId": session_id},
        include={"Execution": True},
    )
    for row in linked:
        execution = row.Execution
        if execution is None or execution.sharedVia != SharedVia.CHAT_LINK:
            continue
        other_link = await ChatLinkedShare.prisma(tx).find_first(
            where={
                "executionId": execution.id,
                "sessionId": {"not": session_id},
            },
        )
        if other_link is not None:
            continue
        # Drop the execution's file allowlist before clearing the share
        # token.  Orphan ``SharedExecutionFile`` rows would still
        # authorise downloads if anyone bookmarked the pre-revoke URL.
        await SharedExecutionFile.prisma(tx).delete_many(
            where={"executionId": execution.id}
        )
        await AgentGraphExecution.prisma(tx).update(
            where={"id": execution.id},
            data={
                "isShared": False,
                "shareToken": None,
                "sharedAt": None,
                "sharedVia": None,
            },
        )


async def _link_executions_to_share(
    session_id: str,
    executions: list[AgentGraphExecution],
    shared_at: datetime,
    *,
    tx: Any | None = None,
) -> None:
    """Insert ChatLinkedShare rows and cascade-enable share on unshared ones.

    Idempotent against the unique (sessionId, executionId) constraint —
    a concurrent ``run_agent`` for the same execution_id may race the
    runtime hook's pre-insert existence check and both reach this
    function.  In that case the second create raises
    ``UniqueViolationError``; we treat it as "already linked, skip"
    rather than a failure, mirroring ``_build_shared_chat_files``.
    """
    for execution in executions:
        try:
            await ChatLinkedShare.prisma(tx).create(
                data={"sessionId": session_id, "executionId": execution.id}
            )
        except UniqueViolationError:
            logger.debug(
                "ChatLinkedShare already exists: %s/%s",
                session_id,
                execution.id,
            )
            continue
        # Conditional update: only flip the execution to ``CHAT_LINK`` when
        # the DB still reports it unshared.  Branching on the pre-fetched
        # ``execution.isShared`` in-memory value can race a concurrent
        # transaction that has just shared (or revoked) it — see the
        # disable_chat_session_share cascade for the inverse race.  When
        # ``update_many`` matches zero rows we leave the execution's
        # existing share state alone; the ChatLinkedShare row above still
        # records the linkage, so the public viewer keeps drilling into
        # the user-shared token.
        new_token = generate_share_token()
        flipped = await AgentGraphExecution.prisma(tx).update_many(
            where={"id": execution.id, "isShared": False},
            data={
                "isShared": True,
                "shareToken": new_token,
                "sharedAt": shared_at,
                "sharedVia": SharedVia.CHAT_LINK,
            },
        )
        if flipped:
            # Mirror enable_execution_sharing's allowlist build so the
            # public execution viewer can serve the run's workspace files.
            # Without this, the chat viewer renders the run card but
            # every file download 404s — uniform 404 (anti-enumeration)
            # masks the bug, but the artifacts are unreachable.
            await _build_shared_execution_file_allowlist(
                execution_id=execution.id,
                share_token=new_token,
                user_id=execution.userId,
                tx=tx,
            )


async def _build_shared_execution_file_allowlist(
    *,
    execution_id: str,
    share_token: str,
    user_id: str,
    tx: Any | None = None,
) -> int:
    """Build ``SharedExecutionFile`` rows for a CHAT_LINK-flipped execution.

    Mirrors :func:`backend.data.execution.create_shared_execution_files` but
    is transaction-aware so the allowlist commits with the same atomic
    boundary as the share-state flip.  It intentionally scans only the
    public execution outputs rendered by the shared execution viewer, not
    every intermediate node output.  That keeps CHAT_LINK shares from
    authorising file downloads the normal execution-share path would not
    expose.
    """
    node_rows = await AgentGraphExecution.prisma(tx).find_unique(
        where={"id": execution_id},
        include={
            "NodeExecutions": {
                "include": {
                    "Input": True,
                    "Node": {"include": {"AgentBlock": True}},
                }
            }
        },
    )
    if node_rows is None or node_rows.NodeExecutions is None:
        return 0

    outputs = _collect_public_execution_outputs(node_rows)
    file_ids = extract_workspace_file_ids(outputs)
    if not file_ids:
        return 0

    workspace = await UserWorkspace.prisma(tx).find_unique(where={"userId": user_id})
    if workspace is None:
        return 0

    owned_files = await UserWorkspaceFile.prisma(tx).find_many(
        where={
            "id": {"in": list(file_ids)},
            "workspaceId": workspace.id,
            "isDeleted": False,
        }
    )

    created = 0
    for file in owned_files:
        try:
            await SharedExecutionFile.prisma(tx).create(
                data={
                    "executionId": execution_id,
                    "fileId": file.id,
                    "shareToken": share_token,
                }
            )
            created += 1
        except UniqueViolationError:
            logger.debug(
                "SharedExecutionFile already exists: %s/%s", execution_id, file.id
            )
        except ForeignKeyViolationError:
            logger.debug("SharedExecutionFile FK violation for file %s", file.id)
    return created


async def _resolve_linked_executions(
    session_id: str,
) -> list[SharedChatLinkedExecution]:
    rows = await ChatLinkedShare.prisma().find_many(
        where={"sessionId": session_id},
        include={"Execution": {"include": {"AgentGraph": True}}},
    )
    out: list[SharedChatLinkedExecution] = []
    for row in rows:
        execution = row.Execution
        if execution is None:
            continue
        out.append(
            SharedChatLinkedExecution(
                execution_id=execution.id,
                graph_id=execution.agentGraphId,
                graph_name=_graph_name(execution.AgentGraph),
                share_token=execution.shareToken if execution.isShared else None,
            )
        )
    return out


async def _collect_execution_ids_from_messages(session_id: str) -> set[str]:
    """Find execution IDs referenced by ``role=tool`` responses in a session.

    Matches both response shapes that ``run_agent`` produces:
    ``ExecutionStartedResponse`` (top-level ``execution_id``) and
    ``AgentOutputResponse`` (nested ``execution.execution_id`` from the
    sync-complete ``wait_for_result`` path).
    """
    rows = await PrismaChatMessage.prisma().find_many(
        where={"sessionId": session_id, "role": "tool"},
    )
    execution_ids: set[str] = set()
    for row in rows:
        if not row.content:
            continue
        payload = _json_or_none(row.content)
        if not isinstance(payload, dict):
            continue
        payload_type = payload.get("type")
        if payload_type == _EXECUTION_STARTED_TYPE:
            if payload.get("status") == _SCHEDULED_STATUS:
                continue
            execution_id = payload.get("execution_id")
        elif payload_type == _AGENT_OUTPUT_TYPE:
            execution = payload.get("execution")
            execution_id = (
                execution.get("execution_id") if isinstance(execution, dict) else None
            )
        elif payload_type == _ERROR_TYPE:
            execution_id = payload.get("execution_id")
        else:
            continue
        if isinstance(execution_id, str) and execution_id:
            execution_ids.add(execution_id)
    return execution_ids


def _collect_public_execution_outputs(
    execution: AgentGraphExecution,
) -> CompletedBlockOutput:
    outputs: CompletedBlockOutput = {}
    if execution.NodeExecutions is None:
        return outputs

    for node_exec in execution.NodeExecutions:
        node = getattr(node_exec, "Node", None)
        block_id = getattr(node, "agentBlockId", None)
        if not block_id:
            continue
        block = get_block(block_id)
        if block is None or block.block_type != BlockType.OUTPUT:
            continue
        input_data = _node_execution_input_data(node_exec)
        if "name" not in input_data:
            continue
        name = input_data["name"]
        value = input_data.get("value")
        outputs.setdefault(name, []).append(value)
    return outputs


def _node_execution_input_data(node_exec: Any) -> BlockInput:
    execution_data = getattr(node_exec, "executionData", None)
    if execution_data:
        return type_utils.convert(execution_data, BlockInput)

    input_data: BlockInput = {}
    for data in getattr(node_exec, "Input", None) or []:
        if data.name and data.data is not None:
            input_data[data.name] = type_utils.convert(data.data, Any)
    return input_data


def _graph_name(graph: "AgentGraph | None") -> str | None:
    if graph is None:
        return None
    return graph.name


def _json_or_none(value: Any) -> Any:
    """Parse *value* as JSON if it's a string; else return as-is."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return None
    return value
