"""Durable subtask handoffs; creation and wake enqueueing are transactional."""

import json
from datetime import datetime
from typing import Any, Literal

from prisma import Prisma
from prisma.models import ChatMessage, ChatSession
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from backend.copilot.background_task import (
    BACKGROUND_TIMEOUT_MESSAGE,
    BackgroundChildResult,
    BackgroundOutcome,
    BackgroundSessionState,
    BackgroundTask,
)
from backend.copilot.model import (
    ChatSessionMetadata,
    _parse_json_field,
    invalidate_session_cache,
)
from backend.copilot.stream_registry import CANCELLED_MESSAGE
from backend.data import db
from backend.util.json import SafeJson


async def get_background_task(task_id: str, user_id: str) -> BackgroundTask | None:
    rows = await db.query_raw_with_schema(
        'SELECT * FROM {schema_prefix}"ChatBackgroundTask" WHERE id=$1 AND "userId"=$2',
        task_id,
        user_id,
        model=BackgroundTask,
    )
    return rows[0] if rows else None


class _TaskState(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    outcome: BackgroundOutcome | None = None
    child_session_id: str
    parent_session_id: str
    followup_state: str


async def get_session_background_state(
    session_id: str, user_id: str
) -> BackgroundSessionState:
    """What a chat shows about background work without reading its transcript:
    a child's recorded outcome, and whether a parent still has a continuation
    to come."""
    rows = await db.query_raw_with_schema(
        """SELECT outcome, "childSessionId", "parentSessionId", "followupState"
        FROM {schema_prefix}"ChatBackgroundTask"
        WHERE "userId"=$2 AND ("childSessionId"=$1 OR "parentSessionId"=$1)""",
        session_id,
        user_id,
        model=_TaskState,
    )
    return BackgroundSessionState(
        outcome=next(
            (row.outcome for row in rows if row.child_session_id == session_id), None
        ),
        pending=any(
            row.parent_session_id == session_id
            and row.followup_state in ("waiting", "queued")
            for row in rows
        ),
    )


async def create_background_task(
    task: BackgroundTask, prompt: str, metadata: dict[str, Any]
) -> None:
    async with db.transaction() as tx:
        parent = await _lock_session(tx, task.parent_session_id)
        child = await _lock_session(tx, task.child_session_id)
        if not _matches(parent, task) or not _matches(child, task):
            raise ValueError("The subtask's owner or scope changed.")
        if (
            parent is None
            or parent.chatStatus != "running"
            or _parse_json_field(parent.metadata, default={}).get("_active_turn_id")
            != task.launch_turn_id
        ):
            raise ValueError("The launching turn is no longer running.")
        if child is None or child.chatStatus != "idle":
            raise ValueError("The subtask already has work in progress.")
        await db.execute_raw_with_schema(
            """INSERT INTO {schema_prefix}"ChatBackgroundTask"
            (id,"userId","parentSessionId","childSessionId","launchTurnId",
             "childTurnId",context,deadline)
            VALUES ($1,$2,$3,$4,$5,$6,$7::jsonb,$8::timestamptz)""",
            task.id,
            task.user_id,
            task.parent_session_id,
            task.child_session_id,
            task.launch_turn_id,
            task.child_turn_id,
            task.context.model_dump_json(),
            task.deadline,
            client=tx,
        )
        await _enqueue(tx, task.child_session_id, task.child_turn_id, prompt, metadata)


async def claim_background_turn(
    task_id: str,
    user_id: str,
    session_id: str,
    turn_id: str,
    stage: Literal["child", "followup"],
) -> bool:
    field = '"startedAt"' if stage == "child" else '"followupStartedAt"'
    target = '"childSessionId"' if stage == "child" else '"parentSessionId"'
    condition = (
        "outcome IS NULL"
        if stage == "child"
        else '"followupState"=\'queued\' AND NOT "wakeSuppressed"'
    )
    task = await get_background_task(task_id, user_id)
    if task is None or turn_id != (
        task.child_turn_id if stage == "child" else task.wake_id
    ):
        return False
    session = await ChatSession.prisma().find_unique(where={"id": session_id})
    if not _matches(session, task):
        raise ValueError("The background subtask's owner or scope changed.")
    updated = await db.execute_raw_with_schema(
        f"""UPDATE {{schema_prefix}}"ChatBackgroundTask" SET {field}=CURRENT_TIMESTAMP
        WHERE id=$1 AND "userId"=$2 AND {target}=$3 AND {field} IS NULL AND {condition}""",
        task_id,
        user_id,
        session_id,
    )
    return updated == 1


async def finish_background_turn(
    user_id: str,
    session_id: str,
    turn_id: str,
    error: str | None,
    *,
    turn_held: bool = False,
) -> None:
    """Record how a child or continuation turn ended, and for a child which
    row holds its answer; the answer itself stays in the child's chat. Every
    row after the launch is the turn's, a clarification it drained mid-turn
    included, when the worker still holds the turn (``turn_held``) or when
    the chat records no later turn; otherwise recovery stops at the next user
    row, since a later turn's rows cannot be told from drained ones."""
    tasks = await db.query_raw_with_schema(
        """SELECT * FROM {schema_prefix}"ChatBackgroundTask"
        WHERE "userId"=$1 AND "childSessionId"=$2 AND "childTurnId"=$3""",
        user_id,
        session_id,
        turn_id,
        model=BackgroundTask,
    )
    if not tasks:
        followups = await db.query_raw_with_schema(
            """SELECT * FROM {schema_prefix}"ChatBackgroundTask"
            WHERE "userId"=$1 AND "parentSessionId"=$2 AND "followupState"='queued' """,
            user_id,
            session_id,
            model=BackgroundTask,
        )
        for followup in followups:
            if followup.wake_id == turn_id:
                async with db.transaction() as tx:
                    await _lock_session(tx, session_id)
                    changed = await _finish_followup(tx, followup, error)
                if changed and error:
                    await invalidate_session_cache(session_id)
        await _complete_child_whose_work_is_back(user_id, session_id, error)
        return
    task = tasks[0]
    outcome = _outcome_of(error)
    if outcome == "completed" and await has_open_handoffs(user_id, session_id):
        # The child handed work off itself: its result is whole only once that
        # work has come back to it, so its record waits for its last
        # continuation to end (``_complete_child_whose_work_is_back``).
        return
    await complete_background_task(
        task.id,
        user_id,
        outcome,
        result_sequence=await _answer_sequence(task.child_turn_id, turn_held),
    )


def _outcome_of(error: str | None) -> BackgroundOutcome:
    """How a turn ended, from the error it ended with, if any."""
    if error is None:
        return "completed"
    if error == BACKGROUND_TIMEOUT_MESSAGE:
        return "timed_out"
    return "cancelled" if error == CANCELLED_MESSAGE else "failed"


async def has_open_handoffs(user_id: str, session_id: str) -> bool:
    """Whether the chat has background work of its own still out: a handoff
    whose wake has not run yet."""
    rows = await db.query_raw_with_schema(
        """SELECT 1 FROM {schema_prefix}"ChatBackgroundTask"
        WHERE "userId"=$1 AND "parentSessionId"=$2
          AND "followupState" IN ('waiting','queued') LIMIT 1""",
        user_id,
        session_id,
    )
    return bool(rows)


async def _complete_child_whose_work_is_back(
    user_id: str, session_id: str, error: str | None
) -> None:
    """Complete the record of a background child whose own turn ended with
    work of its own still out, once the last of that work has come back to
    it: its answer is then the latest row of its chat, its continuations'
    included, and it ends the way that last continuation ended."""
    if await has_open_handoffs(user_id, session_id):
        return
    tasks = await db.query_raw_with_schema(
        """SELECT * FROM {schema_prefix}"ChatBackgroundTask"
        WHERE "userId"=$1 AND "childSessionId"=$2
          AND outcome IS NULL AND "startedAt" IS NOT NULL""",
        user_id,
        session_id,
        model=BackgroundTask,
    )
    for task in tasks:
        await complete_background_task(
            task.id,
            user_id,
            _outcome_of(error),
            result_sequence=await _answer_sequence(task.child_turn_id, True),
        )


async def _answer_sequence(child_turn_id: str, turn_held: bool) -> int | None:
    """The last assistant row of the child's turn, which begins at its launch
    row. The turn owns every later row while the worker holds it, and while
    the chat still records it as the active turn (admission clears that
    identity and dispatch records the next one, so a later turn, started or
    finished, always shows); otherwise it ends at the next user row. One
    statement reads the chat and its rows in one snapshot, so a turn admitted
    between two reads cannot slip its answer past the check."""
    rows = await db.query_raw_with_schema(
        """SELECT message.sequence FROM {schema_prefix}"ChatMessage" message
        JOIN {schema_prefix}"ChatMessage" launch ON launch.id=$1
            AND message."sessionId"=launch."sessionId"
        JOIN {schema_prefix}"ChatSession" chat ON chat.id=launch."sessionId"
        WHERE message.role='assistant' AND message.sequence>launch.sequence
          AND ($2::boolean OR chat.metadata->>'_active_turn_id'=$1 OR NOT EXISTS (
            SELECT 1 FROM {schema_prefix}"ChatMessage" next
            WHERE next."sessionId"=launch."sessionId" AND next.role='user'
              AND next.sequence>launch.sequence AND next.sequence<message.sequence
          )) ORDER BY message.sequence DESC LIMIT 1""",
        child_turn_id,
        turn_held,
    )
    return int(rows[0]["sequence"]) if rows else None


async def complete_background_task(
    task_id: str,
    user_id: str,
    outcome: BackgroundOutcome,
    result_sequence: int | None = None,
) -> None:
    await db.execute_raw_with_schema(
        '''WITH finished AS (UPDATE {schema_prefix}"ChatBackgroundTask"
        SET outcome=$3, "resultSequence"=$4::int, "updatedAt"=CURRENT_TIMESTAMP
        WHERE id=$1 AND "userId"=$2 AND outcome IS NULL RETURNING "childTurnId", outcome)
        UPDATE {schema_prefix}"ChatMessage" message SET metadata=
            COALESCE(metadata,'{{}}'::jsonb) || jsonb_build_object('background_task_outcome',finished.outcome)
        FROM finished WHERE message.id=finished."childTurnId"''',
        task_id,
        user_id,
        outcome,
        result_sequence,
    )


async def list_background_tasks() -> list[BackgroundTask]:
    return await db.query_raw_with_schema(
        """WITH due AS (
            SELECT id FROM {schema_prefix}"ChatBackgroundTask"
            WHERE "followupState" IN ('waiting','queued')
            ORDER BY "updatedAt" LIMIT 100 FOR UPDATE SKIP LOCKED
        ) UPDATE {schema_prefix}"ChatBackgroundTask" task
        SET "updatedAt"=CURRENT_TIMESTAMP FROM due WHERE task.id=due.id RETURNING task.*""",
        model=BackgroundTask,
    )


async def enqueue_background_followup(
    task_id: str,
    user_id: str,
    message: str,
    metadata: dict[str, Any],
    terminal: bool = False,
) -> Literal["queued", "busy", "closed", "delivered"]:
    task = await get_background_task(task_id, user_id)
    if task is None:
        return "closed"
    async with db.transaction() as tx:
        parent = await _lock_session(tx, task.parent_session_id)
        rows = await db.query_raw_with_schema(
            """SELECT * FROM {schema_prefix}"ChatBackgroundTask"
            WHERE id=$1 AND "userId"=$2 FOR UPDATE""",
            task_id,
            user_id,
            model=BackgroundTask,
            client=tx,
        )
        if not rows or rows[0].followup_state != "waiting":
            return "closed"
        task = rows[0]
        if task.outcome is None or task.wake_suppressed:
            return "closed"
        if not _matches_scope(parent, task):
            await _close(tx, task_id, user_id)
            return "closed"
        if parent is None or parent.chatStatus != "idle":
            return "busy"
        if not _matches(parent, task):
            message = (
                "Otto's background follow-up couldn't run because this chat's "
                "connection changed. "
                f"[Open the subtask](/copilot?sessionId={task.child_session_id}) for details."
            )
            metadata = {"background_task_notice": task.id}
            terminal = True
        if terminal:
            await _append(
                tx,
                task.parent_session_id,
                task.wake_id,
                message,
                metadata,
                role="assistant",
            )
            await _close(tx, task_id, user_id)
            return "delivered"
        await _enqueue(tx, task.parent_session_id, task.wake_id, message, metadata)
        await db.execute_raw_with_schema(
            """UPDATE {schema_prefix}"ChatBackgroundTask" SET "followupState"='queued'
            WHERE id=$1 AND "userId"=$2""",
            task_id,
            user_id,
            client=tx,
        )
        return "queued"


async def suppress_background_followup(task_id: str, user_id: str) -> None:
    await _close(db.prisma, task_id, user_id)


async def release_deleted_background_turn(
    task_id: str, user_id: str, session_id: str, turn_id: str
) -> bool:
    """Release only the orphaned dispatch after its handoff was cascade-deleted."""
    return bool(
        await db.execute_raw_with_schema(
            """UPDATE {schema_prefix}"ChatSession" session SET "chatStatus"='idle'
            WHERE session.id=$1 AND "userId"=$2 AND "chatStatus"='running'
              AND session.metadata->>'_active_turn_id'=$3
              AND NOT EXISTS (
                SELECT 1 FROM {schema_prefix}"ChatBackgroundTask" WHERE id=$4
              )""",
            session_id,
            user_id,
            turn_id,
            task_id,
        )
    )


async def release_abandoned_background_turn(
    task: BackgroundTask, session_id: str, turn_id: str, before: datetime
) -> bool:
    """Release a chat still ``running`` one of the task's own turns with no
    live worker: the orphan rule the chat routes apply, fenced to that turn
    and to a row untouched since ``before``."""
    if (session_id, turn_id) not in (
        (task.child_session_id, task.child_turn_id),
        (task.parent_session_id, task.wake_id),
    ):
        return False
    return bool(
        await db.execute_raw_with_schema(
            """UPDATE {schema_prefix}"ChatSession" session SET "chatStatus"='idle'
            WHERE session.id=$1 AND "userId"=$2 AND "chatStatus"='running'
              AND "updatedAt" < $4::timestamptz
              AND session.metadata->>'_active_turn_id'=$3""",
            session_id,
            task.user_id,
            turn_id,
            before,
        )
    )


async def background_child_result(
    session_id: str, user_id: str
) -> BackgroundChildResult | None:
    """A background child's recorded state. None for a chat that is not a
    background child; an outcome of None while the child's turn has not been
    recorded; otherwise the answer and tool calls of that turn, read up to the
    row recorded when it ended, so a later turn in the chat is never mistaken
    for it, and the window the turn's files were written in."""
    tasks = await db.query_raw_with_schema(
        """SELECT * FROM {schema_prefix}"ChatBackgroundTask"
        WHERE "userId"=$1 AND "childSessionId"=$2""",
        user_id,
        session_id,
        model=BackgroundTask,
    )
    task = tasks[0] if tasks else None
    if task is None:
        return None
    if task.outcome is None:
        return BackgroundChildResult()
    if task.result_sequence is None:
        # Ended without an answer row: the turn's window still bounds its
        # files, from its launch to the next user row after it.
        windows = await db.query_raw_with_schema(
            """SELECT launch."createdAt" AS "writtenFrom", (
                SELECT min(next."createdAt") FROM {schema_prefix}"ChatMessage" next
                WHERE next."sessionId"=$1 AND next.role='user'
                  AND next.sequence>launch.sequence
              ) AS "writtenUntil"
            FROM {schema_prefix}"ChatMessage" launch WHERE launch.id=$2""",
            session_id,
            task.child_turn_id,
            model=_TurnWindow,
        )
        window = windows[0] if windows else None
        return BackgroundChildResult(
            outcome=task.outcome,
            written_from=window.written_from if window else None,
            written_until=window.written_until if window else None,
        )
    rows = await db.query_raw_with_schema(
        """SELECT message.content, message."toolCalls",
          launch."createdAt" AS "writtenFrom", (
            SELECT min(next."createdAt") FROM {schema_prefix}"ChatMessage" next
            WHERE next."sessionId"=$1 AND next.role='user' AND next.sequence>$3
          ) AS "writtenUntil"
        FROM {schema_prefix}"ChatMessage" message
        JOIN {schema_prefix}"ChatMessage" launch ON launch.id=$2
        WHERE message."sessionId"=$1 AND message.role='assistant'
          AND message.sequence>launch.sequence AND message.sequence<=$3
        ORDER BY message.sequence""",
        session_id,
        task.child_turn_id,
        task.result_sequence,
        model=_RecordedRow,
    )
    return BackgroundChildResult(
        outcome=task.outcome,
        response_text=(rows[-1].content or "") if rows else "",
        tool_calls=[call for row in rows for call in _tool_calls_of(row.tool_calls)],
        written_from=rows[0].written_from if rows else None,
        written_until=rows[0].written_until if rows else None,
    )


class _TurnWindow(BaseModel):
    """The recorded turn's window: from its launch row to the next user row
    after its answer, if any."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
    written_from: datetime
    written_until: datetime | None = None


class _RecordedRow(_TurnWindow):
    """An assistant row of the recorded turn, with the turn's window."""

    content: str | None = None
    tool_calls: Any = None


def _tool_calls_of(raw: Any) -> list[dict[str, Any]]:
    """The tool calls a persisted assistant row carries, however the driver
    returned the JSON column."""
    calls = json.loads(raw) if isinstance(raw, str) else raw
    return [call for call in (calls or []) if isinstance(call, dict)]


async def cancel_queued_background_followup(user_id: str, session_id: str) -> bool:
    async with db.transaction() as tx:
        parent = await _lock_session(tx, session_id)
        if parent is None or parent.userId != user_id or parent.chatStatus != "queued":
            return False
        tasks = await db.query_raw_with_schema(
            """SELECT * FROM {schema_prefix}"ChatBackgroundTask"
            WHERE "userId"=$1 AND "parentSessionId"=$2 AND "followupState"='queued'
              AND "followupStartedAt" IS NULL""",
            user_id,
            session_id,
            model=BackgroundTask,
            client=tx,
        )
        if not tasks:
            return False
        for task in tasks:
            await _finish_followup(tx, task, CANCELLED_MESSAGE)
        await ChatSession.prisma(tx).update(
            where={"id": session_id}, data={"chatStatus": "idle"}
        )
    await invalidate_session_cache(session_id)
    return True


async def cancel_background_tasks(
    user_id: str,
    session_id: str,
    launch_turn_id: str | None = None,
) -> list[BackgroundTask]:
    return await _cancel_background_tasks(
        db.prisma, user_id, session_id, launch_turn_id
    )


async def cancel_orphaned_background_tasks(
    user_id: str, session_id: str, expected_updated_at: datetime
) -> list[BackgroundTask] | None:
    """Cancel the observed orphan and its children before freeing the parent slot."""
    async with db.transaction() as tx:
        parent = await _lock_session(tx, session_id)
        if (
            parent is None
            or parent.userId != user_id
            or parent.chatStatus != "running"
            or parent.updatedAt != expected_updated_at
        ):
            return None
        turn_id = _parse_json_field(parent.metadata, default={}).get("_active_turn_id")
        tasks = await _cancel_background_tasks(tx, user_id, session_id, turn_id)
        await ChatSession.prisma(tx).update(
            where={"id": session_id}, data={"chatStatus": "idle"}
        )
        return tasks


async def _cancel_background_tasks(
    tx: Prisma, user_id: str, session_id: str, launch_turn_id: str | None
) -> list[BackgroundTask]:
    tasks = await db.query_raw_with_schema(
        """UPDATE {schema_prefix}"ChatBackgroundTask"
        SET outcome=COALESCE(outcome,'cancelled'),
            "wakeSuppressed"="wakeSuppressed" OR ("parentSessionId"=$2),
            "updatedAt"=CURRENT_TIMESTAMP
        WHERE "userId"=$1 AND "followupState" IN ('waiting','queued')
          AND ("childSessionId"=$2 OR ("parentSessionId"=$2 AND "launchTurnId"=$3))
        RETURNING *""",
        user_id,
        session_id,
        launch_turn_id,
        model=BackgroundTask,
        client=tx,
    )
    for task in tasks:
        await db.execute_raw_with_schema(
            """UPDATE {schema_prefix}"ChatMessage" SET metadata=
            COALESCE(metadata,'{{}}'::jsonb) || jsonb_build_object('background_task_outcome',$2::text)
            WHERE id=$1""",
            task.child_turn_id,
            task.outcome,
            client=tx,
        )
    return tasks


async def _finish_followup(tx: Prisma, task: BackgroundTask, error: str | None) -> bool:
    changed = await db.execute_raw_with_schema(
        """UPDATE {schema_prefix}"ChatBackgroundTask" SET "followupState"='finished'
        WHERE id=$1 AND "userId"=$2 AND "followupState"='queued' """,
        task.id,
        task.user_id,
        client=tx,
    )
    if changed and error:
        notice = {
            CANCELLED_MESSAGE: "Otto's background follow-up was cancelled.",
            BACKGROUND_TIMEOUT_MESSAGE: "Otto couldn't complete the background follow-up within its limits.",
        }.get(error, "Otto couldn't complete the background follow-up.")
        await _append(
            tx,
            task.parent_session_id,
            task.wake_id + "-notice",
            f"{notice} "
            f"[Open the subtask](/copilot?sessionId={task.child_session_id}) for details.",
            {"background_task_notice": task.id},
            role="assistant",
        )
    return bool(changed)


async def _enqueue(
    tx: Prisma, session_id: str, message_id: str, content: str, metadata: dict[str, Any]
) -> None:
    await _append(tx, session_id, message_id, content, metadata)
    await ChatSession.prisma(tx).update(
        where={"id": session_id}, data={"chatStatus": "queued"}
    )


async def _append(
    tx: Prisma,
    session_id: str,
    message_id: str,
    content: str,
    metadata: dict[str, Any],
    *,
    role: str = "user",
) -> None:
    last = await ChatMessage.prisma(tx).find_first(
        where={"sessionId": session_id}, order={"sequence": "desc"}
    )
    await ChatMessage.prisma(tx).create(
        data={
            "id": message_id,
            "sessionId": session_id,
            "role": role,
            "content": content,
            "sequence": last.sequence + 1 if last else 0,
            "metadata": SafeJson(metadata),
        }
    )


async def _lock_session(tx: Prisma, session_id: str) -> ChatSession | None:
    await db.query_raw_with_schema(
        'SELECT id FROM {schema_prefix}"ChatSession" WHERE id=$1 FOR UPDATE',
        session_id,
        client=tx,
    )
    return await ChatSession.prisma(tx).find_unique(where={"id": session_id})


def _matches(session: ChatSession | None, task: BackgroundTask) -> bool:
    if session is None or not _matches_scope(session, task):
        return False
    route = ChatSessionMetadata.model_validate(session.metadata)
    return (route.llm_auth_provider, route.llm_credential_id) == (
        task.context.llm_auth_provider,
        task.context.llm_credential_id,
    )


def _matches_scope(session: ChatSession | None, task: BackgroundTask) -> bool:
    return session is not None and (
        session.userId,
        session.organizationId,
        session.teamId,
        session.expertId,
    ) == (
        task.user_id,
        task.context.organization_id,
        task.context.team_id,
        task.context.expert_id,
    )


async def _close(tx: Prisma, task_id: str, user_id: str) -> None:
    await db.execute_raw_with_schema(
        """UPDATE {schema_prefix}"ChatBackgroundTask" SET "followupState"='suppressed'
        WHERE id=$1 AND "userId"=$2 AND "followupState" IN ('waiting','queued')""",
        task_id,
        user_id,
        client=tx,
    )
