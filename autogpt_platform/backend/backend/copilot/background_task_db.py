"""Durable subtask handoffs; creation and wake enqueueing are transactional."""

from datetime import datetime
from typing import Any, Literal

from prisma import Prisma
from prisma.models import ChatMessage, ChatSession
from prisma.types import ChatMessageWhereInput

from backend.copilot.background_task import (
    BACKGROUND_TIMEOUT_MESSAGE,
    BackgroundOutcome,
    BackgroundTask,
)
from backend.copilot.model import (
    ChatSessionMetadata,
    _parse_json_field,
    invalidate_session_cache,
)
from backend.copilot.stream_registry import CANCELLED_MESSAGE
from backend.copilot.tree import TurnEnvelope
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
    user_id: str, session_id: str, turn_id: str, error: str | None
) -> None:
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
                    changed = await db.execute_raw_with_schema(
                        """UPDATE {schema_prefix}"ChatBackgroundTask" SET "followupState"='finished'
                        WHERE id=$1 AND "userId"=$2 AND "followupState"='queued' """,
                        followup.id,
                        user_id,
                        client=tx,
                    )
                    if changed and error:
                        await _lock_session(tx, session_id)
                        notice = {
                            CANCELLED_MESSAGE: "Otto's background follow-up was cancelled.",
                            BACKGROUND_TIMEOUT_MESSAGE: "Otto couldn't complete the background follow-up within its limits.",
                        }.get(error, "Otto couldn't complete the background follow-up.")
                        await _append(
                            tx,
                            session_id,
                            followup.wake_id + "-notice",
                            f"{notice} "
                            f"[Open the subtask](/copilot?sessionId={followup.child_session_id}) for details.",
                            {"background_task_notice": followup.id},
                            role="assistant",
                        )
                if changed and error:
                    await invalidate_session_cache(session_id)
        return
    task = tasks[0]
    outcome: BackgroundOutcome = (
        "completed"
        if error is None
        else (
            "timed_out"
            if error == BACKGROUND_TIMEOUT_MESSAGE
            else "cancelled" if error == CANCELLED_MESSAGE else "failed"
        )
    )
    messages = await db.query_raw_with_schema(
        """SELECT message.* FROM {schema_prefix}"ChatMessage" message
        JOIN {schema_prefix}"ChatMessage" launch ON launch.id=$1
            AND message."sessionId"=launch."sessionId"
        WHERE message.role='assistant' AND message.sequence>launch.sequence
          AND NOT EXISTS (
            SELECT 1 FROM {schema_prefix}"ChatMessage" next
            WHERE next."sessionId"=launch."sessionId" AND next.role='user'
              AND next.sequence>launch.sequence AND next.sequence<message.sequence
          ) ORDER BY message.sequence DESC LIMIT 1""",
        task.child_turn_id,
        model=ChatMessage,
    )
    result = (messages[0].content or "") if messages else ""
    await complete_background_task(task.id, user_id, outcome, result[:24000])


async def complete_background_task(
    task_id: str, user_id: str, outcome: BackgroundOutcome, result: str = ""
) -> None:
    await db.execute_raw_with_schema(
        '''WITH finished AS (UPDATE {schema_prefix}"ChatBackgroundTask"
        SET outcome=$3, result=$4, "updatedAt"=CURRENT_TIMESTAMP
        WHERE id=$1 AND "userId"=$2 AND outcome IS NULL RETURNING "childTurnId", outcome)
        UPDATE {schema_prefix}"ChatMessage" message SET metadata=
            COALESCE(metadata,'{{}}'::jsonb) || jsonb_build_object('background_task_outcome',finished.outcome)
        FROM finished WHERE message.id=finished."childTurnId"''',
        task_id,
        user_id,
        outcome,
        result,
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
        if not _matches(parent, task):
            await _close(tx, task_id, user_id)
            return "closed"
        if parent is None or parent.chatStatus != "idle":
            return "busy"
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


async def get_background_followup_envelope(
    task_id: str, user_id: str
) -> TurnEnvelope | None:
    task = await get_background_task(task_id, user_id)
    if task is None or task.followup_state != "queued":
        return None
    parent = await ChatSession.prisma().find_unique(
        where={"id": task.parent_session_id}
    )
    if parent is None or not _matches(parent, task):
        return None
    where: ChatMessageWhereInput = {
        "id": task.wake_id,
        "sessionId": parent.id,
        "role": "user",
    }
    if parent.chatStatus != "queued" and (
        _parse_json_field(parent.metadata, default={}).get("_active_turn_id")
        != task.wake_id
    ):
        return None
    message = await ChatMessage.prisma().find_first(where=where)
    if message is None:
        return None
    return TurnEnvelope.model_validate(
        _parse_json_field(message.metadata, default={}).get("envelope")
    )


async def release_expired_background_turn(
    task_id: str,
    user_id: str,
    session_id: str,
    turn_id: str,
    deadline: datetime,
) -> None:
    task = await get_background_task(task_id, user_id)
    if task is None or (session_id, turn_id) not in (
        (task.child_session_id, task.child_turn_id),
        (task.parent_session_id, task.wake_id),
    ):
        return
    await db.execute_raw_with_schema(
        """UPDATE {schema_prefix}"ChatSession" session SET "chatStatus"='idle'
        WHERE session.id=$1 AND "userId"=$2 AND "chatStatus"='running'
          AND "updatedAt" < $4::timestamptz
          AND session.metadata->>'_active_turn_id'=$3""",
        session_id,
        user_id,
        turn_id,
        deadline,
    )


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
    if session is None:
        return False
    route = ChatSessionMetadata.model_validate(session.metadata)
    return (
        session.userId,
        session.organizationId,
        session.teamId,
        session.expertId,
    ) == (
        task.user_id,
        task.context.organization_id,
        task.context.team_id,
        task.context.expert_id,
    ) and (
        route.llm_auth_provider,
        route.llm_credential_id,
    ) == (
        task.context.llm_auth_provider,
        task.context.llm_credential_id,
    )


async def _close(tx: Prisma, task_id: str, user_id: str) -> None:
    await db.execute_raw_with_schema(
        """UPDATE {schema_prefix}"ChatBackgroundTask" SET "followupState"='suppressed'
        WHERE id=$1 AND "userId"=$2 AND "followupState" IN ('waiting','queued')""",
        task_id,
        user_id,
        client=tx,
    )
