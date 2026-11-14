"""What a task keeps beside its status: who did what, what it asked the user,
what it made and the proof it happened.

Most of it is written by harness code as the work goes (``copilot.task_recorder``),
so it does not depend on the model remembering. Every read and write is scoped
to the owning user through the task.
"""

import asyncio
import weakref
from datetime import UTC, datetime
from typing import Any

import prisma.models
from prisma import Json
from prisma.enums import (
    TaskActivityKind,
    TaskActor,
    TaskArtifactKind,
    TaskAskKind,
    TaskAskStatus,
    TaskEvidenceKind,
    TaskStatus,
)
from pydantic import BaseModel, Field

from backend.data.task import (
    _WITH_SESSIONS,
    LINE_MAX_CHARS,
    OPEN_STATUSES,
    Task,
    _line,
    get_task,
)
from backend.data.task_text import (
    file_key,
    recorded_tool_names,
    short_result,
    tool_names,
    url_key,
)

ACTIVITY_LIMIT = 200
# One activity writer per task at a time (see ``_append``); a lock lives only
# while a write holds it.
_APPEND_LOCKS: weakref.WeakValueDictionary[str, asyncio.Lock] = (
    weakref.WeakValueDictionary()
)
SO_FAR_LINES = 4
EXCERPT_MAX_CHARS = 280
# How far back a repeated call looks for its own line to fold into.
MERGE_WINDOW = 40
URLS_PER_LINE = 30


class TaskActivity(BaseModel):
    id: str
    task_id: str
    created_at: datetime
    actor_type: TaskActor
    actor_expert_id: str | None = None
    chat_session_id: str | None = None
    kind: TaskActivityKind
    text: str
    is_milestone: bool = False
    data: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_db(cls, row: prisma.models.TaskActivity) -> "TaskActivity":
        return cls(
            id=row.id,
            task_id=row.taskId,
            created_at=row.createdAt,
            actor_type=TaskActor(row.actorType),
            actor_expert_id=row.actorExpertId,
            chat_session_id=row.chatSessionId,
            kind=TaskActivityKind(row.kind),
            text=row.text,
            is_milestone=row.isMilestone,
            data=dict(row.data) if isinstance(row.data, dict) else {},
        )


class TaskAsk(BaseModel):
    id: str
    task_id: str
    created_at: datetime
    chat_session_id: str | None = None
    asked_by_expert_id: str | None = None
    kind: TaskAskKind
    question: str
    options: list[str] = Field(default_factory=list)
    default_guess: str | None = None
    status: TaskAskStatus
    answer: str | None = None
    answered_at: datetime | None = None

    @classmethod
    def from_db(cls, row: prisma.models.TaskAsk) -> "TaskAsk":
        return cls(
            id=row.id,
            task_id=row.taskId,
            created_at=row.createdAt,
            chat_session_id=row.chatSessionId,
            asked_by_expert_id=row.askedByExpertId,
            kind=TaskAskKind(row.kind),
            question=row.question,
            options=list(row.options or []),
            default_guess=row.defaultGuess,
            status=TaskAskStatus(row.status),
            answer=row.answer,
            answered_at=row.answeredAt,
        )


class TaskArtifact(BaseModel):
    id: str
    task_id: str
    created_at: datetime
    kind: TaskArtifactKind
    title: str
    workspace_file_id: str | None = None
    url: str | None = None
    created_by_expert_id: str | None = None
    chat_session_id: str | None = None
    is_missing: bool = False

    @classmethod
    def from_db(cls, row: prisma.models.TaskArtifact) -> "TaskArtifact":
        return cls(
            id=row.id,
            task_id=row.taskId,
            created_at=row.createdAt,
            kind=TaskArtifactKind(row.kind),
            title=row.title,
            workspace_file_id=row.workspaceFileId,
            url=row.url,
            created_by_expert_id=row.createdByExpertId,
            chat_session_id=row.chatSessionId,
            is_missing=row.isMissing,
        )


class TaskEvidence(BaseModel):
    id: str
    task_id: str
    created_at: datetime
    claim: str
    kind: TaskEvidenceKind
    tool_call_id: str | None = None
    chat_session_id: str | None = None
    artifact_id: str | None = None
    url: str | None = None
    excerpt: str | None = None
    verified: bool = False
    added_by: TaskActor = TaskActor.SYSTEM

    @classmethod
    def from_db(cls, row: prisma.models.TaskEvidence) -> "TaskEvidence":
        return cls(
            id=row.id,
            task_id=row.taskId,
            created_at=row.createdAt,
            claim=row.claim,
            kind=TaskEvidenceKind(row.kind),
            tool_call_id=row.toolCallId,
            chat_session_id=row.chatSessionId,
            artifact_id=row.artifactId,
            url=row.url,
            excerpt=row.excerpt,
            verified=row.verified,
            added_by=TaskActor(row.addedBy),
        )


class TaskRecord(BaseModel):
    """Everything the task page shows for one task."""

    task: Task
    subtasks: list[Task] = Field(default_factory=list)
    activity: list[TaskActivity] = Field(
        default_factory=list, description="The task's and its subtasks', newest first."
    )
    so_far: list[str] = Field(
        default_factory=list, description="The last few milestones, newest first."
    )
    asks: list[TaskAsk] = Field(default_factory=list)
    artifacts: list[TaskArtifact] = Field(default_factory=list)
    evidence: list[TaskEvidence] = Field(default_factory=list)
    is_unverified: bool = Field(
        default=False,
        description="Done with a stated done-when but no verified evidence.",
    )


class ActivityInput(BaseModel):
    kind: TaskActivityKind
    text: str
    actor_type: TaskActor = TaskActor.SYSTEM
    actor_expert_id: str | None = None
    chat_session_id: str | None = None
    is_milestone: bool = False
    data: dict[str, Any] = Field(default_factory=dict)
    # Lines with the same key since the last milestone fold into one, counted.
    merge_key: str | None = None
    # The folded line, with ``{count}`` for how many.
    text_many: str | None = None


class AskInput(BaseModel):
    question: str
    kind: TaskAskKind = TaskAskKind.QUESTION
    options: list[str] = Field(default_factory=list)
    default_guess: str | None = None
    held_call_id: str | None = None


class ArtifactInput(BaseModel):
    title: str
    kind: TaskArtifactKind = TaskArtifactKind.FILE
    workspace_file_id: str | None = None
    url: str | None = None


class EvidenceInput(BaseModel):
    """Proof for a claim. ``tool`` and ``file`` name something the task already
    did or made; the claim is verified only when that is found on record."""

    claim: str
    tool: str | None = None
    file: str | None = None
    url: str | None = None
    quote: str | None = None


class ToolCallRecord(BaseModel):
    """One tool call in a task's chat, as the recorder describes it."""

    tool: str
    tool_call_id: str
    label: str
    label_many: str
    artifact: ArtifactInput | None = None
    # Set when the call changed something outside: the claim it proves.
    evidence_claim: str | None = None
    deleted_file_id: str | None = None
    # The page a read fetched, so a link given as proof can be checked.
    url: str | None = None


def _actor(expert_id: str | None) -> TaskActor:
    return TaskActor.EXPERT if expert_id else TaskActor.OTTO


async def work_tasks(session_id: str, user_id: str) -> list[Task]:
    """The open tasks this chat is working on, newest first."""
    rows = await prisma.models.Task.prisma().find_many(
        where={
            "userId": user_id,
            "isArchived": False,
            "workSessionId": session_id,
            "status": {"in": OPEN_STATUSES},
        },
        include=_WITH_SESSIONS,
        order={"createdAt": "desc"},
    )
    return [Task.from_db(row) for row in rows]


async def find_open_task_by_title(
    user_id: str, session_id: str, title: str
) -> Task | None:
    """An open task with this title started from this chat, if there is one:
    asking for the same work twice returns the first task."""
    wanted = " ".join(title.split()).casefold()
    rows = await prisma.models.Task.prisma().find_many(
        where={
            "userId": user_id,
            "isArchived": False,
            "status": {"in": OPEN_STATUSES},
            "Sessions": {"some": {"chatSessionId": session_id}},
        },
        include=_WITH_SESSIONS,
        order={"createdAt": "asc"},
    )
    for row in rows:
        if row.title.casefold() == wanted:
            return Task.from_db(row)
    return None


async def add_activity(task_id: str, user_id: str, entry: ActivityInput) -> None:
    """Append a line to the task's activity; best-effort for a task that is
    gone or not the user's."""
    if await get_task(task_id, user_id) is None:
        return
    await _append(task_id, entry)


async def _append(task_id: str, entry: ActivityInput) -> None:
    """Add a line to the task's activity, one writer at a time per task: a
    model step runs its tool calls together, and two of them reading the
    activity before either wrote would each add a line instead of folding."""
    lock = _APPEND_LOCKS.setdefault(task_id, asyncio.Lock())
    async with lock:
        await _append_line(task_id, entry)


async def _append_line(task_id: str, entry: ActivityInput) -> None:
    text = _line(entry.text) or ""
    if not text:
        return
    activity = prisma.models.TaskActivity.prisma()
    open_line = await _open_line(task_id, entry.merge_key) if entry.merge_key else None
    if open_line is not None:
        data = dict(open_line.data) if isinstance(open_line.data, dict) else {}
        seen = data.get("count")
        count = (seen if isinstance(seen, int) else 1) + 1
        merged = (entry.text_many or text).format(count=count)
        await activity.update(
            where={"id": open_line.id},
            data={
                "text": merged[:LINE_MAX_CHARS],
                "data": Json(
                    {
                        **data,
                        **entry.data,
                        "urls": _urls(data, entry.data),
                        "count": count,
                        "merge_key": entry.merge_key,
                    }
                ),
            },
        )
        await _touch(task_id)
        return
    await activity.create(
        data={
            "taskId": task_id,
            "actorType": entry.actor_type,
            "actorExpertId": entry.actor_expert_id,
            "chatSessionId": entry.chat_session_id,
            "kind": entry.kind,
            "text": text,
            "isMilestone": entry.is_milestone,
            "data": Json(
                {**entry.data, "merge_key": entry.merge_key}
                if entry.merge_key
                else entry.data
            ),
        }
    )
    await _touch(task_id)


async def _open_line(task_id: str, merge_key: str) -> prisma.models.TaskActivity | None:
    """The line a repeated call folds into: the same kind of call since the
    task's last milestone. Between two milestones each tool shows once,
    counted, however the calls were interleaved."""
    recent = await prisma.models.TaskActivity.prisma().find_many(
        where={"taskId": task_id}, order={"createdAt": "desc"}, take=MERGE_WINDOW
    )
    for row in recent:
        if row.isMilestone:
            return None
        if isinstance(row.data, dict) and row.data.get("merge_key") == merge_key:
            return row
    return None


def _urls(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Every page the folded line read, oldest first, without repeats."""
    seen = [*(old.get("urls") or []), *(new.get("urls") or [])]
    return list(dict.fromkeys(str(url) for url in seen))[:URLS_PER_LINE]


async def _touch(task_id: str) -> None:
    await prisma.models.Task.prisma().update(
        where={"id": task_id}, data={"updatedAt": datetime.now(UTC)}
    )


async def set_progress(
    session_id: str,
    user_id: str,
    status_text: str,
    next_step: str | None = None,
    expert_id: str | None = None,
) -> Task | None:
    """Say where the chat's task is and what comes next, in the worker's own
    words. Each new line is a milestone: the tool calls after it fold under
    it. None when this chat is working on no task."""
    tasks = await work_tasks(session_id, user_id)
    if not tasks:
        return None
    task = tasks[0]
    line = _line(status_text)
    await prisma.models.Task.prisma().update(
        where={"id": task.id},
        data={"statusText": line, "nextStep": _line(next_step)},
    )
    if line and line != task.status_text:
        await _append(
            task.id,
            ActivityInput(
                kind=TaskActivityKind.PROGRESS,
                text=line,
                actor_type=_actor(expert_id),
                actor_expert_id=expert_id,
                chat_session_id=session_id,
                is_milestone=True,
            ),
        )
    return await get_task(task.id, user_id)


async def record_tool_call(
    session_id: str, user_id: str, call: ToolCallRecord, expert_id: str | None = None
) -> bool:
    """Write one tool call into the record of the task this chat is working
    on. False when the chat is working on none, so the caller can stop asking."""
    if call.deleted_file_id:
        await prisma.models.TaskArtifact.prisma().update_many(
            where={
                "workspaceFileId": call.deleted_file_id,
                "Task": {"is": {"userId": user_id}},
            },
            data={"isMissing": True},
        )
    tasks = await work_tasks(session_id, user_id)
    if not tasks:
        return False
    task = tasks[0]
    await _append(
        task.id,
        ActivityInput(
            kind=TaskActivityKind.ARTIFACT if call.artifact else TaskActivityKind.TOOL,
            text=call.label,
            text_many=call.label_many,
            actor_type=_actor(expert_id),
            actor_expert_id=expert_id,
            chat_session_id=session_id,
            is_milestone=call.artifact is not None or call.evidence_claim is not None,
            data={
                "tool": call.tool,
                "tool_call_id": call.tool_call_id,
                **({"urls": [call.url]} if call.url else {}),
            },
            merge_key=None if call.artifact or call.evidence_claim else call.tool,
        ),
    )
    if not await _has_worker_progress(task.id):
        # The worker has said nothing itself: the last thing it did stands in.
        await prisma.models.Task.prisma().update(
            where={"id": task.id}, data={"statusText": _line(call.label)}
        )
    artifact_id = None
    if call.artifact:
        artifact_id = await _save_artifact(
            task.id, call.artifact, session_id, expert_id
        )
    if call.evidence_claim:
        await prisma.models.TaskEvidence.prisma().create(
            data={
                "taskId": task.id,
                "claim": _line(call.evidence_claim) or call.label,
                "kind": TaskEvidenceKind.TOOL_RESULT,
                "toolCallId": call.tool_call_id,
                "chatSessionId": session_id,
                "artifactId": artifact_id,
                "verified": True,
                "addedBy": TaskActor.SYSTEM,
            }
        )
    return True


async def _has_worker_progress(task_id: str) -> bool:
    row = await prisma.models.TaskActivity.prisma().find_first(
        where={"taskId": task_id, "kind": TaskActivityKind.PROGRESS}
    )
    return row is not None


async def _save_artifact(
    task_id: str, artifact: ArtifactInput, session_id: str, expert_id: str | None
) -> str:
    """Record what the task made; writing the same file again keeps one row."""
    artifacts = prisma.models.TaskArtifact.prisma()
    title = _line(artifact.title) or "Untitled"
    existing = await artifacts.find_first(where={"taskId": task_id, "title": title})
    if existing:
        await artifacts.update(
            where={"id": existing.id},
            data={
                "workspaceFileId": artifact.workspace_file_id,
                "url": artifact.url,
                "isMissing": False,
            },
        )
        return existing.id
    row = await artifacts.create(
        data={
            "taskId": task_id,
            "kind": artifact.kind,
            "title": title,
            "workspaceFileId": artifact.workspace_file_id,
            "url": artifact.url,
            "createdByExpertId": expert_id,
            "chatSessionId": session_id,
        }
    )
    return row.id


async def add_evidence(
    session_id: str, user_id: str, evidence: EvidenceInput, expert_id: str | None = None
) -> TaskEvidence | None:
    """Attach proof to the task this chat is working on. The harness looks the
    source up itself: a file the task really made, a tool it really ran or a
    page it really read is verified; anything else is kept as unverified.
    None when this chat is working on no task."""
    tasks = await work_tasks(session_id, user_id)
    if not tasks:
        return None
    task = tasks[0]
    source = await _find_source([task.id, *await _subtask_ids(task.id)], evidence)
    row = await prisma.models.TaskEvidence.prisma().create(
        data={
            "taskId": task.id,
            "claim": _line(evidence.claim) or "Unnamed claim",
            "kind": source.kind,
            "toolCallId": source.tool_call_id,
            "chatSessionId": session_id,
            "artifactId": source.artifact_id,
            "url": evidence.url,
            "excerpt": redact(evidence.quote),
            "verified": source.verified,
            "addedBy": _actor(expert_id),
        }
    )
    await _append(
        task.id,
        ActivityInput(
            kind=TaskActivityKind.EVIDENCE,
            text="Added proof",
            text_many="Added {count} proofs",
            actor_type=_actor(expert_id),
            actor_expert_id=expert_id,
            chat_session_id=session_id,
            data={"verified": source.verified},
            merge_key="evidence",
        ),
    )
    return TaskEvidence.from_db(row)


class _Source(BaseModel):
    """What a piece of proof points at, and whether it was found on record."""

    kind: TaskEvidenceKind = TaskEvidenceKind.QUOTE
    verified: bool = False
    tool_call_id: str | None = None
    artifact_id: str | None = None


async def _find_source(family: list[str], evidence: EvidenceInput) -> _Source:
    """Look up what the proof names among the task's own files, tool calls
    and pages read. Names are matched loosely (``task_text``): a worker
    writes 'mcp__copilot__bash_exec' or a sandbox path as often as the bare
    name. The first source found verifies the claim."""
    if evidence.file:
        wanted = file_key(evidence.file)
        made = await prisma.models.TaskArtifact.prisma().find_many(
            where={"taskId": {"in": family}, "isMissing": False},
            order={"createdAt": "desc"},
        )
        for artifact in made:
            if wanted and wanted in (
                file_key(artifact.title),
                artifact.workspaceFileId,
            ):
                return _Source(
                    kind=TaskEvidenceKind.FILE, verified=True, artifact_id=artifact.id
                )
    cited, page = tool_names(evidence.tool), url_key(evidence.url)
    if cited or page:
        calls = await prisma.models.TaskActivity.prisma().find_many(
            where={
                "taskId": {"in": family},
                "kind": {"in": [TaskActivityKind.TOOL, TaskActivityKind.ARTIFACT]},
            },
            order={"createdAt": "desc"},
            take=ACTIVITY_LIMIT,
        )
        for call in calls:
            data = call.data if isinstance(call.data, dict) else {}
            call_id = data.get("tool_call_id")
            if cited & recorded_tool_names(str(data.get("tool") or "")):
                return _Source(
                    kind=TaskEvidenceKind.TOOL_RESULT,
                    verified=True,
                    tool_call_id=call_id,
                )
            if page and page in {url_key(str(url)) for url in data.get("urls") or []}:
                return _Source(
                    kind=TaskEvidenceKind.LINK, verified=True, tool_call_id=call_id
                )
    if evidence.file:
        return _Source(kind=TaskEvidenceKind.FILE)
    if evidence.tool:
        return _Source(kind=TaskEvidenceKind.TOOL_RESULT)
    return _Source(
        kind=TaskEvidenceKind.LINK if evidence.url else TaskEvidenceKind.QUOTE
    )


def redact(text: str | None) -> str | None:
    """A short excerpt with anything shaped like a key or token taken out."""
    text = " ".join((text or "").split())
    if not text:
        return None
    words = [
        "[removed]" if _looks_secret(word) else word
        for word in text[:EXCERPT_MAX_CHARS].split(" ")
    ]
    return " ".join(words)


_SECRET_PREFIXES = ("sk-", "pk-", "ghp_", "gho_", "xoxb-", "xoxp-", "AKIA", "eyJ")


def _looks_secret(word: str) -> bool:
    bare = word.strip("\"'`,.;:()[]{}")
    if bare.startswith(_SECRET_PREFIXES) and len(bare) >= 12:
        return True
    return bare.lower() in ("bearer",) or (
        len(bare) >= 32 and bare.isalnum() and not bare.isalpha() and not bare.isdigit()
    )


async def open_asks(
    session_id: str, user_id: str, asks: list[AskInput], expert_id: str | None = None
) -> int:
    """Park the questions on the task this chat is working on, so the task
    shows what it waits on. Returns how many were recorded."""
    tasks = await work_tasks(session_id, user_id)
    if not tasks or not asks:
        return 0
    task = tasks[0]
    for ask in asks:
        await prisma.models.TaskAsk.prisma().create(
            data={
                "taskId": task.id,
                "chatSessionId": session_id,
                "askedByExpertId": expert_id,
                "kind": ask.kind,
                "question": ask.question.strip()[:1000],
                "options": ask.options[:12],
                "defaultGuess": _line(ask.default_guess),
                "heldCallId": ask.held_call_id,
            }
        )
        await _append(
            task.id,
            ActivityInput(
                kind=TaskActivityKind.ASKED,
                text=f"Asked you: {ask.question}",
                actor_type=_actor(expert_id),
                actor_expert_id=expert_id,
                chat_session_id=session_id,
                is_milestone=True,
            ),
        )
    return len(asks)


async def answer_session_asks(session_id: str, user_id: str, answer: str) -> int:
    """The user replied in the chat that asked: close its open asks and keep
    the answer, so the task shows what was decided."""
    asks = prisma.models.TaskAsk.prisma()
    open_rows = await asks.find_many(
        where={
            "chatSessionId": session_id,
            "status": TaskAskStatus.OPEN,
            "Task": {"is": {"userId": user_id}},
        }
    )
    if not open_rows:
        return 0
    answer = " ".join(answer.split())[:1000]
    await asks.update_many(
        where={"id": {"in": [row.id for row in open_rows]}},
        data={
            "status": TaskAskStatus.ANSWERED,
            "answer": answer,
            "answeredAt": datetime.now(UTC),
        },
    )
    for task_id in {row.taskId for row in open_rows}:
        await _append(
            task_id,
            ActivityInput(
                kind=TaskActivityKind.ANSWERED,
                text=f"You answered: {answer}",
                actor_type=TaskActor.USER,
                chat_session_id=session_id,
                is_milestone=True,
            ),
        )
    return len(open_rows)


_SETTLED_LINE = {
    TaskStatus.DONE: (TaskActivityKind.DONE, "Finished"),
    TaskStatus.FAILED: (TaskActivityKind.FAILED, "Failed"),
    TaskStatus.CANCELLED: (TaskActivityKind.STATUS, "Cancelled"),
}


async def settle_work_tasks(
    session_id: str,
    user_id: str,
    status: TaskStatus,
    result: str | None = None,
    failure_reason: str | None = None,
) -> int:
    """Close the tasks a finished worker was running, with the outcome.

    Only running ones move: a task waiting on the user stays open, since the
    worker ended by asking, and so does one whose subtasks are still open.
    Idempotent, so a delivery retried later moves nothing twice.
    """
    tasks = prisma.models.Task.prisma()
    running = await tasks.find_many(
        where={
            "userId": user_id,
            "isArchived": False,
            "status": {"in": [TaskStatus.TODO, TaskStatus.IN_PROGRESS]},
            "OR": [
                {"workSessionId": session_id},
                {
                    "workSessionId": None,
                    "Sessions": {"some": {"chatSessionId": session_id}},
                },
            ],
        },
        include={"Subtasks": True},
    )
    moved = 0
    for row in running:
        children = row.Subtasks or []
        if status == TaskStatus.DONE and any(
            child.status in OPEN_STATUSES for child in children
        ):
            continue
        kind, word = _SETTLED_LINE[status]
        await tasks.update(
            where={"id": row.id},
            data={
                "status": status,
                "completedAt": datetime.now(UTC),
                # A summary the worker left (``task_result.finish_task``)
                # stands; a bare reply is cut down to its opening.
                "result": (
                    row.result or short_result(result)
                    if status == TaskStatus.DONE
                    else row.result
                ),
                "failureReason": (
                    _line(failure_reason) if status == TaskStatus.FAILED else None
                ),
                "statusText": None,
                "nextStep": None,
            },
        )
        await prisma.models.TaskAsk.prisma().update_many(
            where={"taskId": row.id, "status": TaskAskStatus.OPEN},
            data={"status": TaskAskStatus.WITHDRAWN},
        )
        reason = (
            f": {failure_reason}"
            if status == TaskStatus.FAILED and failure_reason
            else ""
        )
        await _append(
            row.id,
            ActivityInput(
                kind=kind,
                text=f"{word}{reason}",
                chat_session_id=session_id,
                is_milestone=True,
            ),
        )
        moved += 1
    return moved


async def _subtask_ids(task_id: str) -> list[str]:
    rows = await prisma.models.Task.prisma().find_many(
        where={"parentTaskId": task_id, "isArchived": False}
    )
    return [row.id for row in rows]


async def get_task_record(task_id: str, user_id: str) -> TaskRecord | None:
    """The task with its subtasks and everything on record for them."""
    task = await get_task(task_id, user_id)
    if task is None:
        return None
    child_rows = await prisma.models.Task.prisma().find_many(
        where={"parentTaskId": task_id, "isArchived": False, "userId": user_id},
        include=_WITH_SESSIONS,
        order={"createdAt": "asc"},
    )
    subtasks = [Task.from_db(row) for row in child_rows]
    family = [task_id, *(child.id for child in subtasks)]
    in_family = {"taskId": {"in": family}}
    activity = [
        TaskActivity.from_db(row)
        for row in await prisma.models.TaskActivity.prisma().find_many(
            where=in_family, order={"createdAt": "desc"}, take=ACTIVITY_LIMIT
        )
    ]
    asks = [
        TaskAsk.from_db(row)
        for row in await prisma.models.TaskAsk.prisma().find_many(
            where=in_family, order={"createdAt": "desc"}
        )
    ]
    artifacts = [
        TaskArtifact.from_db(row)
        for row in await prisma.models.TaskArtifact.prisma().find_many(
            where=in_family, order={"createdAt": "asc"}
        )
    ]
    evidence = [
        TaskEvidence.from_db(row)
        for row in await prisma.models.TaskEvidence.prisma().find_many(
            where=in_family, order={"createdAt": "asc"}
        )
    ]
    return TaskRecord(
        task=task,
        subtasks=subtasks,
        activity=activity,
        so_far=[line.text for line in activity if line.is_milestone][:SO_FAR_LINES],
        asks=asks,
        artifacts=artifacts,
        evidence=evidence,
        is_unverified=is_unverified(task, evidence),
    )


def is_unverified(task: Task, evidence: list[TaskEvidence]) -> bool:
    """Done against a stated done-when, with nothing verified to show for it."""
    return (
        task.status == TaskStatus.DONE
        and bool(task.done_when)
        and not any(item.verified for item in evidence)
    )
