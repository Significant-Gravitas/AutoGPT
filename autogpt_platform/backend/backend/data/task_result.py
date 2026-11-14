"""How a task ends: a short summary the user can take in at a glance, and the
document that holds the full result. The worker says both with
``finish_task``; the chat that started the task is then told the same two
things, so it can answer in a line instead of retelling the work."""

import prisma.models
from prisma.enums import TaskActor, TaskArtifactKind, TaskEvidenceKind, TaskStatus
from pydantic import BaseModel

from backend.data.task import RESULT_MAX_CHARS
from backend.data.task_record import work_tasks
from backend.data.task_text import clip, file_key


class TaskFinish(BaseModel):
    """What ``finish_task`` saved."""

    task_id: str
    document_title: str | None = None
    # The file named as the full result is not one this task saved.
    missing_file: bool = False
    # The chat is a background worker: nobody waits on a closing message.
    in_background: bool = False


class TaskOutcome(BaseModel):
    """What a worker's task came to, for the chat that started it."""

    task_id: str
    title: str
    status: TaskStatus
    result: str | None = None
    failure_reason: str | None = None
    document_title: str | None = None
    document_file_id: str | None = None


async def finish_task(
    session_id: str, user_id: str, summary: str, file: str | None = None
) -> TaskFinish | None:
    """Save the result of the task this chat is working on: the summary, and
    the file that holds the full result, which becomes the task's document
    with proof that it was saved. Nothing is saved when the named file is
    not one of the task's own. None when this chat is working on no task."""
    tasks = await work_tasks(session_id, user_id)
    if not tasks:
        return None
    task = tasks[0]
    document = await _saved_file(task.id, file) if file else None
    if file and document is None:
        return TaskFinish(task_id=task.id, missing_file=True)
    await prisma.models.Task.prisma().update(
        where={"id": task.id},
        data={"result": clip(summary, RESULT_MAX_CHARS), "nextStep": None},
    )
    in_background = await _is_background_chat(session_id)
    if document is None:
        return TaskFinish(task_id=task.id, in_background=in_background)
    await prisma.models.TaskArtifact.prisma().update(
        where={"id": document.id}, data={"kind": TaskArtifactKind.DOCUMENT}
    )
    evidence = prisma.models.TaskEvidence.prisma()
    if await evidence.find_first(where={"artifactId": document.id}) is None:
        await evidence.create(
            data={
                "taskId": task.id,
                "claim": f"Result saved as {document.title}",
                "kind": TaskEvidenceKind.FILE,
                "chatSessionId": session_id,
                "artifactId": document.id,
                "verified": True,
                "addedBy": TaskActor.SYSTEM,
            }
        )
    return TaskFinish(
        task_id=task.id, document_title=document.title, in_background=in_background
    )


async def is_worker_chat(session_id: str, user_id: str) -> bool:
    """True for a background worker's chat that is working on a task: the
    user is not looking at it, so it decides more and asks less."""
    if not await _is_background_chat(session_id):
        return False
    return bool(await work_tasks(session_id, user_id))


async def _is_background_chat(session_id: str) -> bool:
    row = await prisma.models.ChatBackgroundTask.prisma().find_first(
        where={"childSessionId": session_id}
    )
    return row is not None


async def task_outcome(session_id: str, user_id: str) -> TaskOutcome | None:
    """The task a worker's chat ran, as it stands now, with its document."""
    row = await prisma.models.Task.prisma().find_first(
        where={"userId": user_id, "workSessionId": session_id, "isArchived": False},
        order={"updatedAt": "desc"},
        include={"Artifacts": True},
    )
    if row is None:
        return None
    documents = [
        artifact
        for artifact in row.Artifacts or []
        if artifact.kind == TaskArtifactKind.DOCUMENT and not artifact.isMissing
    ]
    document = max(documents, key=lambda a: a.createdAt, default=None)
    return TaskOutcome(
        task_id=row.id,
        title=row.title,
        status=row.status,
        result=row.result,
        failure_reason=row.failureReason,
        document_title=document.title if document else None,
        document_file_id=document.workspaceFileId if document else None,
    )


async def _saved_file(task_id: str, file: str) -> prisma.models.TaskArtifact | None:
    """The file this task saved under that name, newest first."""
    wanted = file_key(file)
    saved = await prisma.models.TaskArtifact.prisma().find_many(
        where={"taskId": task_id, "isMissing": False}, order={"createdAt": "desc"}
    )
    return next(
        (
            artifact
            for artifact in saved
            if wanted and wanted in (file_key(artifact.title), artifact.workspaceFileId)
        ),
        None,
    )
