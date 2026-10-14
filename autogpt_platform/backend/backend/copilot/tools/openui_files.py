"""OpenUI views saved as workspace files, shared by render_ui and the
workspace file tools.

A saved view is a ``.openui`` file whose body is the OpenUI program; its
metadata names the view and the chat that last wrote it (see
``backend.data.openui_files``). Every write is validated first, and goes
through ``WorkspaceManager`` so size limits, quota and virus scanning apply.
"""

import json
import logging
import re

from pydantic import BaseModel

from backend.copilot.context import get_workspace_manager
from backend.copilot.tools.models import ErrorResponse, ResponseType, ToolResponseBase
from backend.copilot.tools.openui_source import validate_complete_delimiters
from backend.copilot.tools.openui_validator import (
    ValidatorUnavailable,
    validate_openui_source,
)
from backend.data.db_accessors import openui_files_db
from backend.data.openui_files import (
    is_openui_file,
    is_openui_name,
    openui_filename,
    openui_metadata,
    openui_summary,
    openui_view_id,
)
from backend.data.workspace import OPENUI_EXTENSION, OPENUI_MIME_TYPE, WorkspaceFile
from backend.util.workspace import WorkspaceManager

logger = logging.getLogger(__name__)

MAX_SOURCE_CHARS = 60_000
_MAX_EDITS_BYTES = 20_000
_FILE_REF = re.compile(
    r"^(?:workspace://)?([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})(?:#.*)?$",
    re.I,
)
_AGENT_ORIGIN = {"origin": "agent-created"}

RENDER_RETRY = (
    'Correct these issues and retry run_capability(id="tool:render_ui", '
    "input={...}) with a complete source."
)
RENDER_FALLBACK = "Provide the complete answer as plain text instead"
WRITE_RETRY = (
    "Correct these issues and retry write_workspace_file with the complete program."
)
WRITE_FALLBACK = "Save the note as Markdown (.md) instead"


class ViewFileError(ValueError):
    """A model-facing reason the view could not be saved as a file."""


class PublishedView(BaseModel):
    view_id: str
    summary: str


class OpenUIFileContentResponse(ToolResponseBase):
    """A saved view read back: the program, its generated summary and the
    user's current edits (field values, ticks), so the model sees both."""

    type: ResponseType = ResponseType.WORKSPACE_FILE_CONTENT
    file_id: str
    name: str
    path: str
    mime_type: str
    view_id: str | None = None
    summary: str
    source: str
    user_edits: dict | None = None


def check_workspace_program(source: str) -> str:
    if not re.match(r"root\s*=\s*Workspace\s*\(", source):
        raise ValueError("Start source with root = Workspace(...), without fences")
    validate_complete_delimiters(source)
    return source


async def validate_view(
    source: str, *, session_id: str, retry: str, fallback: str
) -> PublishedView | ErrorResponse:
    """Validate a complete program; errors tell the model how to repair it."""
    try:
        validation = await validate_openui_source(source)
    except ValidatorUnavailable as error:
        logger.warning(f"OpenUI validation unavailable: {error}")
        return ErrorResponse(
            message="Interactive validation is temporarily unavailable. "
            f"{fallback} of retrying this view.",
            session_id=session_id,
        )
    if not validation.valid:
        return ErrorResponse(
            message=f"View not published: {validation.error} {retry} "
            "Preserve the user's data and constraints; the summary is generated.",
            session_id=session_id,
        )
    if not validation.summary:
        return ErrorResponse(
            message="Interactive validation returned no text summary. " f"{fallback}.",
            session_id=session_id,
        )
    if not validation.view_id:
        return ErrorResponse(
            message="View not published: Workspace requires a stable viewId from "
            "the first render so edits can be saved. Retry with "
            'Workspace(title, description, children, "stable-view-id"). '
            "Reuse this ID when revising this task; use a new ID for a new task.",
            session_id=session_id,
        )
    return PublishedView(view_id=validation.view_id, summary=validation.summary)


def parse_file_ref(value: str) -> str | None:
    """The file ID in ``workspace://<id>[#mime]`` or a bare ID, else None."""
    match = _FILE_REF.match(value.strip())
    return match.group(1).lower() if match else None


async def save_view(
    user_id: str,
    session_id: str,
    *,
    source: str,
    view: PublishedView,
    file: str | None,
) -> WorkspaceFile | None:
    """Save a validated view for render_ui.

    ``file`` names a new file or references an existing one by ID. Without
    it, a file that already holds this view for the chat is still revised so
    the saved copy never falls behind; otherwise the view stays chat-only and
    this returns None.
    """
    file_id = parse_file_ref(file) if file else None
    if file_id is None:
        file_id = await _chat_view_file(
            user_id, session_id, view.view_id, required=bool(file)
        )
    if file_id is None and not file:
        return None
    manager = await get_workspace_manager(user_id, session_id)
    if file_id is not None:
        return await revise_view_file(manager, file_id, source, view, session_id)
    assert file is not None
    created = await manager.write_file(
        content=source.encode(),
        filename=openui_filename(file),
        mime_type=OPENUI_MIME_TYPE,
        metadata=openui_metadata(_AGENT_ORIGIN, view.view_id, session_id, view.summary),
    )
    await _move_chat_edits(user_id, session_id, view.view_id, created)
    return created


async def revise_view_file(
    manager: WorkspaceManager,
    file_id: str,
    source: str,
    view: PublishedView,
    session_id: str,
) -> WorkspaceFile:
    """Replace a view file's program in place, so its ID and edits stay."""
    existing = await manager.get_file_info(file_id)
    if existing is None or not is_openui_file(existing):
        raise ViewFileError(
            f"{file_id} is not an OpenUI view file in this workspace. "
            "Pass a new filename to save the view as a new file."
        )
    return await manager.replace_file_content(
        existing.id,
        source.encode(),
        mime_type=OPENUI_MIME_TYPE,
        metadata=openui_metadata(
            existing.metadata, view.view_id, session_id, view.summary
        ),
    )


async def write_view_file(
    manager: WorkspaceManager,
    session_id: str,
    *,
    content: bytes,
    filename: str,
    path: str | None,
    overwrite: bool,
) -> WorkspaceFile | ErrorResponse:
    """write_workspace_file for a ``.openui`` name: validated like render_ui."""
    source = _program_text(content, session_id)
    if isinstance(source, ErrorResponse):
        return source
    view = await validate_view(
        source, session_id=session_id, retry=WRITE_RETRY, fallback=WRITE_FALLBACK
    )
    if isinstance(view, ErrorResponse):
        return view
    name = openui_filename(filename)
    if path is not None and not is_openui_name(path):
        path = f"{path.rstrip('/')}{OPENUI_EXTENSION}"
    file_id = await openui_files_db().find_openui_view_file(
        manager.user_id, session_id, view.view_id
    )
    if file_id is None and overwrite:
        existing = await manager.get_file_info_by_path(path or f"/{name}")
        if existing is not None and is_openui_file(existing):
            file_id = existing.id
    if file_id is not None:
        return await revise_view_file(manager, file_id, source, view, session_id)
    created = await manager.write_file(
        content=source.encode(),
        filename=name,
        path=path,
        mime_type=OPENUI_MIME_TYPE,
        overwrite=overwrite,
        metadata=openui_metadata(_AGENT_ORIGIN, view.view_id, session_id, view.summary),
    )
    await _move_chat_edits(manager.user_id, session_id, view.view_id, created)
    return created


async def read_view_file(
    user_id: str, session_id: str, file: WorkspaceFile, content: bytes
) -> OpenUIFileContentResponse:
    source = content.decode("utf-8", errors="replace")
    summary = openui_summary(file.metadata) or await _computed_summary(source)
    edits = await openui_files_db().get_openui_file_state(user_id, file.id)
    note = ""
    if edits is not None and len(json.dumps(edits).encode()) > _MAX_EDITS_BYTES:
        edits, note = None, " The user's edits are too large to include."
    return OpenUIFileContentResponse(
        file_id=file.id,
        name=file.name,
        path=file.path,
        mime_type=OPENUI_MIME_TYPE,
        view_id=openui_view_id(file.metadata),
        summary=summary,
        source=source,
        user_edits=edits,
        message=(
            f"Read OpenUI view {file.name}. summary is its generated text, "
            "source the program, user_edits the user's current field values "
            "and ticks. To revise it, call tool:render_ui with "
            f'file="workspace://{file.id}" and a complete program that keeps '
            f"its viewId.{note}"
        ),
        session_id=session_id,
    )


async def _computed_summary(source: str) -> str:
    """The summary of a file saved before summaries were stored with it."""
    try:
        return (await validate_openui_source(source)).summary
    except ValidatorUnavailable:
        return ""


def _program_text(content: bytes, session_id: str) -> str | ErrorResponse:
    try:
        source = content.decode("utf-8").strip()
        if len(source) > MAX_SOURCE_CHARS:
            raise ValueError("The program exceeds 60,000 characters")
        return check_workspace_program(source)
    except ValueError as error:
        return ErrorResponse(
            message=f"View not saved: {error}. An .openui file holds one "
            "complete OpenUI program beginning with root = Workspace(...); "
            'describe_capability(id="tool:render_ui") lists the components.',
            session_id=session_id,
        )


async def _chat_view_file(
    user_id: str, session_id: str, view_id: str, *, required: bool
) -> str | None:
    """The file already holding this view for the chat. A chat-only render
    must not fail on this lookup, so there it is best effort."""
    try:
        return await openui_files_db().find_openui_view_file(
            user_id, session_id, view_id
        )
    except Exception:
        if required:
            raise
        logger.warning("Saved view lookup failed", exc_info=True)
        return None


async def _move_chat_edits(
    user_id: str, session_id: str, view_id: str, file: WorkspaceFile
) -> None:
    """A new file for a view takes over the chat's unsaved edits to it. The
    file is already written, so a failure here must not fail the call."""
    try:
        await openui_files_db().move_chat_edits_to_file(
            user_id, session_id, view_id, file.id
        )
    except Exception:
        logger.warning("Moving chat edits onto a new view file failed", exc_info=True)
