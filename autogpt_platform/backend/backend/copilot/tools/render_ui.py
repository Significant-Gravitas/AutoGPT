import logging
from functools import cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from backend.api.features.store.exceptions import VirusDetectedError, VirusScanError
from backend.copilot.model import ChatSession
from backend.copilot.tools.base import BaseTool
from backend.copilot.tools.models import ErrorResponse, ResponseType, ToolResponseBase
from backend.copilot.tools.openui_files import (
    MAX_SOURCE_CHARS,
    RENDER_FALLBACK,
    RENDER_RETRY,
    check_workspace_program,
    save_view,
    validate_view,
)
from backend.data.workspace import OPENUI_MIME_TYPE, WorkspaceFile
from backend.data.workspace_scope import WorkspaceAccessDeniedError

logger = logging.getLogger(__name__)

_MAX_RESPONSE_BYTES = 70_000
# Room for the file fields, which are only known once the view is saved.
_FILE_FIELDS_BYTES = 1_000


class RenderUIInput(BaseModel):
    model_config = ConfigDict(strict=True, str_strip_whitespace=True)

    source: str = Field(min_length=1, max_length=MAX_SOURCE_CHARS)
    summary: str = Field(default="", max_length=10_000)
    file: str | None = Field(default=None, min_length=1, max_length=255)

    @field_validator("source")
    @classmethod
    def workspace_program(cls, value: str) -> str:
        return check_workspace_program(value)


class RenderUIResponse(ToolResponseBase):
    type: ResponseType = ResponseType.UI_RENDERED
    version: Literal[1] = 1
    source: str
    view_id: str | None = None
    file_id: str | None = None
    file_uri: str | None = None
    file_path: str | None = None
    file_name: str | None = None


@cache
def openui_library() -> str:
    return Path(__file__).with_name("openui_library.txt").read_text()


class RenderUITool(BaseTool):
    @property
    def name(self) -> str:
        return "render_ui"

    @property
    def description(self) -> str:
        return (
            "Present an inline view in the conversation using existing data: "
            "notes (Markdown prose), metrics, charts, tables, checklists, timelines, "
            "maps, itineraries, comparisons, result lists, artifacts or input forms. "
            "Forms continue this conversation; rendering never performs external "
            "actions."
        )

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": openui_library(),
                    "maxLength": MAX_SOURCE_CHARS,
                },
                "summary": {
                    "type": "string",
                    "description": (
                        "Optional legacy summary. The saved fallback is generated "
                        "from the validated view, including its computed values. "
                        "Put all findings, caveats and sources in the view itself."
                    ),
                    "maxLength": 10_000,
                },
                "file": {
                    "type": "string",
                    "description": (
                        "New filename to also save the view as a workspace "
                        "file, or workspace://<file_id> of a view file to "
                        "revise. Omit for chat-only views."
                    ),
                    "maxLength": 255,
                },
            },
            "required": ["source"],
        }

    async def _execute(
        self, user_id: str | None, session: ChatSession, **kwargs: object
    ) -> ToolResponseBase:
        if not user_id or user_id != session.user_id:
            return ErrorResponse(
                message="Interactive views are not available for this session.",
                session_id=session.session_id,
            )
        try:
            request = RenderUIInput.model_validate(kwargs)
        except ValidationError as error:
            issue = error.errors(include_input=False, include_url=False)[0]
            return ErrorResponse(
                message=(
                    f"{issue['msg']}. "
                    "Provide a complete OpenUI program beginning with root = "
                    "Workspace(...), at most 60,000 characters. Any legacy "
                    "summary must be at most 10,000 characters."
                ),
                session_id=session.session_id,
            )
        response = RenderUIResponse(
            source=request.source,
            message="",
            session_id=session.session_id,
        )
        if _too_large(response):
            return _too_large_error(session.session_id)
        published = await _validated_response(response)
        if isinstance(published, ErrorResponse):
            return published
        return await _saved_response(user_id, published, request.file)


async def _validated_response(
    response: RenderUIResponse,
) -> RenderUIResponse | ErrorResponse:
    session_id = response.session_id or ""
    view = await validate_view(
        response.source,
        session_id=session_id,
        retry=RENDER_RETRY,
        fallback=RENDER_FALLBACK,
    )
    if isinstance(view, ErrorResponse):
        return view
    response.message = view.summary
    response.view_id = view.view_id
    if _too_large(response, reserve=_FILE_FIELDS_BYTES):
        return _too_large_error(session_id)
    return response


async def _saved_response(
    user_id: str, response: RenderUIResponse, file: str | None
) -> RenderUIResponse | ErrorResponse:
    """Save the validated view when asked to, or when this chat already
    saved it, and point the response at the file."""
    session_id = response.session_id or ""
    assert response.view_id is not None
    try:
        saved = await save_view(
            user_id,
            session_id,
            source=response.source,
            view_id=response.view_id,
            file=file,
        )
    except (
        ValueError,
        FileNotFoundError,
        WorkspaceAccessDeniedError,
        VirusDetectedError,
        VirusScanError,
    ) as error:
        return _not_saved_error(str(error), session_id)
    if saved is not None:
        _attach_file(response, saved)
    return response


def _attach_file(response: RenderUIResponse, file: WorkspaceFile) -> None:
    response.file_id = file.id
    response.file_uri = f"workspace://{file.id}#{OPENUI_MIME_TYPE}"
    response.file_path = file.path
    response.file_name = file.name


def _too_large(response: RenderUIResponse, reserve: int = 0) -> bool:
    size = len(response.model_dump_json().encode())
    return size > _MAX_RESPONSE_BYTES - reserve


def _too_large_error(session_id: str) -> ErrorResponse:
    return ErrorResponse(
        message="This view is too large. Use fewer rows or shorter descriptions.",
        session_id=session_id,
    )


def _not_saved_error(reason: str, session_id: str) -> ErrorResponse:
    return ErrorResponse(
        message=f"View not published: it could not be saved as a file. {reason} "
        "Fix the file argument, or omit it to show the view in chat only.",
        session_id=session_id,
    )
