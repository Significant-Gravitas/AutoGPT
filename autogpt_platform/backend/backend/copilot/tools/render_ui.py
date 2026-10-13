import logging
import re
from functools import cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from backend.copilot.model import ChatSession
from backend.copilot.tools.base import BaseTool
from backend.copilot.tools.models import ErrorResponse, ResponseType, ToolResponseBase
from backend.copilot.tools.openui_source import validate_complete_delimiters
from backend.copilot.tools.openui_validator import (
    ValidatorUnavailable,
    validate_openui_source,
)

logger = logging.getLogger(__name__)


class RenderUIInput(BaseModel):
    model_config = ConfigDict(strict=True, str_strip_whitespace=True)

    source: str = Field(min_length=1, max_length=60_000)
    summary: str = Field(default="", max_length=10_000)

    @field_validator("source")
    @classmethod
    def workspace_program(cls, value: str) -> str:
        if not re.match(r"root\s*=\s*Workspace\s*\(", value):
            raise ValueError("Start source with root = Workspace(...), without fences")
        validate_complete_delimiters(value)
        return value


class RenderUIResponse(ToolResponseBase):
    type: ResponseType = ResponseType.UI_RENDERED
    version: Literal[1] = 1
    source: str
    view_id: str | None = None


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
            "Present an inline artifact, connected dashboard, editable data grid, map, "
            "itinerary, task board, chart, comparison, live execution status, "
            "or input form in "
            "the conversation using existing data. Forms continue this conversation; "
            "rendering never performs external actions."
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
                    "maxLength": 60_000,
                },
                "summary": {
                    "type": "string",
                    "description": (
                        "Optional legacy summary. The saved fallback is generated "
                        "from the validated view, including its computed totals. "
                        "Put all findings, caveats and sources in the view itself."
                    ),
                    "maxLength": 10_000,
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
        if len(response.model_dump_json().encode()) > 70_000:
            return ErrorResponse(
                message="This view is too large. Use fewer rows or shorter descriptions.",
                session_id=session.session_id,
            )
        return await _validated_response(response)


async def _validated_response(response: RenderUIResponse) -> ToolResponseBase:
    try:
        validation = await validate_openui_source(response.source)
    except ValidatorUnavailable as error:
        logger.warning(f"OpenUI validation unavailable: {error}")
        return ErrorResponse(
            message="Interactive validation is temporarily unavailable. "
            "Provide the complete answer as plain text instead of retrying this view.",
            session_id=response.session_id,
        )
    if not validation.valid:
        return ErrorResponse(
            message=f"View not published: {validation.error} "
            'Correct these issues and retry run_capability(id="tool:render_ui", '
            "input={...}) with a complete source. "
            "Preserve the user's data and constraints; the summary is generated.",
            session_id=response.session_id,
        )
    if not validation.summary:
        return ErrorResponse(
            message="Interactive validation returned no text summary. "
            "Provide the complete answer as plain text instead.",
            session_id=response.session_id,
        )
    if not validation.view_id:
        return ErrorResponse(
            message="View not published: Workspace requires a stable viewId from "
            "the first render so edits can be saved. Retry with "
            'Workspace(title, description, children, "stable-view-id"). '
            "Reuse this ID when revising this task; use a new ID for a new task.",
            session_id=response.session_id,
        )
    response.message = validation.summary
    response.view_id = validation.view_id
    if len(response.model_dump_json().encode()) > 70_000:
        return ErrorResponse(
            message="This view is too large. Use fewer rows or shorter descriptions.",
            session_id=response.session_id,
        )
    return response
