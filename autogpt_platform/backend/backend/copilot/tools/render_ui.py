import logging
import re
from functools import cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from backend.copilot.config import ChatConfig
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
    summary: str = Field(min_length=1, max_length=10_000)

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
            "Present an interactive map, timeline, chart, comparison, checklist, "
            "or input form in "
            "the conversation using existing data. Buttons continue this conversation; "
            "rendering never performs external actions."
        )

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def is_available(self) -> bool:
        return ChatConfig().openui_enabled

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
                        "A self-contained plain-text account of the same findings or "
                        "questions. Saved with the view and shown if rendering fails "
                        "or the client cannot display interactive UI."
                    ),
                    "maxLength": 10_000,
                },
            },
            "required": ["source", "summary"],
        }

    async def _execute(
        self, user_id: str | None, session: ChatSession, **kwargs: object
    ) -> ToolResponseBase:
        if not self.is_available or not user_id or user_id != session.user_id:
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
                    "Workspace(...), at most 60,000 characters, and a nonempty "
                    "plain-text summary of at most 10,000 characters."
                ),
                session_id=session.session_id,
            )
        response = RenderUIResponse(
            source=request.source,
            message=request.summary,
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
            "Correct these issues and retry render_ui with a complete source "
            "and matching plain-text summary. Preserve the user's data and constraints.",
            session_id=response.session_id,
        )
    return response
