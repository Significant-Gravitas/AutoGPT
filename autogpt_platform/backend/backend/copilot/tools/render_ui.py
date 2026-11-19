"""RenderUITool - Show the user visual cards instead of markdown."""

from typing import Any

import orjson
from pydantic import ValidationError

from backend.copilot.model import ChatSession
from backend.copilot.permissions import CopilotPermissions
from backend.util.feature_flag import Flag, is_feature_enabled

from .base import BaseTool
from .models import ErrorResponse, ResponseType, ToolResponseBase
from .render_ui_models import RenderUIInput, UIBlocksResponse, render_ui_parameters

RENDER_UI_TOOL = "render_ui"
_MAX_REPORTED_PROBLEMS = 5


class RenderUITool(BaseTool):
    """Validate a list of blocks and hand them to the chat to draw."""

    @property
    def name(self) -> str:
        return RENDER_UI_TOOL

    @property
    def description(self) -> str:
        return (
            "Show the user visual cards: numbers, charts, workflow steps, images, "
            "timelines, tables and one-click options. Use when those are the "
            "point of the answer, after one or two plain sentences. Don't repeat "
            "in text what the cards show, and keep short or prose answers in "
            "text. Say 'workflow' in card text, never 'agent' or 'graph'."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return render_ui_parameters()

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs: Any,
    ) -> ToolResponseBase:
        del user_id
        try:
            parsed = RenderUIInput.model_validate({"blocks": kwargs.get("blocks")})
        except ValidationError as e:
            return ErrorResponse(
                message=_describe(e),
                error="invalid_blocks",
                session_id=session.session_id,
            ).from_outside()
        # Otto wrote these itself this turn, so there is nothing outside to judge.
        return UIBlocksResponse(
            message="Shown to the user.",
            session_id=session.session_id,
            blocks=parsed.blocks,
        ).from_outside()


async def render_ui_disabled_tools(session: ChatSession) -> frozenset[str]:
    """Hide ``render_ui`` where nobody would see its cards.

    Chat platforms can't draw them, and text-only callers keep only the final
    text, so the numbers in a card would be lost to whoever asked. A session
    another session opened reports back as text too, unless it was handed off
    and became the user's own chat. Reading that off the provenance also
    covers threads opened before ``text_only_result`` existed.
    """
    hidden = frozenset({RENDER_UI_TOOL})
    metadata = session.metadata
    reports_back = (
        metadata.delegated_by_session_id is not None
        and not metadata.handed_off_from_expert_id
    )
    if (
        metadata.source_platform
        or metadata.text_only_result
        or reports_back
        or not session.user_id
    ):
        return hidden
    enabled = await is_feature_enabled(Flag.COPILOT_VISUAL_ANSWERS, session.user_id)
    return frozenset() if enabled else hidden


def shows_cards(output: Any) -> bool:
    """Whether a ``render_ui`` call's output is cards rather than a refusal."""
    if isinstance(output, str):
        try:
            output = orjson.loads(output)
        except orjson.JSONDecodeError:
            return False
    return (
        isinstance(output, dict) and output.get("type") == ResponseType.UI_BLOCKS.value
    )


def without_render_ui(
    permissions: CopilotPermissions | None,
) -> CopilotPermissions:
    """*permissions* with ``render_ui`` refused, for one turn whose caller
    keeps only the final text.

    The session flag can't do this alone: a caller may resume a session made
    before the flag existed, or one a person reads. An allow-list naming only
    ``render_ui`` is left alone, since emptying it would allow every tool.
    """
    base = permissions or CopilotPermissions()
    if not base.tools:
        return base.model_copy(
            update={"tools": [RENDER_UI_TOOL], "tools_exclude": True}
        )
    if base.tools_exclude:
        if RENDER_UI_TOOL in base.tools:
            return base
        return base.model_copy(update={"tools": [*base.tools, RENDER_UI_TOOL]})
    kept = [tool for tool in base.tools if tool != RENDER_UI_TOOL]
    return base.model_copy(update={"tools": kept}) if kept else base


def _describe(error: ValidationError) -> str:
    problems = [
        f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
        for item in error.errors(include_input=False, include_url=False)
    ]
    shown = "; ".join(problems[:_MAX_REPORTED_PROBLEMS])
    return f"Invalid blocks. {shown}"
