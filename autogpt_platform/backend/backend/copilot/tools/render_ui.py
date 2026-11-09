"""RenderUITool - Show the user visual cards instead of markdown."""

from typing import Any

from pydantic import ValidationError

from backend.copilot.model import ChatSession
from backend.util.feature_flag import Flag, is_feature_enabled

from .base import BaseTool
from .models import ErrorResponse, ToolResponseBase
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
    text, so the numbers in a card would be lost to whoever asked.
    """
    hidden = frozenset({RENDER_UI_TOOL})
    metadata = session.metadata
    if metadata.source_platform or metadata.text_only_result or not session.user_id:
        return hidden
    enabled = await is_feature_enabled(Flag.COPILOT_VISUAL_ANSWERS, session.user_id)
    return frozenset() if enabled else hidden


def _describe(error: ValidationError) -> str:
    problems = [
        f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
        for item in error.errors(include_input=False, include_url=False)
    ]
    shown = "; ".join(problems[:_MAX_REPORTED_PROBLEMS])
    return f"Invalid blocks. {shown}"
