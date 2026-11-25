"""Tool for reading the lists in the user's business profile in full."""

from typing import Any

from backend.copilot.model import ChatSession
from backend.data.db_accessors import understanding_db
from backend.data.understanding import CONTEXT_LIST_ITEMS, LIST_FIELDS

from .base import BaseTool
from .models import ErrorResponse, ToolResponseBase, UnderstandingResponse


class ReadUnderstandingTool(BaseTool):
    """Returns every item the user's business profile holds in one or all lists."""

    @property
    def name(self) -> str:
        return "read_understanding"

    @property
    def description(self) -> str:
        return (
            "Read the user's business profile lists in full, oldest first; "
            f"the chat context shows each list's newest {CONTEXT_LIST_ITEMS} only."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "field": {
                    "type": "string",
                    "enum": list(LIST_FIELDS),
                    "description": "The list to read; omit for every list.",
                },
            },
            "required": [],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        field: str | None = None,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id
        if not user_id:
            return ErrorResponse(
                message="Authentication required to read the business profile.",
                session_id=session_id,
            )
        if field is not None and field not in LIST_FIELDS:
            return ErrorResponse(
                message=f"Unknown list {field!r}; use one of {', '.join(LIST_FIELDS)}.",
                session_id=session_id,
            )

        fields = [field] if field else list(LIST_FIELDS)
        understanding = await understanding_db().get_business_understanding(user_id)
        lists: dict[str, list[str]] = (
            understanding.model_dump(include=set(fields))
            if understanding
            else {name: [] for name in fields}
        )
        counts = ", ".join(f"{name}: {len(items)}" for name, items in lists.items())
        return UnderstandingResponse(
            message=f"Items per list: {counts}.",
            session_id=session_id,
            lists=lists,
        )
