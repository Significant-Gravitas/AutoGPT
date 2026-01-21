"""Tool for capturing user business understanding incrementally."""

import logging
from typing import Any

from langfuse import observe

from backend.api.features.chat.model import ChatSession
from backend.data.understanding import (
    BusinessUnderstandingInput,
    upsert_business_understanding,
)

from .base import BaseTool
from .models import ErrorResponse, ToolResponseBase, UnderstandingUpdatedResponse

logger = logging.getLogger(__name__)


class AddUnderstandingTool(BaseTool):
    """Tool for capturing user's business understanding incrementally."""

    @property
    def name(self) -> str:
        return "add_understanding"

    @property
    def description(self) -> str:
        return """Capture and store information about the user's business context,
workflows, pain points, and automation goals. Call this tool whenever the user
shares information about their business. Each call incrementally adds to the
existing understanding - you don't need to provide all fields at once.

Use this to build a comprehensive profile that helps recommend better agents
and automations for the user's specific needs."""

    @property
    def parameters(self) -> dict[str, Any]:
        # Auto-generate from Pydantic model schema
        schema = BusinessUnderstandingInput.model_json_schema()
        properties = {}
        for field_name, field_schema in schema.get("properties", {}).items():
            prop: dict[str, Any] = {"description": field_schema.get("description", "")}
            # Handle anyOf for Optional types
            if "anyOf" in field_schema:
                for option in field_schema["anyOf"]:
                    if option.get("type") != "null":
                        prop["type"] = option.get("type", "string")
                        if "items" in option:
                            prop["items"] = option["items"]
                        break
            else:
                prop["type"] = field_schema.get("type", "string")
                if "items" in field_schema:
                    prop["items"] = field_schema["items"]
            properties[field_name] = prop
        return {"type": "object", "properties": properties, "required": []}

    @property
    def requires_auth(self) -> bool:
        """Requires authentication to store user-specific data."""
        return True

    @observe(as_type="tool", name="add_understanding")
    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        """
        Capture and store business understanding incrementally.

        Each call merges new data with existing understanding:
        - String fields are overwritten if provided
        - List fields are appended (with deduplication)
        """
        session_id = session.session_id

        if not user_id:
            return ErrorResponse(
                message="Authentication required to save business understanding.",
                session_id=session_id,
            )

        # Check if any data was provided
        if not any(v is not None for v in kwargs.values()):
            return ErrorResponse(
                message="Please provide at least one field to update.",
                session_id=session_id,
            )

        # Build input model from kwargs (only include fields defined in the model)
        valid_fields = set(BusinessUnderstandingInput.model_fields.keys())
        input_data = BusinessUnderstandingInput(
            **{k: v for k, v in kwargs.items() if k in valid_fields}
        )

        # Track which fields were updated
        updated_fields = [
            k for k, v in kwargs.items() if k in valid_fields and v is not None
        ]

        # Upsert with merge
        understanding = await upsert_business_understanding(user_id, input_data)

        # Build current understanding summary (filter out empty values)
        current_understanding = {
            k: v
            for k, v in understanding.model_dump(
                exclude={"id", "user_id", "created_at", "updated_at"}
            ).items()
            if v is not None and v != [] and v != ""
        }

        return UnderstandingUpdatedResponse(
            message=f"Updated understanding with: {', '.join(updated_fields)}. "
            "I now have a better picture of your business context.",
            session_id=session_id,
            updated_fields=updated_fields,
            current_understanding=current_understanding,
        )
