"""Tool for capturing user business understanding incrementally."""

import logging
from typing import Any

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
        return {
            "type": "object",
            "properties": {
                "user_name": {
                    "type": "string",
                    "description": "The user's name",
                },
                "job_title": {
                    "type": "string",
                    "description": "The user's job title (e.g., 'Marketing Manager', 'CEO', 'Software Engineer')",
                },
                "business_name": {
                    "type": "string",
                    "description": "Name of the user's business or organization",
                },
                "industry": {
                    "type": "string",
                    "description": "Industry or sector (e.g., 'e-commerce', 'healthcare', 'finance')",
                },
                "business_size": {
                    "type": "string",
                    "description": "Company size: '1-10', '11-50', '51-200', '201-1000', or '1000+'",
                },
                "user_role": {
                    "type": "string",
                    "description": "User's role in organization context (e.g., 'decision maker', 'implementer', 'end user')",
                },
                "key_workflows": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key business workflows (e.g., 'lead qualification', 'content publishing')",
                },
                "daily_activities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Regular daily activities the user performs",
                },
                "pain_points": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Current pain points or challenges",
                },
                "bottlenecks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Process bottlenecks slowing things down",
                },
                "manual_tasks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Manual or repetitive tasks that could be automated",
                },
                "automation_goals": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Desired automation outcomes or goals",
                },
                "current_software": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Software and tools currently in use",
                },
                "existing_automation": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Any existing automations or integrations",
                },
                "additional_notes": {
                    "type": "string",
                    "description": "Any other relevant context or notes",
                },
            },
            "required": [],
        }

    @property
    def requires_auth(self) -> bool:
        """Requires authentication to store user-specific data."""
        return True

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

        # Build input model
        input_data = BusinessUnderstandingInput(
            user_name=kwargs.get("user_name"),
            job_title=kwargs.get("job_title"),
            business_name=kwargs.get("business_name"),
            industry=kwargs.get("industry"),
            business_size=kwargs.get("business_size"),
            user_role=kwargs.get("user_role"),
            key_workflows=kwargs.get("key_workflows"),
            daily_activities=kwargs.get("daily_activities"),
            pain_points=kwargs.get("pain_points"),
            bottlenecks=kwargs.get("bottlenecks"),
            manual_tasks=kwargs.get("manual_tasks"),
            automation_goals=kwargs.get("automation_goals"),
            current_software=kwargs.get("current_software"),
            existing_automation=kwargs.get("existing_automation"),
            additional_notes=kwargs.get("additional_notes"),
        )

        # Track which fields were updated
        updated_fields = [k for k, v in kwargs.items() if v is not None]

        # Upsert with merge
        understanding = await upsert_business_understanding(user_id, input_data)

        # Build current understanding summary for the response
        current_understanding = {
            "user_name": understanding.user_name,
            "job_title": understanding.job_title,
            "business_name": understanding.business_name,
            "industry": understanding.industry,
            "business_size": understanding.business_size,
            "user_role": understanding.user_role,
            "key_workflows": understanding.key_workflows,
            "daily_activities": understanding.daily_activities,
            "pain_points": understanding.pain_points,
            "bottlenecks": understanding.bottlenecks,
            "manual_tasks": understanding.manual_tasks,
            "automation_goals": understanding.automation_goals,
            "current_software": understanding.current_software,
            "existing_automation": understanding.existing_automation,
            "additional_notes": understanding.additional_notes,
        }

        # Filter out empty values for cleaner response
        current_understanding = {
            k: v
            for k, v in current_understanding.items()
            if v is not None and v != [] and v != ""
        }

        return UnderstandingUpdatedResponse(
            message=f"Updated understanding with: {', '.join(updated_fields)}. "
            "I now have a better picture of your business context.",
            session_id=session_id,
            updated_fields=updated_fields,
            current_understanding=current_understanding,
        )
