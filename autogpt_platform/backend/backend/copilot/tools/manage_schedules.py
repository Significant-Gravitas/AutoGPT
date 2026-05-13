"""Tools for listing and deleting agent execution schedules."""

import logging
from typing import Any

from pydantic import BaseModel

from backend.api.features.library.db import get_library_agent
from backend.copilot.model import ChatSession
from backend.util.clients import get_scheduler_client
from backend.util.exceptions import NotAuthorizedError, NotFoundError

from .base import BaseTool
from .models import ErrorResponse, ResponseType, ToolResponseBase

logger = logging.getLogger(__name__)


class ScheduleSummary(BaseModel):
    """Summary of a single schedule."""

    schedule_id: str
    name: str
    cron: str
    timezone: str
    next_run_time: str
    graph_id: str
    graph_version: int


class ScheduleListResponse(ToolResponseBase):
    """Response containing a list of schedules."""

    type: ResponseType = ResponseType.SCHEDULE_LIST
    schedules: list[ScheduleSummary]


class ScheduleDeletedResponse(ToolResponseBase):
    """Response confirming a schedule was deleted."""

    type: ResponseType = ResponseType.SCHEDULE_DELETED
    schedule_id: str


class ListSchedulesTool(BaseTool):
    """List the user's existing scheduled agent executions.

    Use this to find a schedule before deleting it, or to show the user
    which schedules they currently have set up. Optionally filter by
    graph_id to list schedules for a specific agent.
    """

    @property
    def name(self) -> str:
        return "list_schedules"

    @property
    def description(self) -> str:
        return "List agent run schedules. Use before delete_schedule."

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "library_agent_id": {
                    "type": "string",
                    "description": "Filter by library agent.",
                },
                "graph_id": {
                    "type": "string",
                    "description": "Filter by graph.",
                },
            },
            "required": [],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id if session else None
        if not user_id:
            return ErrorResponse(
                message="Authentication required.",
                error="auth_required",
                session_id=session_id,
            )

        library_agent_id: str | None = kwargs.get("library_agent_id")
        graph_id: str | None = kwargs.get("graph_id")

        # Resolve library_agent_id → graph_id (also verifies ownership)
        if library_agent_id:
            try:
                lib_agent = await get_library_agent(
                    id=library_agent_id, user_id=user_id
                )
            except NotFoundError as e:
                return ErrorResponse(
                    message=f"Library agent not found: {e}",
                    error="library_agent_not_found",
                    session_id=session_id,
                )
            graph_id = lib_agent.graph_id

        jobs = await get_scheduler_client().get_execution_schedules(
            graph_id=graph_id,
            user_id=user_id,
        )

        schedules = [
            ScheduleSummary(
                schedule_id=job.id,
                name=job.name,
                cron=job.cron,
                timezone=job.timezone,
                next_run_time=job.next_run_time,
                graph_id=job.graph_id,
                graph_version=job.graph_version,
            )
            for job in jobs
        ]

        message = (
            f"Found {len(schedules)} schedule(s)."
            if schedules
            else "No schedules found."
        )
        return ScheduleListResponse(
            message=message,
            schedules=schedules,
            session_id=session_id,
        )


class DeleteScheduleTool(BaseTool):
    """Delete an agent run schedule.

    Use list_schedules first to find the schedule_id.
    """

    @property
    def name(self) -> str:
        return "delete_schedule"

    @property
    def description(self) -> str:
        return "Delete an agent run schedule by schedule_id."

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "schedule_id": {
                    "type": "string",
                    "description": "Schedule ID from list_schedules.",
                },
            },
            "required": ["schedule_id"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id if session else None
        if not user_id:
            return ErrorResponse(
                message="Authentication required.",
                error="auth_required",
                session_id=session_id,
            )

        schedule_id: str | None = kwargs.get("schedule_id")
        if not schedule_id:
            return ErrorResponse(
                message="schedule_id is required.",
                error="missing_schedule_id",
                session_id=session_id,
            )

        try:
            await get_scheduler_client().delete_schedule(
                schedule_id=schedule_id,
                user_id=user_id,
            )
        except NotFoundError as e:
            return ErrorResponse(
                message=f"Schedule not found: {e}",
                error="schedule_not_found",
                session_id=session_id,
            )
        except NotAuthorizedError as e:
            return ErrorResponse(
                message=f"Not authorized: {e}",
                error="not_authorized",
                session_id=session_id,
            )

        return ScheduleDeletedResponse(
            message=f"Schedule {schedule_id} deleted.",
            schedule_id=schedule_id,
            session_id=session_id,
        )
