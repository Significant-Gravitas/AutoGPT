"""CheckOperationStatusTool — query the status of a long-running operation."""

import logging
from typing import Any

from backend.copilot.model import ChatSession
from backend.copilot.tools.base import BaseTool
from backend.copilot.tools.models import ErrorResponse, ResponseType, ToolResponseBase

logger = logging.getLogger(__name__)


class OperationStatusResponse(ToolResponseBase):
    """Response for check_operation_status tool."""

    type: ResponseType = ResponseType.OPERATION_STATUS
    task_id: str
    operation_id: str
    status: str  # "running", "completed", "failed"
    tool_name: str | None = None
    message: str = ""


class CheckOperationStatusTool(BaseTool):
    """Check the status of a long-running operation (create_agent, edit_agent, etc.).

    The CoPilot uses this tool to report back to the user whether an
    operation that was started earlier has completed, failed, or is still
    running.
    """

    @property
    def name(self) -> str:
        return "check_operation_status"

    @property
    def description(self) -> str:
        return (
            "Check the current status of a long-running operation such as "
            "create_agent or edit_agent. Accepts either an operation_id or "
            "task_id from a previous operation_started response. "
            "Returns the current status: running, completed, or failed."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation_id": {
                    "type": "string",
                    "description": (
                        "The operation_id from an operation_started response."
                    ),
                },
                "task_id": {
                    "type": "string",
                    "description": (
                        "The task_id from an operation_started response. "
                        "Used as fallback if operation_id is not provided."
                    ),
                },
            },
            "required": [],
        }

    @property
    def requires_auth(self) -> bool:
        return False

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        from backend.copilot import stream_registry

        operation_id = (kwargs.get("operation_id") or "").strip()
        task_id = (kwargs.get("task_id") or "").strip()

        if not operation_id and not task_id:
            return ErrorResponse(
                message="Please provide an operation_id or task_id.",
                error="missing_parameter",
            )

        task = None
        if operation_id:
            task = await stream_registry.find_task_by_operation_id(operation_id)
        if task is None and task_id:
            task = await stream_registry.get_task(task_id)

        if task is None:
            # Task not in Redis — it may have already expired (TTL).
            # Check conversation history for the result instead.
            return ErrorResponse(
                message=(
                    "Operation not found — it may have already completed and "
                    "expired from the status tracker. Check the conversation "
                    "history for the result."
                ),
                error="not_found",
            )

        status_messages = {
            "running": (
                f"The {task.tool_name or 'operation'} is still running. "
                "Please wait for it to complete."
            ),
            "completed": (
                f"The {task.tool_name or 'operation'} has completed successfully."
            ),
            "failed": f"The {task.tool_name or 'operation'} has failed.",
        }

        return OperationStatusResponse(
            task_id=task.task_id,
            operation_id=task.operation_id,
            status=task.status,
            tool_name=task.tool_name,
            message=status_messages.get(task.status, f"Status: {task.status}"),
        )
