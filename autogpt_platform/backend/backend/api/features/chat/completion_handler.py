"""Shared completion handling for operation success and failure.

This module provides common logic for handling operation completion from both:
- The Redis Streams consumer (completion_consumer.py)
- The HTTP webhook endpoint (routes.py)
"""

import logging
from typing import Any

import orjson
from prisma import Prisma

from . import service as chat_service
from . import stream_registry
from .response_model import StreamError, StreamToolOutputAvailable
from .tools.models import ErrorResponse

logger = logging.getLogger(__name__)

# Tools that produce agent_json that needs to be saved to library
AGENT_GENERATION_TOOLS = {"create_agent", "edit_agent"}

# Keys that should be stripped from agent_json when returning in error responses
SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "api_secret",
        "password",
        "secret",
        "credentials",
        "credential",
        "token",
        "access_token",
        "refresh_token",
        "private_key",
        "privatekey",
        "auth",
        "authorization",
    }
)


def _sanitize_agent_json(obj: Any) -> Any:
    """Recursively sanitize agent_json by removing sensitive keys.

    Args:
        obj: The object to sanitize (dict, list, or primitive)

    Returns:
        Sanitized copy with sensitive keys removed/redacted
    """
    if isinstance(obj, dict):
        return {
            k: "[REDACTED]" if k.lower() in SENSITIVE_KEYS else _sanitize_agent_json(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [_sanitize_agent_json(item) for item in obj]
    else:
        return obj


class ToolMessageUpdateError(Exception):
    """Raised when updating a tool message in the database fails."""

    pass


async def _update_tool_message(
    session_id: str,
    tool_call_id: str,
    content: str,
    prisma_client: Prisma | None,
) -> None:
    """Update tool message in database.

    Args:
        session_id: The session ID
        tool_call_id: The tool call ID to update
        content: The new content for the message
        prisma_client: Optional Prisma client. If None, uses chat_service.

    Raises:
        ToolMessageUpdateError: If the database update fails. The caller should
            handle this to avoid marking the task as completed with inconsistent state.
    """
    try:
        if prisma_client:
            # Use provided Prisma client (for consumer with its own connection)
            updated_count = await prisma_client.chatmessage.update_many(
                where={
                    "sessionId": session_id,
                    "toolCallId": tool_call_id,
                },
                data={"content": content},
            )
            # Check if any rows were updated - 0 means message not found
            if updated_count == 0:
                raise ToolMessageUpdateError(
                    f"No message found with tool_call_id={tool_call_id} in session {session_id}"
                )
        else:
            # Use service function (for webhook endpoint)
            await chat_service._update_pending_operation(
                session_id=session_id,
                tool_call_id=tool_call_id,
                result=content,
            )
    except ToolMessageUpdateError:
        raise
    except Exception as e:
        logger.error(f"[COMPLETION] Failed to update tool message: {e}", exc_info=True)
        raise ToolMessageUpdateError(
            f"Failed to update tool message for tool_call_id={tool_call_id}: {e}"
        ) from e


def serialize_result(result: dict | list | str | int | float | bool | None) -> str:
    """Serialize result to JSON string with sensible defaults.

    Args:
        result: The result to serialize. Can be a dict, list, string,
            number, boolean, or None.

    Returns:
        JSON string representation of the result. Returns '{"status": "completed"}'
        only when result is explicitly None.
    """
    if isinstance(result, str):
        return result
    if result is None:
        return '{"status": "completed"}'
    return orjson.dumps(result).decode("utf-8")


async def _save_agent_from_result(
    result: dict[str, Any],
    user_id: str | None,
    tool_name: str,
) -> dict[str, Any]:
    """Save agent to library if result contains agent_json.

    Args:
        result: The result dict that may contain agent_json
        user_id: The user ID to save the agent for
        tool_name: The tool name (create_agent or edit_agent)

    Returns:
        Updated result dict with saved agent details, or original result if no agent_json
    """
    if not user_id:
        logger.warning("[COMPLETION] Cannot save agent: no user_id in task")
        return result

    agent_json = result.get("agent_json")
    if not agent_json:
        logger.warning(
            f"[COMPLETION] {tool_name} completed but no agent_json in result"
        )
        return result

    try:
        from .tools.agent_generator import save_agent_to_library

        is_update = tool_name == "edit_agent"
        created_graph, library_agent = await save_agent_to_library(
            agent_json, user_id, is_update=is_update
        )

        logger.info(
            f"[COMPLETION] Saved agent '{created_graph.name}' to library "
            f"(graph_id={created_graph.id}, library_agent_id={library_agent.id})"
        )

        # Return a response similar to AgentSavedResponse
        return {
            "type": "agent_saved",
            "message": f"Agent '{created_graph.name}' has been saved to your library!",
            "agent_id": created_graph.id,
            "agent_name": created_graph.name,
            "library_agent_id": library_agent.id,
            "library_agent_link": f"/library/agents/{library_agent.id}",
            "agent_page_link": f"/build?flowID={created_graph.id}",
        }
    except Exception as e:
        logger.error(
            f"[COMPLETION] Failed to save agent to library: {e}",
            exc_info=True,
        )
        # Return error but don't fail the whole operation
        # Sanitize agent_json to remove sensitive keys before returning
        return {
            "type": "error",
            "message": f"Agent was generated but failed to save: {str(e)}",
            "error": str(e),
            "agent_json": _sanitize_agent_json(agent_json),
        }


async def process_operation_success(
    task: stream_registry.ActiveTask,
    result: dict | str | None,
    prisma_client: Prisma | None = None,
) -> None:
    """Handle successful operation completion.

    Publishes the result to the stream registry, updates the database,
    generates LLM continuation, and marks the task as completed.

    Args:
        task: The active task that completed
        result: The result data from the operation
        prisma_client: Optional Prisma client for database operations.
            If None, uses chat_service._update_pending_operation instead.

    Raises:
        ToolMessageUpdateError: If the database update fails. The task will be
            marked as failed instead of completed to avoid inconsistent state.
    """
    # For agent generation tools, save the agent to library
    if task.tool_name in AGENT_GENERATION_TOOLS and isinstance(result, dict):
        result = await _save_agent_from_result(result, task.user_id, task.tool_name)

    # Serialize result for output (only substitute default when result is exactly None)
    result_output = result if result is not None else {"status": "completed"}
    output_str = (
        result_output
        if isinstance(result_output, str)
        else orjson.dumps(result_output).decode("utf-8")
    )

    # Publish result to stream registry
    await stream_registry.publish_chunk(
        task.task_id,
        StreamToolOutputAvailable(
            toolCallId=task.tool_call_id,
            toolName=task.tool_name,
            output=output_str,
            success=True,
        ),
    )

    # Update pending operation in database
    # If this fails, we must not continue to mark the task as completed
    result_str = serialize_result(result)
    try:
        await _update_tool_message(
            session_id=task.session_id,
            tool_call_id=task.tool_call_id,
            content=result_str,
            prisma_client=prisma_client,
        )
    except ToolMessageUpdateError:
        # DB update failed - mark task as failed to avoid inconsistent state
        logger.error(
            f"[COMPLETION] DB update failed for task {task.task_id}, "
            "marking as failed instead of completed"
        )
        await stream_registry.publish_chunk(
            task.task_id,
            StreamError(errorText="Failed to save operation result to database"),
        )
        await stream_registry.mark_task_completed(task.task_id, status="failed")
        raise

    # Generate LLM continuation with streaming
    try:
        await chat_service._generate_llm_continuation_with_streaming(
            session_id=task.session_id,
            user_id=task.user_id,
            task_id=task.task_id,
        )
    except Exception as e:
        logger.error(
            f"[COMPLETION] Failed to generate LLM continuation: {e}",
            exc_info=True,
        )

    # Mark task as completed and release Redis lock
    await stream_registry.mark_task_completed(task.task_id, status="completed")
    try:
        await chat_service._mark_operation_completed(task.tool_call_id)
    except Exception as e:
        logger.error(f"[COMPLETION] Failed to mark operation completed: {e}")

    logger.info(
        f"[COMPLETION] Successfully processed completion for task {task.task_id}"
    )


async def process_operation_failure(
    task: stream_registry.ActiveTask,
    error: str | None,
    prisma_client: Prisma | None = None,
) -> None:
    """Handle failed operation completion.

    Publishes the error to the stream registry, updates the database with
    the error response, and marks the task as failed.

    Args:
        task: The active task that failed
        error: The error message from the operation
        prisma_client: Optional Prisma client for database operations.
            If None, uses chat_service._update_pending_operation instead.
    """
    error_msg = error or "Operation failed"

    # Publish error to stream registry
    await stream_registry.publish_chunk(
        task.task_id,
        StreamError(errorText=error_msg),
    )

    # Update pending operation with error
    # If this fails, we still continue to mark the task as failed
    error_response = ErrorResponse(
        message=error_msg,
        error=error,
    )
    try:
        await _update_tool_message(
            session_id=task.session_id,
            tool_call_id=task.tool_call_id,
            content=error_response.model_dump_json(),
            prisma_client=prisma_client,
        )
    except ToolMessageUpdateError:
        # DB update failed - log but continue with cleanup
        logger.error(
            f"[COMPLETION] DB update failed while processing failure for task {task.task_id}, "
            "continuing with cleanup"
        )

    # Mark task as failed and release Redis lock
    await stream_registry.mark_task_completed(task.task_id, status="failed")
    try:
        await chat_service._mark_operation_completed(task.tool_call_id)
    except Exception as e:
        logger.error(f"[COMPLETION] Failed to mark operation completed: {e}")

    logger.info(f"[COMPLETION] Processed failure for task {task.task_id}: {error_msg}")
