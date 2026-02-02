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
from .response_model import StreamError, StreamFinish, StreamToolOutputAvailable
from .tools.models import ErrorResponse

logger = logging.getLogger(__name__)

# Tools that produce agent_json that needs to be saved to library
AGENT_GENERATION_TOOLS = {"create_agent", "edit_agent"}


def serialize_result(result: dict | str | None) -> str:
    """Serialize result to JSON string with sensible defaults.

    Args:
        result: The result to serialize (dict, string, or None)

    Returns:
        JSON string representation of the result
    """
    if isinstance(result, str):
        return result
    if result:
        return orjson.dumps(result).decode("utf-8")
    return '{"status": "completed"}'


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
        logger.warning(
            "[COMPLETION] Cannot save agent: no user_id in task"
        )
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
        return {
            "type": "error",
            "message": f"Agent was generated but failed to save: {str(e)}",
            "error": str(e),
            "agent_json": agent_json,  # Include the JSON so user can retry
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
    """
    # For agent generation tools, save the agent to library
    if task.tool_name in AGENT_GENERATION_TOOLS and isinstance(result, dict):
        result = await _save_agent_from_result(result, task.user_id, task.tool_name)

    # Serialize result for output
    result_output = result if result else {"status": "completed"}
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
    result_str = serialize_result(result)
    try:
        if prisma_client:
            # Use provided Prisma client (for consumer with its own connection)
            await prisma_client.chatmessage.update_many(
                where={
                    "sessionId": task.session_id,
                    "toolCallId": task.tool_call_id,
                },
                data={"content": result_str},
            )
            logger.info(
                f"[COMPLETION] Updated tool message for session {task.session_id}"
            )
        else:
            # Use service function (for webhook endpoint)
            await chat_service._update_pending_operation(
                session_id=task.session_id,
                tool_call_id=task.tool_call_id,
                result=result_str,
            )
    except Exception as e:
        logger.error(f"[COMPLETION] Failed to update tool message: {e}", exc_info=True)

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
    await stream_registry.publish_chunk(task.task_id, StreamFinish())

    # Update pending operation with error
    error_response = ErrorResponse(
        message=error_msg,
        error=error,
    )
    try:
        if prisma_client:
            # Use provided Prisma client (for consumer with its own connection)
            await prisma_client.chatmessage.update_many(
                where={
                    "sessionId": task.session_id,
                    "toolCallId": task.tool_call_id,
                },
                data={"content": error_response.model_dump_json()},
            )
            logger.info(
                f"[COMPLETION] Updated tool message with error for session {task.session_id}"
            )
        else:
            # Use service function (for webhook endpoint)
            await chat_service._update_pending_operation(
                session_id=task.session_id,
                tool_call_id=task.tool_call_id,
                result=error_response.model_dump_json(),
            )
    except Exception as e:
        logger.error(f"[COMPLETION] Failed to update tool message: {e}", exc_info=True)

    # Mark task as failed and release Redis lock
    await stream_registry.mark_task_completed(task.task_id, status="failed")
    try:
        await chat_service._mark_operation_completed(task.tool_call_id)
    except Exception as e:
        logger.error(f"[COMPLETION] Failed to mark operation completed: {e}")

    logger.info(f"[COMPLETION] Processed failure for task {task.task_id}: {error_msg}")
