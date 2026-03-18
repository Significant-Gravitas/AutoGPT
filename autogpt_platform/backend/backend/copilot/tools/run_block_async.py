"""Tool for starting block execution as a background task."""

import asyncio
import logging
from typing import Any

from backend.blocks._base import AnyBlockSchema
from backend.copilot.model import ChatSession

from .base import BaseTool
from .helpers import (
    BlockPreparation,
    check_hitl_review,
    execute_block,
    get_inputs_from_schema,
    prepare_block_for_execution,
)
from .models import BlockJobStartedResponse, ErrorResponse, ToolResponseBase

logger = logging.getLogger(__name__)


def _log_task_exception(task: asyncio.Task, job_id: str) -> None:
    """Log unhandled exceptions from background block tasks.

    Without this callback, tasks that fail before get_block_result is called
    would emit a silent 'Task exception was never retrieved' warning.
    """
    if not task.cancelled() and task.exception() is not None:
        logger.warning(
            "Background block job %s raised an unhandled exception",
            job_id,
            exc_info=task.exception(),
        )


class RunBlockAsyncTool(BaseTool):
    """Start block execution as a background task and return a job_id immediately.

    This allows the caller to issue multiple run_block_async calls in quick
    succession (e.g. to run several blocks in parallel) and then collect the
    results with get_block_result once they are needed.
    """

    @property
    def name(self) -> str:
        return "run_block_async"

    @property
    def description(self) -> str:
        return (
            "Start a block execution in the background and return a job_id immediately. "
            "Use this instead of run_block when you want to run multiple blocks in parallel. "
            "IMPORTANT: You MUST call find_block first to get the block's 'id'. "
            "Call run_block_async for each block you want to run concurrently, then call "
            "get_block_result with each job_id to retrieve the outputs. "
            "If a block requires human review, use continue_run_block after approval."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "block_id": {
                    "type": "string",
                    "description": (
                        "The block's 'id' field from find_block results. "
                        "NEVER guess this - always get it from find_block first."
                    ),
                },
                "block_name": {
                    "type": "string",
                    "description": "The block's human-readable name from find_block results.",
                },
                "input_data": {
                    "type": "object",
                    "description": "Input values for the block.",
                },
            },
            "required": ["block_id", "block_name", "input_data"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs,
    ) -> ToolResponseBase:
        block_id = kwargs.get("block_id", "").strip()
        input_data = kwargs.get("input_data", {})
        session_id = session.session_id

        if not block_id:
            return ErrorResponse(
                message="Please provide a block_id", session_id=session_id
            )

        if not isinstance(input_data, dict):
            return ErrorResponse(
                message="input_data must be an object", session_id=session_id
            )

        if not user_id:
            return ErrorResponse(
                message="Authentication required", session_id=session_id
            )

        logger.info(f"Starting async block execution: {block_id} for user {user_id}")

        prep_or_err = await prepare_block_for_execution(
            block_id=block_id,
            input_data=input_data,
            user_id=user_id,
            session=session,
            session_id=session_id,
            execution_mode="background",
        )
        if isinstance(prep_or_err, ToolResponseBase):
            return prep_or_err
        prep: BlockPreparation = prep_or_err

        # Unlike run_block (which shows a schema preview), we require all inputs
        # upfront — there is no two-step UX for background execution.
        if not (prep.required_non_credential_keys <= prep.provided_input_keys):
            missing = sorted(
                prep.required_non_credential_keys - prep.provided_input_keys
            )
            return ErrorResponse(
                message=(
                    f"Block '{prep.block.name}' is missing required inputs: "
                    f"{', '.join(missing)}. Please provide all required inputs."
                ),
                session_id=session_id,
            )

        hitl_or_err = await check_hitl_review(prep, user_id, session_id)
        if isinstance(hitl_or_err, ToolResponseBase):
            return hitl_or_err
        synthetic_node_exec_id, input_data = hitl_or_err

        # Start execution as a background task and return immediately.
        import uuid

        job_id = uuid.uuid4().hex

        async def _run() -> Any:
            return await execute_block(
                block=prep.block,
                block_id=block_id,
                input_data=input_data,
                user_id=user_id,
                session_id=session_id,
                node_exec_id=synthetic_node_exec_id,
                matched_credentials=prep.matched_credentials,
            )

        task = asyncio.create_task(_run())
        task.add_done_callback(lambda t: _log_task_exception(t, job_id))

        # Lazy import to avoid circular dependency at module level:
        # tools/__init__.py → run_block_async → tool_adapter → tools/__init__.py
        from backend.copilot.sdk.tool_adapter import get_background_jobs

        jobs = get_background_jobs()
        if jobs is None:
            logger.warning(
                "Background job store not initialised for session %s; "
                "run_block_async results will not be retrievable.",
                session_id,
            )
        else:
            jobs[job_id] = task

        return BlockJobStartedResponse(
            message=(
                f"Block '{prep.block.name}' started in the background. "
                f"Call get_block_result with job_id='{job_id}' to retrieve the output."
            ),
            session_id=session_id,
            job_id=job_id,
            block_id=block_id,
            block_name=prep.block.name,
        )

    def _get_inputs_list(self, block: AnyBlockSchema) -> list[dict[str, Any]]:
        schema = block.input_schema.jsonschema()
        credentials_fields = set(block.input_schema.get_credentials_fields().keys())
        return get_inputs_from_schema(schema, exclude_fields=credentials_fields)
