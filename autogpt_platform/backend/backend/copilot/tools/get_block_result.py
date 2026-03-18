"""Tool for retrieving the result of a background block execution."""

import asyncio
import logging
from typing import Any

from backend.copilot.model import ChatSession

from .base import BaseTool
from .models import (
    BlockJobResultResponse,
    BlockOutputResponse,
    ErrorResponse,
    ToolResponseBase,
)

logger = logging.getLogger(__name__)


class GetBlockResultTool(BaseTool):
    """Await and return the output of a background block job started by run_block_async."""

    @property
    def name(self) -> str:
        return "get_block_result"

    @property
    def description(self) -> str:
        return (
            "Retrieve the output of a block execution started with run_block_async. "
            "Provide the job_id returned by run_block_async. "
            "This call blocks until the block finishes, so issue all run_block_async "
            "calls first, then collect results with get_block_result."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "The job_id returned by run_block_async.",
                },
            },
            "required": ["job_id"],
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
        job_id = kwargs.get("job_id", "").strip()
        session_id = session.session_id

        if not job_id:
            return ErrorResponse(
                message="Please provide a job_id", session_id=session_id
            )

        if not user_id:
            return ErrorResponse(
                message="Authentication required", session_id=session_id
            )

        from backend.copilot.sdk.tool_adapter import (
            get_background_jobs,  # avoid circular import at module level
        )

        jobs = get_background_jobs()
        if jobs is None:
            return ErrorResponse(
                message="Background job store is not available in this session.",
                session_id=session_id,
            )

        task = jobs.get(job_id)
        if task is None:
            return ErrorResponse(
                message=f"No background job found with job_id='{job_id}'.",
                session_id=session_id,
            )

        try:
            result = await task
        except asyncio.CancelledError:
            jobs.pop(job_id, None)
            return BlockJobResultResponse(
                message="Block execution was cancelled.",
                session_id=session_id,
                job_id=job_id,
                block_id="",
                block_name="",
                success=False,
                error="Task was cancelled.",
            )
        except Exception as exc:
            logger.warning("Background block job %s failed: %s", job_id, exc)
            # Clean up the finished task
            jobs.pop(job_id, None)
            return BlockJobResultResponse(
                message=f"Block execution failed: {exc}",
                session_id=session_id,
                job_id=job_id,
                block_id="",
                block_name="",
                success=False,
                error=str(exc),
            )

        # Clean up the finished task
        jobs.pop(job_id, None)

        # execute_block returns ErrorResponse on failure — propagate it directly
        # rather than accessing .block_id/.outputs which only exist on BlockOutputResponse.
        if not isinstance(result, BlockOutputResponse):
            return result

        return BlockJobResultResponse(
            message="Block execution completed successfully.",
            session_id=session_id,
            job_id=job_id,
            block_id=result.block_id,
            block_name=result.block_name,
            outputs=result.outputs,
            success=result.success,
        )
