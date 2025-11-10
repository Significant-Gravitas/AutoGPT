import asyncio
import logging
from datetime import datetime
from typing import Optional

from autogpt_libs.auth import get_user_id, requires_admin_user
from fastapi import APIRouter, HTTPException, Security
from pydantic import BaseModel, Field

from backend.data.execution import (
    ExecutionStatus,
    GraphExecutionMeta,
    get_graph_executions,
    update_graph_execution_stats,
)
from backend.data.model import GraphExecutionStats
from backend.executor.activity_status_generator import (
    generate_activity_status_for_execution,
)
from backend.executor.manager import get_db_async_client
from backend.util.settings import Settings

logger = logging.getLogger(__name__)


class ExecutionAnalyticsRequest(BaseModel):
    graph_id: str = Field(..., description="Graph ID to analyze")
    graph_version: Optional[int] = Field(None, description="Optional graph version")
    user_id: Optional[str] = Field(None, description="Optional user ID filter")
    created_after: Optional[datetime] = Field(
        None, description="Optional created date lower bound"
    )
    model_name: Optional[str] = Field(
        "gpt-4o-mini", description="Model to use for generation"
    )
    batch_size: int = Field(
        10, description="Batch size for concurrent processing", le=25, ge=1
    )


class ExecutionAnalyticsResult(BaseModel):
    agent_id: str
    version_id: int
    user_id: str
    exec_id: str
    summary_text: Optional[str]
    score: Optional[float]
    status: str  # "success", "failed", "skipped"
    error_message: Optional[str] = None


class ExecutionAnalyticsResponse(BaseModel):
    total_executions: int
    processed_executions: int
    successful_analytics: int
    failed_analytics: int
    skipped_executions: int
    results: list[ExecutionAnalyticsResult]


router = APIRouter(
    prefix="/admin",
    tags=["admin", "execution_analytics"],
    dependencies=[Security(requires_admin_user)],
)


@router.post(
    "/execution_analytics",
    response_model=ExecutionAnalyticsResponse,
    summary="Generate Execution Analytics",
)
async def generate_execution_analytics(
    request: ExecutionAnalyticsRequest,
    admin_user_id: str = Security(get_user_id),
):
    """
    Generate activity summaries and correctness scores for graph executions.

    This endpoint:
    1. Fetches all completed executions matching the criteria
    2. Identifies executions missing activity_status or correctness_score
    3. Generates missing data using AI in batches
    4. Updates the database with new stats
    5. Returns a detailed report of the analytics operation
    """
    logger.info(
        f"Admin user {admin_user_id} starting execution analytics generation for graph {request.graph_id}"
    )

    try:
        # Validate model configuration
        settings = Settings()
        if not settings.secrets.openai_internal_api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")

        # Get database client
        db_client = get_db_async_client()

        # Fetch executions to process
        executions = await get_graph_executions(
            graph_id=request.graph_id,
            user_id=request.user_id,
            created_time_gte=request.created_after,
            statuses=[
                ExecutionStatus.COMPLETED,
                ExecutionStatus.FAILED,
                ExecutionStatus.TERMINATED,
            ],  # Only process finished executions
        )

        logger.info(
            f"Found {len(executions)} total executions for graph {request.graph_id}"
        )

        # Filter executions that need analytics generation (missing activity_status or correctness_score)
        executions_to_process = []
        for execution in executions:
            if (
                not execution.stats
                or not execution.stats.activity_status
                or execution.stats.correctness_score is None
            ):

                # If version is specified, filter by it
                if (
                    request.graph_version is None
                    or execution.graph_version == request.graph_version
                ):
                    executions_to_process.append(execution)

        logger.info(
            f"Found {len(executions_to_process)} executions needing analytics generation"
        )

        # Create results for ALL executions - processed and skipped
        results = []
        successful_count = 0
        failed_count = 0

        # Process executions that need analytics generation
        if executions_to_process:
            total_batches = len(
                range(0, len(executions_to_process), request.batch_size)
            )

            for batch_idx, i in enumerate(
                range(0, len(executions_to_process), request.batch_size)
            ):
                batch = executions_to_process[i : i + request.batch_size]
                logger.info(
                    f"Processing batch {batch_idx + 1}/{total_batches} with {len(batch)} executions"
                )

                batch_results = await _process_batch(
                    batch, request.model_name or "gpt-4o-mini", db_client
                )

                for result in batch_results:
                    results.append(result)
                    if result.status == "success":
                        successful_count += 1
                    elif result.status == "failed":
                        failed_count += 1

                # Small delay between batches to avoid overwhelming the LLM API
                if batch_idx < total_batches - 1:  # Don't delay after the last batch
                    await asyncio.sleep(2)

        # Add ALL executions to results (both processed and skipped)
        for execution in executions:
            # Skip if already processed (added to results above)
            if execution in executions_to_process:
                continue

            results.append(
                ExecutionAnalyticsResult(
                    agent_id=execution.graph_id,
                    version_id=execution.graph_version,
                    user_id=execution.user_id,
                    exec_id=execution.id,
                    summary_text=(
                        execution.stats.activity_status if execution.stats else None
                    ),
                    score=(
                        execution.stats.correctness_score if execution.stats else None
                    ),
                    status="skipped",
                    error_message=None,  # Not an error - just already processed
                )
            )

        response = ExecutionAnalyticsResponse(
            total_executions=len(executions),
            processed_executions=len(executions_to_process),
            successful_analytics=successful_count,
            failed_analytics=failed_count,
            skipped_executions=len(executions) - len(executions_to_process),
            results=results,
        )

        logger.info(
            f"Analytics generation completed: {successful_count} successful, {failed_count} failed, "
            f"{response.skipped_executions} skipped"
        )

        return response

    except Exception as e:
        logger.exception(f"Error during execution analytics generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _process_batch(
    executions, model_name: str, db_client
) -> list[ExecutionAnalyticsResult]:
    """Process a batch of executions concurrently."""

    async def process_single_execution(execution) -> ExecutionAnalyticsResult:
        try:
            # Generate activity status and score using the specified model
            # Convert stats to GraphExecutionStats if needed
            if execution.stats:
                if isinstance(execution.stats, GraphExecutionMeta.Stats):
                    stats_for_generation = execution.stats.to_db()
                else:
                    # Already GraphExecutionStats
                    stats_for_generation = execution.stats
            else:
                stats_for_generation = GraphExecutionStats()

            activity_response = await generate_activity_status_for_execution(
                graph_exec_id=execution.id,
                graph_id=execution.graph_id,
                graph_version=execution.graph_version,
                execution_stats=stats_for_generation,
                db_client=db_client,
                user_id=execution.user_id,
                execution_status=execution.status,
                model_name=model_name,  # Pass model name parameter
                skip_feature_flag=True,  # Admin endpoint bypasses feature flags
            )

            if not activity_response:
                return ExecutionAnalyticsResult(
                    agent_id=execution.graph_id,
                    version_id=execution.graph_version,
                    user_id=execution.user_id,
                    exec_id=execution.id,
                    summary_text=None,
                    score=None,
                    status="skipped",
                    error_message="Activity generation returned None",
                )

            # Update the execution stats
            # Convert GraphExecutionMeta.Stats to GraphExecutionStats for DB compatibility
            if execution.stats:
                if isinstance(execution.stats, GraphExecutionMeta.Stats):
                    updated_stats = execution.stats.to_db()
                else:
                    # Already GraphExecutionStats
                    updated_stats = execution.stats
            else:
                updated_stats = GraphExecutionStats()

            updated_stats.activity_status = activity_response["activity_status"]
            updated_stats.correctness_score = activity_response["correctness_score"]

            # Save to database with correct stats type
            await update_graph_execution_stats(
                graph_exec_id=execution.id, stats=updated_stats
            )

            return ExecutionAnalyticsResult(
                agent_id=execution.graph_id,
                version_id=execution.graph_version,
                user_id=execution.user_id,
                exec_id=execution.id,
                summary_text=activity_response["activity_status"],
                score=activity_response["correctness_score"],
                status="success",
            )

        except Exception as e:
            logger.exception(f"Error processing execution {execution.id}: {e}")
            return ExecutionAnalyticsResult(
                agent_id=execution.graph_id,
                version_id=execution.graph_version,
                user_id=execution.user_id,
                exec_id=execution.id,
                summary_text=None,
                score=None,
                status="failed",
                error_message=str(e),
            )

    # Process all executions in the batch concurrently
    return await asyncio.gather(
        *[process_single_execution(execution) for execution in executions]
    )
