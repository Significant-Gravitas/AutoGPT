import asyncio
import logging
from datetime import datetime
from typing import Optional

from autogpt_libs.auth import get_user_id, requires_admin_user
from fastapi import APIRouter, HTTPException, Security
from pydantic import BaseModel, Field

from backend.blocks.llm import LlmModel
from backend.data.analytics import (
    AccuracyTrendsResponse,
    get_accuracy_trends_and_alerts,
)
from backend.data.execution import (
    ExecutionStatus,
    GraphExecutionMeta,
    get_graph_executions,
    update_graph_execution_stats,
)
from backend.data.model import GraphExecutionStats
from backend.executor.activity_status_generator import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    generate_activity_status_for_execution,
)
from backend.executor.manager import get_db_async_client
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()


class ExecutionAnalyticsRequest(BaseModel):
    graph_id: str = Field(..., description="Graph ID to analyze")
    graph_version: Optional[int] = Field(None, description="Optional graph version")
    user_id: Optional[str] = Field(None, description="Optional user ID filter")
    created_after: Optional[datetime] = Field(
        None, description="Optional created date lower bound"
    )
    model_name: str = Field("gpt-4o-mini", description="Model to use for generation")
    batch_size: int = Field(
        10, description="Batch size for concurrent processing", le=25, ge=1
    )
    system_prompt: Optional[str] = Field(
        None, description="Custom system prompt (default: built-in prompt)"
    )
    user_prompt: Optional[str] = Field(
        None,
        description="Custom user prompt with {{GRAPH_NAME}} and {{EXECUTION_DATA}} placeholders (default: built-in prompt)",
    )
    skip_existing: bool = Field(
        True,
        description="Whether to skip executions that already have activity status and correctness score",
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
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None


class ExecutionAnalyticsResponse(BaseModel):
    total_executions: int
    processed_executions: int
    successful_analytics: int
    failed_analytics: int
    skipped_executions: int
    results: list[ExecutionAnalyticsResult]


class ModelInfo(BaseModel):
    value: str
    label: str
    provider: str


class ExecutionAnalyticsConfig(BaseModel):
    available_models: list[ModelInfo]
    default_system_prompt: str
    default_user_prompt: str
    recommended_model: str


class AccuracyTrendsRequest(BaseModel):
    graph_id: str = Field(..., description="Graph ID to analyze", min_length=1)
    user_id: Optional[str] = Field(None, description="Optional user ID filter")
    days_back: int = Field(30, description="Number of days to look back", ge=7, le=90)
    drop_threshold: float = Field(
        10.0, description="Alert threshold percentage", ge=1.0, le=50.0
    )
    include_historical: bool = Field(
        False, description="Include historical data for charts"
    )


router = APIRouter(
    prefix="/admin",
    tags=["admin", "execution_analytics"],
    dependencies=[Security(requires_admin_user)],
)


@router.get(
    "/execution_analytics/config",
    response_model=ExecutionAnalyticsConfig,
    summary="Get Execution Analytics Configuration",
)
async def get_execution_analytics_config(
    admin_user_id: str = Security(get_user_id),
):
    """
    Get the configuration for execution analytics including:
    - Available AI models with metadata
    - Default system and user prompts
    - Recommended model selection
    """
    logger.info(f"Admin user {admin_user_id} requesting execution analytics config")

    # Generate model list from LlmModel enum with provider information
    available_models = []

    # Function to generate friendly display names from model values
    def generate_model_label(model: LlmModel) -> str:
        """Generate a user-friendly label from the model enum value."""
        value = model.value

        # For all models, convert underscores/hyphens to spaces and title case
        # e.g., "gpt-4-turbo" -> "GPT 4 Turbo", "claude-3-haiku-20240307" -> "Claude 3 Haiku"
        parts = value.replace("_", "-").split("-")

        # Handle provider prefixes (e.g., "google/", "x-ai/")
        if "/" in value:
            _, model_name = value.split("/", 1)
            parts = model_name.replace("_", "-").split("-")

        # Capitalize and format parts
        formatted_parts = []
        for part in parts:
            # Skip date-like patterns - check for various date formats:
            # - Long dates like "20240307" (8 digits)
            # - Year components like "2024", "2025" (4 digit years >= 2020)
            # - Month/day components like "04", "16" when they appear to be dates
            if part.isdigit():
                if len(part) >= 8:  # Long date format like "20240307"
                    continue
                elif len(part) == 4 and int(part) >= 2020:  # Year like "2024", "2025"
                    continue
                elif len(part) <= 2 and int(part) <= 31:  # Month/day like "04", "16"
                    # Skip if this looks like a date component (basic heuristic)
                    continue
            # Keep version numbers as-is
            if part.replace(".", "").isdigit():
                formatted_parts.append(part)
            # Capitalize normal words
            else:
                formatted_parts.append(
                    part.upper()
                    if part.upper() in ["GPT", "LLM", "API", "V0"]
                    else part.capitalize()
                )

        model_name = " ".join(formatted_parts)

        # Format provider name for better display
        provider_name = model.provider.replace("_", " ").title()

        # Return with provider prefix for clarity
        return f"{provider_name}: {model_name}"

    # Include all LlmModel values (no more filtering by hardcoded list)
    recommended_model = LlmModel.GPT4O_MINI.value
    for model in LlmModel:
        label = generate_model_label(model)
        # Add "(Recommended)" suffix to the recommended model
        if model.value == recommended_model:
            label += " (Recommended)"

        available_models.append(
            ModelInfo(
                value=model.value,
                label=label,
                provider=model.provider,
            )
        )

    # Sort models by provider and name for better UX
    available_models.sort(key=lambda x: (x.provider, x.label))

    return ExecutionAnalyticsConfig(
        available_models=available_models,
        default_system_prompt=DEFAULT_SYSTEM_PROMPT,
        default_user_prompt=DEFAULT_USER_PROMPT,
        recommended_model=recommended_model,
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
        # Get database client
        db_client = get_db_async_client()

        # Fetch executions to process
        executions = await get_graph_executions(
            graph_id=request.graph_id,
            graph_version=request.graph_version,
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

        # Filter executions that need analytics generation
        executions_to_process = []
        for execution in executions:
            # Skip if we should skip existing analytics and both activity_status and correctness_score exist
            if (
                request.skip_existing
                and execution.stats
                and execution.stats.activity_status
                and execution.stats.correctness_score is not None
            ):
                continue

            # Add execution to processing list
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

                batch_results = await _process_batch(batch, request, db_client)

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
                    started_at=execution.started_at,
                    ended_at=execution.ended_at,
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
    executions, request: ExecutionAnalyticsRequest, db_client
) -> list[ExecutionAnalyticsResult]:
    """Process a batch of executions concurrently."""

    if not settings.secrets.openai_internal_api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

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
                model_name=request.model_name,
                skip_feature_flag=True,  # Admin endpoint bypasses feature flags
                system_prompt=request.system_prompt or DEFAULT_SYSTEM_PROMPT,
                user_prompt=request.user_prompt or DEFAULT_USER_PROMPT,
                skip_existing=request.skip_existing,
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
                    started_at=execution.started_at,
                    ended_at=execution.ended_at,
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
                started_at=execution.started_at,
                ended_at=execution.ended_at,
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
                started_at=execution.started_at,
                ended_at=execution.ended_at,
            )

    # Process all executions in the batch concurrently
    return await asyncio.gather(
        *[process_single_execution(execution) for execution in executions]
    )


@router.get(
    "/execution_accuracy_trends",
    response_model=AccuracyTrendsResponse,
    summary="Get Execution Accuracy Trends and Alerts",
)
async def get_execution_accuracy_trends(
    graph_id: str,
    user_id: Optional[str] = None,
    days_back: int = 30,
    drop_threshold: float = 10.0,
    include_historical: bool = False,
    admin_user_id: str = Security(get_user_id),
) -> AccuracyTrendsResponse:
    """
    Get execution accuracy trends with moving averages and alert detection.
    Simple single-query approach.
    """
    logger.info(
        f"Admin user {admin_user_id} requesting accuracy trends for graph {graph_id}"
    )

    try:
        result = await get_accuracy_trends_and_alerts(
            graph_id=graph_id,
            days_back=days_back,
            user_id=user_id,
            drop_threshold=drop_threshold,
            include_historical=include_historical,
        )

        return result

    except Exception as e:
        logger.exception(f"Error getting accuracy trends for graph {graph_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
