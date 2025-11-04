"""
Diagnostics data layer for admin operations.
Provides functions to query and manage system diagnostics including executions and agents.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from prisma.enums import AgentExecutionStatus
from prisma.models import AgentGraphExecution
from pydantic import BaseModel

from backend.data.execution import GraphExecutionEntry, UserContext
from backend.data.rabbitmq import SyncRabbitMQ
from backend.executor.utils import (
    GRAPH_EXECUTION_CANCEL_EXCHANGE,
    GRAPH_EXECUTION_QUEUE_NAME,
    CancelExecutionEvent,
    create_execution_queue_config,
)
from backend.util.clients import get_async_execution_queue

logger = logging.getLogger(__name__)


class RunningExecutionDetail(BaseModel):
    """Details about a running execution for admin view"""

    execution_id: str
    graph_id: str
    graph_name: str  # Will default to "Unknown" if not available
    graph_version: int
    user_id: str
    user_email: Optional[str]
    status: str
    created_at: datetime  # When execution was created
    started_at: Optional[datetime]  # When execution started running
    queue_status: Optional[str] = None


class FailedExecutionDetail(BaseModel):
    """Details about a failed execution for admin view"""

    execution_id: str
    graph_id: str
    graph_name: str
    graph_version: int
    user_id: str
    user_email: Optional[str]
    status: str
    created_at: datetime
    started_at: Optional[datetime]
    failed_at: Optional[datetime]
    error_message: Optional[str]


class ExecutionDiagnosticsSummary(BaseModel):
    """Summary of execution diagnostics"""

    # Current execution state
    running_count: int
    queued_db_count: int
    rabbitmq_queue_depth: int
    cancel_queue_depth: int

    # Orphaned execution detection (old DB records not in executor)
    orphaned_running: int  # Running but created >24h ago (likely orphaned)
    orphaned_queued: int  # Queued but created >24h ago (likely orphaned)

    # Failure metrics
    failed_count_1h: int
    failed_count_24h: int
    failure_rate_24h: float  # failures per hour over last 24h

    # Long-running detection (active executions)
    stuck_running_24h: int  # Running for more than 24 hours
    stuck_running_1h: int  # Running for more than 1 hour
    oldest_running_hours: Optional[float]  # Age of oldest running execution

    # Stuck queued detection
    stuck_queued_1h: int  # Queued for more than 1 hour
    queued_never_started: int  # Queued but started_at is null

    # Throughput metrics
    completed_1h: int
    completed_24h: int
    throughput_per_hour: float  # completions per hour over last 24h

    timestamp: str


class AgentDiagnosticsSummary(BaseModel):
    """Summary of agent diagnostics"""

    agents_with_active_executions: int
    timestamp: str


class ScheduleDetail(BaseModel):
    """Details about a schedule for admin view"""

    schedule_id: str
    schedule_name: str
    graph_id: str
    graph_name: str
    graph_version: int
    user_id: str
    user_email: Optional[str]
    cron: str
    timezone: str
    next_run_time: str
    created_at: Optional[datetime] = None  # Not available from APScheduler


class ScheduleHealthMetrics(BaseModel):
    """Summary of schedule health diagnostics"""

    total_schedules: int
    user_schedules: int  # Excludes system monitoring jobs
    system_schedules: int

    # Orphan detection
    orphaned_deleted_graph: int
    orphaned_no_library_access: int
    orphaned_invalid_credentials: int
    orphaned_validation_failed: int
    total_orphaned: int

    # Upcoming
    schedules_next_hour: int
    schedules_next_24h: int

    timestamp: str


class OrphanedScheduleDetail(BaseModel):
    """Details about an orphaned schedule"""

    schedule_id: str
    schedule_name: str
    graph_id: str
    graph_version: int
    user_id: str
    orphan_reason: (
        str  # deleted_graph, no_library_access, invalid_credentials, validation_failed
    )
    error_detail: Optional[str]
    next_run_time: str


async def get_execution_diagnostics() -> ExecutionDiagnosticsSummary:
    """
    Get comprehensive execution diagnostics including database and queue metrics.

    Returns:
        ExecutionDiagnosticsSummary with current execution state
    """
    try:
        now = datetime.now(timezone.utc)
        one_hour_ago = now - timedelta(hours=1)
        twenty_four_hours_ago = now - timedelta(hours=24)

        # Get running executions count
        running_count = await AgentGraphExecution.prisma().count(
            where={"executionStatus": AgentExecutionStatus.RUNNING}
        )

        # Get queued executions from database
        queued_db_count = await AgentGraphExecution.prisma().count(
            where={"executionStatus": AgentExecutionStatus.QUEUED}
        )

        # Get RabbitMQ queue depths (both execution and cancel queues)
        rabbitmq_queue_depth = get_rabbitmq_queue_depth()
        cancel_queue_depth = get_rabbitmq_cancel_queue_depth()

        # Orphaned execution detection (>24h old, likely not in executor)
        orphaned_running = await AgentGraphExecution.prisma().count(
            where={
                "executionStatus": AgentExecutionStatus.RUNNING,
                "createdAt": {"lt": twenty_four_hours_ago},
            }
        )

        orphaned_queued = await AgentGraphExecution.prisma().count(
            where={
                "executionStatus": AgentExecutionStatus.QUEUED,
                "createdAt": {"lt": twenty_four_hours_ago},
            }
        )

        # Failure metrics
        failed_count_1h = await AgentGraphExecution.prisma().count(
            where={
                "executionStatus": AgentExecutionStatus.FAILED,
                "updatedAt": {"gte": one_hour_ago},
            }
        )

        failed_count_24h = await AgentGraphExecution.prisma().count(
            where={
                "executionStatus": AgentExecutionStatus.FAILED,
                "updatedAt": {"gte": twenty_four_hours_ago},
            }
        )

        failure_rate_24h = failed_count_24h / 24.0 if failed_count_24h > 0 else 0.0

        # Long-running detection (created >24h ago, still running)
        stuck_running_24h = await AgentGraphExecution.prisma().count(
            where={
                "executionStatus": AgentExecutionStatus.RUNNING,
                "createdAt": {"lt": twenty_four_hours_ago},
            }
        )

        stuck_running_1h = await AgentGraphExecution.prisma().count(
            where={
                "executionStatus": AgentExecutionStatus.RUNNING,
                "createdAt": {"lt": one_hour_ago},
            }
        )

        # Find oldest running execution
        oldest_running = await AgentGraphExecution.prisma().find_first(
            where={"executionStatus": AgentExecutionStatus.RUNNING},
            order={"createdAt": "asc"},
        )

        oldest_running_hours = None
        if oldest_running:
            age_seconds = (now - oldest_running.createdAt).total_seconds()
            oldest_running_hours = age_seconds / 3600.0

        # Stuck queued detection
        stuck_queued_1h = await AgentGraphExecution.prisma().count(
            where={
                "executionStatus": AgentExecutionStatus.QUEUED,
                "createdAt": {"lt": one_hour_ago},
            }
        )

        queued_never_started = await AgentGraphExecution.prisma().count(
            where={
                "executionStatus": AgentExecutionStatus.QUEUED,
                "startedAt": None,
            }
        )

        # Throughput metrics
        completed_1h = await AgentGraphExecution.prisma().count(
            where={
                "executionStatus": AgentExecutionStatus.COMPLETED,
                "updatedAt": {"gte": one_hour_ago},
            }
        )

        completed_24h = await AgentGraphExecution.prisma().count(
            where={
                "executionStatus": AgentExecutionStatus.COMPLETED,
                "updatedAt": {"gte": twenty_four_hours_ago},
            }
        )

        throughput_per_hour = completed_24h / 24.0 if completed_24h > 0 else 0.0

        return ExecutionDiagnosticsSummary(
            running_count=running_count,
            queued_db_count=queued_db_count,
            rabbitmq_queue_depth=rabbitmq_queue_depth,
            cancel_queue_depth=cancel_queue_depth,
            orphaned_running=orphaned_running,
            orphaned_queued=orphaned_queued,
            failed_count_1h=failed_count_1h,
            failed_count_24h=failed_count_24h,
            failure_rate_24h=failure_rate_24h,
            stuck_running_24h=stuck_running_24h,
            stuck_running_1h=stuck_running_1h,
            oldest_running_hours=oldest_running_hours,
            stuck_queued_1h=stuck_queued_1h,
            queued_never_started=queued_never_started,
            completed_1h=completed_1h,
            completed_24h=completed_24h,
            throughput_per_hour=throughput_per_hour,
            timestamp=now.isoformat(),
        )
    except Exception as e:
        logger.error(f"Error getting execution diagnostics: {e}")
        raise


async def get_agent_diagnostics() -> AgentDiagnosticsSummary:
    """
    Get comprehensive agent diagnostics.

    Returns:
        AgentDiagnosticsSummary with agent metrics
    """
    try:
        # Get distinct agent graph IDs with active executions
        executions = await AgentGraphExecution.prisma().find_many(
            where={
                "executionStatus": {
                    "in": [AgentExecutionStatus.RUNNING, AgentExecutionStatus.QUEUED]  # type: ignore
                }
            },
            distinct=["agentGraphId"],
        )

        return AgentDiagnosticsSummary(
            agents_with_active_executions=len(executions),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        logger.error(f"Error getting agent diagnostics: {e}")
        raise


async def get_schedule_health_metrics() -> ScheduleHealthMetrics:
    """
    Get comprehensive schedule diagnostics via Scheduler service.

    Returns:
        ScheduleHealthMetrics with schedule health info
    """
    try:
        from backend.util.clients import get_scheduler_client

        scheduler = get_scheduler_client()

        # System job IDs (exclude from user schedule counts)
        SYSTEM_JOB_IDS = {
            "cleanup_expired_files",
            "report_late_executions",
            "report_block_error_rates",
            "process_existing_batches",
            "process_weekly_summary",
        }

        # Get all schedules from scheduler service
        all_schedules = await scheduler.get_execution_schedules()

        # Filter user vs system schedules
        user_schedules = [s for s in all_schedules if s.id not in SYSTEM_JOB_IDS]
        system_schedules_count = len(all_schedules) - len(user_schedules)

        # Detect orphaned schedules
        orphans = await _detect_orphaned_schedules(user_schedules)

        # Count schedules by next run time
        now = datetime.now(timezone.utc)
        one_hour_from_now = now + timedelta(hours=1)
        twenty_four_hours_from_now = now + timedelta(hours=24)

        schedules_next_hour = sum(
            1
            for s in user_schedules
            if s.next_run_time
            and datetime.fromisoformat(s.next_run_time.replace("Z", "+00:00"))
            <= one_hour_from_now
        )

        schedules_next_24h = sum(
            1
            for s in user_schedules
            if s.next_run_time
            and datetime.fromisoformat(s.next_run_time.replace("Z", "+00:00"))
            <= twenty_four_hours_from_now
        )

        return ScheduleHealthMetrics(
            total_schedules=len(all_schedules),
            user_schedules=len(user_schedules),
            system_schedules=system_schedules_count,
            orphaned_deleted_graph=len(orphans["deleted_graph"]),
            orphaned_no_library_access=len(orphans["no_library_access"]),
            orphaned_invalid_credentials=len(orphans["invalid_credentials"]),
            orphaned_validation_failed=len(orphans["validation_failed"]),
            total_orphaned=sum(len(v) for v in orphans.values()),
            schedules_next_hour=schedules_next_hour,
            schedules_next_24h=schedules_next_24h,
            timestamp=now.isoformat(),
        )
    except Exception as e:
        logger.error(f"Error getting schedule diagnostics: {e}")
        raise


async def _detect_orphaned_schedules(schedules: list) -> dict:
    """
    Detect orphaned schedules by validating graph, library access, and credentials.

    Args:
        schedules: List of GraphExecutionJobInfo from scheduler service

    Returns:
        Dict categorizing orphans by type
    """
    from prisma.models import AgentGraph, LibraryAgent

    orphans = {
        "deleted_graph": [],
        "no_library_access": [],
        "invalid_credentials": [],
        "validation_failed": [],
    }

    for schedule in schedules:
        try:
            # Check 1: Graph exists
            graph = await AgentGraph.prisma().find_unique(
                where={
                    "graphVersionId": {
                        "id": schedule.graph_id,
                        "version": schedule.graph_version,
                    }
                }
            )

            if not graph:
                orphans["deleted_graph"].append(schedule.id)
                continue

            # Check 2: User has library access (not deleted/archived)
            library_agent = await LibraryAgent.prisma().find_first(
                where={
                    "userId": schedule.user_id,
                    "agentGraphId": schedule.graph_id,
                    "isDeleted": False,
                    "isArchived": False,
                }
            )

            if not library_agent:
                orphans["no_library_access"].append(schedule.id)
                continue

            # Check 3: Credentials exist (if any)
            # Note: Full credential validation would require integration_creds_manager
            # For now, skip credential validation to avoid complexity
            # Orphaned credentials will be caught during execution attempt
            # if schedule.input_credentials:
            #     # TODO: Add credential validation when needed

        except Exception as e:
            logger.error(f"Error validating schedule {schedule.id}: {e}")
            orphans["validation_failed"].append(schedule.id)

    return orphans


def get_rabbitmq_queue_depth() -> int:
    """
    Get the number of messages in the RabbitMQ execution queue.

    Returns:
        Number of messages in queue, or -1 if error
    """
    try:
        # Create a temporary connection to query the queue
        config = create_execution_queue_config()
        rabbitmq = SyncRabbitMQ(config)
        rabbitmq.connect()

        # Use passive queue_declare to get queue info without modifying it
        if rabbitmq._channel:
            method_frame = rabbitmq._channel.queue_declare(
                queue=GRAPH_EXECUTION_QUEUE_NAME, passive=True
            )
        else:
            raise RuntimeError("RabbitMQ channel not initialized")

        message_count = method_frame.method.message_count

        # Clean up connection
        rabbitmq.disconnect()

        return message_count
    except Exception as e:
        logger.error(f"Error getting RabbitMQ queue depth: {e}")
        # Return -1 to indicate an error state rather than failing the entire request
        return -1


async def get_all_schedules_details(
    limit: int = 100, offset: int = 0
) -> List[ScheduleDetail]:
    """
    Get detailed information about all user schedules via Scheduler service.

    Args:
        limit: Maximum number of schedules to return
        offset: Number of schedules to skip

    Returns:
        List of ScheduleDetail objects
    """
    try:
        from prisma.models import AgentGraph

        from backend.util.clients import get_scheduler_client

        scheduler = get_scheduler_client()

        # System job IDs to exclude
        SYSTEM_JOB_IDS = {
            "cleanup_expired_files",
            "report_late_executions",
            "report_block_error_rates",
            "process_existing_batches",
            "process_weekly_summary",
        }

        # Get all schedules from scheduler
        all_schedules = await scheduler.get_execution_schedules()

        # Filter to user schedules only
        user_schedules = [s for s in all_schedules if s.id not in SYSTEM_JOB_IDS]

        # Apply pagination
        paginated_schedules = user_schedules[offset : offset + limit]

        # Enrich with graph and user details
        results = []
        for schedule in paginated_schedules:
            # Get graph name
            graph = await AgentGraph.prisma().find_unique(
                where={
                    "graphVersionId": {
                        "id": schedule.graph_id,
                        "version": schedule.graph_version,
                    }
                },
                include={"User": True},
            )

            graph_name = graph.name if graph and graph.name else "Unknown"
            user_email = graph.User.email if graph and graph.User else None

            results.append(
                ScheduleDetail(
                    schedule_id=schedule.id,
                    schedule_name=schedule.name,
                    graph_id=schedule.graph_id,
                    graph_name=graph_name,
                    graph_version=schedule.graph_version,
                    user_id=schedule.user_id,
                    user_email=user_email,
                    cron=schedule.cron,
                    timezone=schedule.timezone,
                    next_run_time=schedule.next_run_time,
                )
            )

        return results
    except Exception as e:
        logger.error(f"Error getting schedule details: {e}")
        raise


async def get_orphaned_schedules_details() -> List[OrphanedScheduleDetail]:
    """
    Get detailed list of orphaned schedules with orphan reasons.

    Returns:
        List of OrphanedScheduleDetail objects
    """
    try:
        from backend.util.clients import get_scheduler_client

        scheduler = get_scheduler_client()

        # System job IDs to exclude
        SYSTEM_JOB_IDS = {
            "cleanup_expired_files",
            "report_late_executions",
            "report_block_error_rates",
            "process_existing_batches",
            "process_weekly_summary",
        }

        # Get all schedules
        all_schedules = await scheduler.get_execution_schedules()
        user_schedules = [s for s in all_schedules if s.id not in SYSTEM_JOB_IDS]

        # Detect orphans with categorization
        orphan_categories = await _detect_orphaned_schedules(user_schedules)

        # Build detailed orphan list
        results = []
        for orphan_type, schedule_ids in orphan_categories.items():
            for schedule_id in schedule_ids:
                # Find the schedule
                schedule = next(
                    (s for s in user_schedules if s.id == schedule_id), None
                )
                if not schedule:
                    continue

                results.append(
                    OrphanedScheduleDetail(
                        schedule_id=schedule.id,
                        schedule_name=schedule.name,
                        graph_id=schedule.graph_id,
                        graph_version=schedule.graph_version,
                        user_id=schedule.user_id,
                        orphan_reason=orphan_type,
                        error_detail=None,  # Could add more detail in future
                        next_run_time=schedule.next_run_time,
                    )
                )

        return results
    except Exception as e:
        logger.error(f"Error getting orphaned schedule details: {e}")
        raise


async def cleanup_orphaned_schedules_bulk(
    schedule_ids: List[str], admin_user_id: str
) -> int:
    """
    Cleanup multiple orphaned schedules by deleting from scheduler.

    Args:
        schedule_ids: List of schedule IDs to delete
        admin_user_id: ID of the admin user performing the operation

    Returns:
        Number of schedules successfully deleted
    """
    import asyncio

    try:
        from backend.util.clients import get_scheduler_client

        logger.info(
            f"Admin user {admin_user_id} cleaning up {len(schedule_ids)} orphaned schedules"
        )

        scheduler = get_scheduler_client()

        # Delete schedules in parallel
        async def delete_schedule(schedule_id: str) -> bool:
            try:
                # Note: delete_schedule requires user_id but we're admin
                # We'll need to get the user_id from the schedule first
                all_schedules = await scheduler.get_execution_schedules()
                schedule = next((s for s in all_schedules if s.id == schedule_id), None)

                if not schedule:
                    logger.warning(f"Schedule {schedule_id} not found")
                    return False

                await scheduler.delete_schedule(
                    schedule_id=schedule_id, user_id=schedule.user_id
                )
                return True
            except Exception as e:
                logger.error(f"Failed to delete schedule {schedule_id}: {e}")
                return False

        results = await asyncio.gather(
            *[delete_schedule(schedule_id) for schedule_id in schedule_ids],
            return_exceptions=False,
        )

        deleted_count = sum(1 for success in results if success)

        logger.info(
            f"Admin {admin_user_id} deleted {deleted_count}/{len(schedule_ids)} orphaned schedules"
        )

        return deleted_count
    except Exception as e:
        logger.error(f"Error cleaning up orphaned schedules: {e}")
        return 0


def get_rabbitmq_cancel_queue_depth() -> int:
    """
    Get the number of messages in the RabbitMQ cancel queue.

    Returns:
        Number of messages in cancel queue, or -1 if error
    """
    try:
        from backend.executor.utils import GRAPH_EXECUTION_CANCEL_QUEUE_NAME

        # Create a temporary connection to query the queue
        config = create_execution_queue_config()
        rabbitmq = SyncRabbitMQ(config)
        rabbitmq.connect()

        # Use passive queue_declare to get queue info without modifying it
        if rabbitmq._channel:
            method_frame = rabbitmq._channel.queue_declare(
                queue=GRAPH_EXECUTION_CANCEL_QUEUE_NAME, passive=True
            )
        else:
            raise RuntimeError("RabbitMQ channel not initialized")

        message_count = method_frame.method.message_count

        # Clean up connection
        rabbitmq.disconnect()

        return message_count
    except Exception as e:
        logger.error(f"Error getting RabbitMQ cancel queue depth: {e}")
        # Return -1 to indicate an error state rather than failing the entire request
        return -1


async def get_running_executions_details(
    limit: int = 100, offset: int = 0
) -> List[RunningExecutionDetail]:
    """
    Get detailed information about running and queued executions.

    Args:
        limit: Maximum number of executions to return
        offset: Number of executions to skip

    Returns:
        List of RunningExecutionDetail objects
    """
    try:
        executions = await AgentGraphExecution.prisma().find_many(
            where={
                "executionStatus": {
                    "in": [AgentExecutionStatus.RUNNING, AgentExecutionStatus.QUEUED]  # type: ignore
                }
            },
            include={
                "AgentGraph": True,
                "User": True,
            },
            take=limit,
            skip=offset,
            order={"createdAt": "desc"},
        )

        results = []
        for exec in executions:
            results.append(
                RunningExecutionDetail(
                    execution_id=exec.id,
                    graph_id=exec.agentGraphId,
                    graph_name=(
                        exec.AgentGraph.name
                        if exec.AgentGraph and exec.AgentGraph.name
                        else "Unknown"
                    ),
                    graph_version=exec.agentGraphVersion,
                    user_id=exec.userId,
                    user_email=exec.User.email if exec.User else None,
                    status=exec.executionStatus,
                    created_at=exec.createdAt,
                    started_at=exec.startedAt,
                    queue_status=None,  # Queue status not available from AgentGraphExecution model
                )
            )

        return results
    except Exception as e:
        logger.error(f"Error getting running execution details: {e}")
        raise


async def get_orphaned_executions_details(
    limit: int = 100, offset: int = 0
) -> List[RunningExecutionDetail]:
    """
    Get detailed information about orphaned executions (>24h old, likely not in executor).

    Args:
        limit: Maximum number of executions to return
        offset: Number of executions to skip

    Returns:
        List of orphaned RunningExecutionDetail objects
    """
    try:
        # Executions older than 24 hours are likely orphaned
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

        executions = await AgentGraphExecution.prisma().find_many(
            where={
                "executionStatus": {
                    "in": [AgentExecutionStatus.RUNNING, AgentExecutionStatus.QUEUED]  # type: ignore
                },
                "createdAt": {"lt": cutoff},
            },
            include={
                "AgentGraph": True,
                "User": True,
            },
            take=limit,
            skip=offset,
            order={"createdAt": "asc"},  # Oldest first for orphaned
        )

        results = []
        for exec in executions:
            results.append(
                RunningExecutionDetail(
                    execution_id=exec.id,
                    graph_id=exec.agentGraphId,
                    graph_name=(
                        exec.AgentGraph.name
                        if exec.AgentGraph and exec.AgentGraph.name
                        else "Unknown"
                    ),
                    graph_version=exec.agentGraphVersion,
                    user_id=exec.userId,
                    user_email=exec.User.email if exec.User else None,
                    status=exec.executionStatus,
                    created_at=exec.createdAt,
                    started_at=exec.startedAt,
                    queue_status=None,
                )
            )

        return results
    except Exception as e:
        logger.error(f"Error getting orphaned execution details: {e}")
        raise


async def get_long_running_executions_details(
    limit: int = 100, offset: int = 0
) -> List[RunningExecutionDetail]:
    """
    Get detailed information about long-running executions (RUNNING status >24h).

    Args:
        limit: Maximum number of executions to return
        offset: Number of executions to skip

    Returns:
        List of long-running RunningExecutionDetail objects
    """
    try:
        # RUNNING executions older than 24 hours
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

        executions = await AgentGraphExecution.prisma().find_many(
            where={
                "executionStatus": AgentExecutionStatus.RUNNING,
                "createdAt": {"lt": cutoff},
            },
            include={
                "AgentGraph": True,
                "User": True,
            },
            take=limit,
            skip=offset,
            order={"createdAt": "asc"},  # Oldest first
        )

        results = []
        for exec in executions:
            results.append(
                RunningExecutionDetail(
                    execution_id=exec.id,
                    graph_id=exec.agentGraphId,
                    graph_name=(
                        exec.AgentGraph.name
                        if exec.AgentGraph and exec.AgentGraph.name
                        else "Unknown"
                    ),
                    graph_version=exec.agentGraphVersion,
                    user_id=exec.userId,
                    user_email=exec.User.email if exec.User else None,
                    status=exec.executionStatus,
                    created_at=exec.createdAt,
                    started_at=exec.startedAt,
                    queue_status=None,
                )
            )

        return results
    except Exception as e:
        logger.error(f"Error getting long-running execution details: {e}")
        raise


async def get_stuck_queued_executions_details(
    limit: int = 100, offset: int = 0
) -> List[RunningExecutionDetail]:
    """
    Get detailed information about stuck queued executions (QUEUED >1h, never started).

    Args:
        limit: Maximum number of executions to return
        offset: Number of executions to skip

    Returns:
        List of stuck queued RunningExecutionDetail objects
    """
    try:
        # QUEUED executions older than 1 hour that never started
        one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)

        executions = await AgentGraphExecution.prisma().find_many(
            where={
                "executionStatus": AgentExecutionStatus.QUEUED,
                "createdAt": {"lt": one_hour_ago},
            },
            include={
                "AgentGraph": True,
                "User": True,
            },
            take=limit,
            skip=offset,
            order={"createdAt": "asc"},  # Oldest first
        )

        results = []
        for exec in executions:
            results.append(
                RunningExecutionDetail(
                    execution_id=exec.id,
                    graph_id=exec.agentGraphId,
                    graph_name=(
                        exec.AgentGraph.name
                        if exec.AgentGraph and exec.AgentGraph.name
                        else "Unknown"
                    ),
                    graph_version=exec.agentGraphVersion,
                    user_id=exec.userId,
                    user_email=exec.User.email if exec.User else None,
                    status=exec.executionStatus,
                    created_at=exec.createdAt,
                    started_at=exec.startedAt,
                    queue_status=None,
                )
            )

        return results
    except Exception as e:
        logger.error(f"Error getting stuck queued execution details: {e}")
        raise


async def get_failed_executions_details(
    limit: int = 100, offset: int = 0, hours: int = 24
) -> List[FailedExecutionDetail]:
    """
    Get detailed information about failed executions.

    Args:
        limit: Maximum number of executions to return
        offset: Number of executions to skip
        hours: Number of hours to look back (default 24)

    Returns:
        List of FailedExecutionDetail objects
    """
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        executions = await AgentGraphExecution.prisma().find_many(
            where={
                "executionStatus": AgentExecutionStatus.FAILED,
                "updatedAt": {"gte": cutoff},
            },
            include={
                "AgentGraph": True,
                "User": True,
            },
            take=limit,
            skip=offset,
            order={"updatedAt": "desc"},  # Most recent failures first
        )

        results = []
        for exec in executions:
            # Extract error from stats JSON field
            error_message = None
            if exec.stats and isinstance(exec.stats, dict):
                error_message = exec.stats.get("error")

            results.append(
                FailedExecutionDetail(
                    execution_id=exec.id,
                    graph_id=exec.agentGraphId,
                    graph_name=(
                        exec.AgentGraph.name
                        if exec.AgentGraph and exec.AgentGraph.name
                        else "Unknown"
                    ),
                    graph_version=exec.agentGraphVersion,
                    user_id=exec.userId,
                    user_email=exec.User.email if exec.User else None,
                    status=exec.executionStatus,
                    created_at=exec.createdAt,
                    started_at=exec.startedAt,
                    failed_at=exec.updatedAt,
                    error_message=error_message,
                )
            )

        return results
    except Exception as e:
        logger.error(f"Error getting failed execution details: {e}")
        raise


async def stop_execution(execution_id: str, admin_user_id: str) -> bool:
    """
    Stop a single active execution by sending cancel signal.
    Admin-only operation for executions likely active in executor.

    Args:
        execution_id: ID of the execution to stop
        admin_user_id: ID of the admin user performing the operation

    Returns:
        True if cancel signal was sent successfully, False otherwise
    """
    try:
        logger.info(f"Admin user {admin_user_id} stopping execution {execution_id}")

        # Verify execution exists (without user filtering for admin)
        execution = await AgentGraphExecution.prisma().find_unique(
            where={"id": execution_id}
        )

        if not execution:
            logger.error(f"Execution {execution_id} not found")
            return False

        # Send cancel signal directly to RabbitMQ
        # The executor will handle the cancellation and update status
        queue_client = await get_async_execution_queue()
        await queue_client.publish_message(
            routing_key="",
            message=CancelExecutionEvent(graph_exec_id=execution_id).model_dump_json(),
            exchange=GRAPH_EXECUTION_CANCEL_EXCHANGE,
        )

        logger.info(
            f"Admin {admin_user_id} sent cancel signal for execution {execution_id}"
        )
        return True
    except Exception as e:
        logger.error(f"Error stopping execution {execution_id}: {e}")
        return False


async def cleanup_orphaned_execution(execution_id: str, admin_user_id: str) -> bool:
    """
    Cleanup orphaned execution by directly updating DB status.
    For executions that are in DB but not actually running in executor.

    Args:
        execution_id: ID of the execution to cleanup
        admin_user_id: ID of the admin user performing the operation

    Returns:
        True if execution was cleaned up, False otherwise
    """
    try:
        logger.info(
            f"Admin user {admin_user_id} cleaning up orphaned execution {execution_id}"
        )

        # Update DB status directly without sending cancel signal
        result = await AgentGraphExecution.prisma().update(
            where={"id": execution_id},
            data={
                "executionStatus": AgentExecutionStatus.FAILED,
                "updatedAt": datetime.now(timezone.utc),
            },
        )

        logger.info(
            f"Admin {admin_user_id} marked orphaned execution {execution_id} as FAILED"
        )
        return result is not None
    except Exception as e:
        logger.error(f"Error cleaning up orphaned execution {execution_id}: {e}")
        return False


async def stop_executions_bulk(execution_ids: List[str], admin_user_id: str) -> int:
    """
    Stop multiple active executions by sending cancel signals.
    For executions likely active in executor (created in last 24h).

    Args:
        execution_ids: List of execution IDs to stop
        admin_user_id: ID of the admin user performing the operation

    Returns:
        Number of executions for which cancel signals were sent successfully
    """
    import asyncio

    try:
        logger.info(
            f"Admin user {admin_user_id} stopping {len(execution_ids)} active executions"
        )

        # Verify executions exist (without user filtering for admin)
        executions = await AgentGraphExecution.prisma().find_many(
            where={"id": {"in": execution_ids}}
        )

        if not executions:
            logger.warning("No executions found to stop")
            return 0

        queue_client = await get_async_execution_queue()

        # Send cancel signals in parallel
        async def send_cancel_signal(exec_id: str) -> bool:
            try:
                await queue_client.publish_message(
                    routing_key="",
                    message=CancelExecutionEvent(
                        graph_exec_id=exec_id
                    ).model_dump_json(),
                    exchange=GRAPH_EXECUTION_CANCEL_EXCHANGE,
                )
                return True
            except Exception as e:
                logger.error(f"Failed to send cancel for {exec_id}: {e}")
                return False

        results = await asyncio.gather(
            *[send_cancel_signal(exec.id) for exec in executions],
            return_exceptions=False,
        )

        stopped_count = sum(1 for success in results if success)

        logger.info(
            f"Admin {admin_user_id} sent cancel signals for {stopped_count}/{len(execution_ids)} executions"
        )

        return stopped_count
    except Exception as e:
        logger.error(f"Error stopping executions in bulk: {e}")
        return 0


async def cleanup_orphaned_executions_bulk(
    execution_ids: List[str], admin_user_id: str
) -> int:
    """
    Cleanup multiple orphaned executions by directly updating DB status.
    For executions in DB but not actually running in executor (old/orphaned).

    Args:
        execution_ids: List of execution IDs to cleanup
        admin_user_id: ID of the admin user performing the operation

    Returns:
        Number of executions successfully cleaned up
    """
    try:
        logger.info(
            f"Admin user {admin_user_id} cleaning up {len(execution_ids)} orphaned executions"
        )

        # Update all executions in DB directly (no cancel signals)
        result = await AgentGraphExecution.prisma().update_many(
            where={"id": {"in": execution_ids}},
            data={
                "executionStatus": AgentExecutionStatus.FAILED,
                "updatedAt": datetime.now(timezone.utc),
            },
        )

        logger.info(
            f"Admin {admin_user_id} marked {result} orphaned executions as FAILED in DB"
        )

        return result
    except Exception as e:
        logger.error(f"Error cleaning up orphaned executions in bulk: {e}")
        return 0


async def requeue_execution(execution_id: str, admin_user_id: str) -> bool:
    """
    Requeue a stuck QUEUED execution by publishing to RabbitMQ.
    Admin-only operation for executions that are stuck in QUEUED status.

    Args:
        execution_id: ID of the execution to requeue
        admin_user_id: ID of the admin user performing the operation

    Returns:
        True if successfully requeued, False otherwise
    """
    try:
        logger.info(f"Admin user {admin_user_id} requeueing execution {execution_id}")

        # Verify execution exists and is QUEUED
        execution = await AgentGraphExecution.prisma().find_unique(
            where={"id": execution_id}
        )

        if not execution:
            logger.error(f"Execution {execution_id} not found")
            return False

        if execution.executionStatus != AgentExecutionStatus.QUEUED:
            logger.error(
                f"Execution {execution_id} is not QUEUED (status: {execution.executionStatus})"
            )
            return False

        # Publish to execution queue
        queue_client = await get_async_execution_queue()

        # Get user timezone (default to UTC if not found)
        user_timezone = "UTC"  # Default timezone for admin requeue

        # Create execution entry message
        entry = GraphExecutionEntry(
            graph_id=execution.agentGraphId,
            graph_version=execution.agentGraphVersion,
            graph_exec_id=execution_id,
            user_id=execution.userId,
            user_context=UserContext(timezone=user_timezone),
        )

        await queue_client.publish_message(
            routing_key="",
            message=entry.model_dump_json(),
            exchange=None,
        )

        logger.info(
            f"Admin {admin_user_id} requeued execution {execution_id} to RabbitMQ"
        )
        return True
    except Exception as e:
        logger.error(f"Error requeueing execution {execution_id}: {e}")
        return False


async def requeue_executions_bulk(execution_ids: List[str], admin_user_id: str) -> int:
    """
    Requeue multiple stuck QUEUED executions by publishing to RabbitMQ.
    Admin-only operation.

    Args:
        execution_ids: List of execution IDs to requeue
        admin_user_id: ID of the admin user performing the operation

    Returns:
        Number of executions successfully requeued
    """
    import asyncio

    try:
        logger.info(
            f"Admin user {admin_user_id} requeueing {len(execution_ids)} executions"
        )

        # Verify executions exist and are QUEUED
        executions = await AgentGraphExecution.prisma().find_many(
            where={
                "id": {"in": execution_ids},
                "executionStatus": AgentExecutionStatus.QUEUED,
            }
        )

        if not executions:
            logger.warning("No QUEUED executions found to requeue")
            return 0

        queue_client = await get_async_execution_queue()

        # Publish all to queue in parallel
        async def publish_to_queue(exec: AgentGraphExecution) -> bool:
            try:
                # Use UTC timezone for admin requeue (keeping it simple)
                user_timezone = "UTC"

                entry = GraphExecutionEntry(
                    graph_id=exec.agentGraphId,
                    graph_version=exec.agentGraphVersion,
                    graph_exec_id=exec.id,
                    user_id=exec.userId,
                    user_context=UserContext(timezone=user_timezone),
                )

                await queue_client.publish_message(
                    routing_key="",
                    message=entry.model_dump_json(),
                    exchange=None,
                )
                return True
            except Exception as e:
                logger.error(f"Failed to requeue {exec.id}: {e}")
                return False

        results = await asyncio.gather(
            *[publish_to_queue(exec) for exec in executions],
            return_exceptions=False,
        )

        requeued_count = sum(1 for success in results if success)

        logger.info(
            f"Admin {admin_user_id} requeued {requeued_count}/{len(execution_ids)} executions"
        )

        return requeued_count
    except Exception as e:
        logger.error(f"Error requeueing executions in bulk: {e}")
        return 0
