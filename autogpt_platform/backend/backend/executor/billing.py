import asyncio
import logging
from typing import TYPE_CHECKING, Any, cast

from backend.blocks import get_block
from backend.blocks._base import Block
from backend.blocks.io import AgentOutputBlock
from backend.data import redis_client as redis
from backend.data.credit import UsageTransactionMetadata
from backend.data.execution import (
    ExecutionStatus,
    GraphExecutionEntry,
    NodeExecutionEntry,
)
from backend.data.graph import Node
from backend.data.model import GraphExecutionStats, NodeExecutionStats
from backend.data.notifications import (
    AgentRunData,
    LowBalanceData,
    NotificationEventModel,
    NotificationType,
    ZeroBalanceData,
)
from backend.notifications.notifications import queue_notification
from backend.util.clients import (
    get_database_manager_client,
    get_notification_manager_client,
)
from backend.util.exceptions import InsufficientBalanceError
from backend.util.logging import TruncatedLogger
from backend.util.metrics import DiscordChannel
from backend.util.settings import Settings

from .utils import LogMetadata, block_usage_cost, execution_usage_cost

if TYPE_CHECKING:
    from backend.data.db_manager import DatabaseManagerClient

_logger = logging.getLogger(__name__)
logger = TruncatedLogger(_logger, prefix="[Billing]")
settings = Settings()

# Redis key prefix for tracking insufficient funds Discord notifications.
# We only send one notification per user per agent until they top up credits.
INSUFFICIENT_FUNDS_NOTIFIED_PREFIX = "insufficient_funds_discord_notified"
# TTL for the notification flag (30 days) - acts as a fallback cleanup
INSUFFICIENT_FUNDS_NOTIFIED_TTL_SECONDS = 30 * 24 * 60 * 60

# Hard cap on the multiplier passed to charge_extra_runtime_cost to
# protect against a corrupted llm_call_count draining a user's balance.
# Real agent-mode runs are bounded by agent_mode_max_iterations (~50);
# 200 leaves headroom while preventing runaway charges.
_MAX_EXTRA_RUNTIME_COST = 200


def get_db_client() -> "DatabaseManagerClient":
    return get_database_manager_client()


async def clear_insufficient_funds_notifications(user_id: str) -> int:
    """
    Clear all insufficient funds notification flags for a user.

    This should be called when a user tops up their credits, allowing
    Discord notifications to be sent again if they run out of funds.

    Args:
        user_id: The user ID to clear notifications for.

    Returns:
        The number of keys that were deleted.
    """
    try:
        redis_client = await redis.get_redis_async()
        pattern = f"{INSUFFICIENT_FUNDS_NOTIFIED_PREFIX}:{user_id}:*"
        keys = [key async for key in redis_client.scan_iter(match=pattern)]
        if keys:
            return await redis_client.delete(*keys)
        return 0
    except Exception as e:
        logger.warning(
            f"Failed to clear insufficient funds notification flags for user "
            f"{user_id}: {e}"
        )
        return 0


def resolve_block_cost(
    node_exec: NodeExecutionEntry,
) -> tuple["Block | None", int, dict[str, Any]]:
    """Look up the block and compute its base usage cost for an exec.

    Shared by charge_usage and charge_extra_runtime_cost so the
    (get_block, block_usage_cost) lookup lives in exactly one place.
    Returns ``(block, cost, matching_filter)``. ``block`` is ``None`` if
    the block id can't be resolved — callers should treat that as
    "nothing to charge".
    """
    block = get_block(node_exec.block_id)
    if not block:
        logger.error(f"Block {node_exec.block_id} not found.")
        return None, 0, {}
    cost, matching_filter = block_usage_cost(block=block, input_data=node_exec.inputs)
    return block, cost, matching_filter


def charge_usage(
    node_exec: NodeExecutionEntry,
    execution_count: int,
) -> tuple[int, int]:
    total_cost = 0
    remaining_balance = 0
    db_client = get_db_client()
    block, cost, matching_filter = resolve_block_cost(node_exec)
    if not block:
        return total_cost, 0

    if cost > 0:
        remaining_balance = db_client.spend_credits(
            user_id=node_exec.user_id,
            cost=cost,
            metadata=UsageTransactionMetadata(
                graph_exec_id=node_exec.graph_exec_id,
                graph_id=node_exec.graph_id,
                node_exec_id=node_exec.node_exec_id,
                node_id=node_exec.node_id,
                block_id=node_exec.block_id,
                block=block.name,
                input=matching_filter,
                reason=f"Ran block {node_exec.block_id} {block.name}",
            ),
        )
        total_cost += cost

    # execution_count=0 is used by charge_node_usage for nested tool calls
    # which must not be pushed into higher execution-count tiers.
    # execution_usage_cost(0) would trigger a charge because 0 % threshold == 0,
    # so skip it entirely when execution_count is 0.
    cost, usage_count = (
        execution_usage_cost(execution_count) if execution_count > 0 else (0, 0)
    )
    if cost > 0:
        remaining_balance = db_client.spend_credits(
            user_id=node_exec.user_id,
            cost=cost,
            metadata=UsageTransactionMetadata(
                graph_exec_id=node_exec.graph_exec_id,
                graph_id=node_exec.graph_id,
                input={
                    "execution_count": usage_count,
                    "charge": "Execution Cost",
                },
                reason=f"Execution Cost for {usage_count} blocks of ex_id:{node_exec.graph_exec_id} g_id:{node_exec.graph_id}",
            ),
        )
        total_cost += cost

    return total_cost, remaining_balance


def _charge_extra_runtime_cost_sync(
    node_exec: NodeExecutionEntry,
    capped_count: int,
) -> tuple[int, int]:
    """Synchronous implementation — runs in a thread-pool worker.

    Called only from charge_extra_runtime_cost. Do not call directly from
    async code.

    Note: ``resolve_block_cost`` is called again here (rather than reusing
    the result from ``charge_usage`` at the start of execution) because the
    two calls happen in separate thread-pool workers and sharing mutable
    state across workers would require locks. The block config is immutable
    during a run, so the repeated lookup is safe and produces the same cost;
    the only overhead is an extra registry lookup.
    """
    db_client = get_db_client()
    block, cost, matching_filter = resolve_block_cost(node_exec)
    if not block or cost <= 0:
        return 0, 0
    total_extra_cost = cost * capped_count
    remaining_balance = db_client.spend_credits(
        user_id=node_exec.user_id,
        cost=total_extra_cost,
        metadata=UsageTransactionMetadata(
            graph_exec_id=node_exec.graph_exec_id,
            graph_id=node_exec.graph_id,
            node_exec_id=node_exec.node_exec_id,
            node_id=node_exec.node_id,
            block_id=node_exec.block_id,
            block=block.name,
            input={
                **matching_filter,
                "extra_runtime_cost_count": capped_count,
            },
            reason=(
                f"Extra agent-mode iterations for {block.name} "
                f"({capped_count} additional LLM calls)"
            ),
        ),
    )
    return total_extra_cost, remaining_balance


async def charge_extra_runtime_cost(
    node_exec: NodeExecutionEntry,
    extra_count: int,
) -> tuple[int, int]:
    """Charge a block extra runtime cost beyond the initial run.

    Used by agent-mode blocks (e.g. OrchestratorBlock) that make multiple
    LLM calls within a single node execution. The first iteration is already
    charged by charge_usage; this method charges *extra_count* additional
    copies of the block's base cost.

    Returns ``(total_extra_cost, remaining_balance)``. May raise
    ``InsufficientBalanceError`` if the user can't afford the charge.
    """
    if extra_count <= 0:
        return 0, 0
    # Cap to protect against a corrupted llm_call_count.
    capped = min(extra_count, _MAX_EXTRA_RUNTIME_COST)
    if extra_count > _MAX_EXTRA_RUNTIME_COST:
        logger.warning(
            f"extra_count {extra_count} exceeds cap {_MAX_EXTRA_RUNTIME_COST};"
            f" charging {_MAX_EXTRA_RUNTIME_COST} (llm_call_count may be corrupted)"
        )
    return await asyncio.to_thread(_charge_extra_runtime_cost_sync, node_exec, capped)


async def charge_node_usage(node_exec: NodeExecutionEntry) -> tuple[int, int]:
    """Charge a single node execution to the user.

    Public async wrapper around charge_usage for blocks (e.g. the
    OrchestratorBlock) that spawn nested node executions outside the main
    queue and therefore need to charge them explicitly.

    Also handles low-balance notification so callers don't need to touch
    private functions directly.

    Note: this **does not** increment the global execution counter
    (``increment_execution_count``). Nested tool executions are sub-steps
    of a single block run from the user's perspective and should not push
    them into higher per-execution cost tiers.
    """

    def _run():
        total_cost, remaining = charge_usage(node_exec, 0)
        if total_cost > 0:
            handle_low_balance(
                get_db_client(), node_exec.user_id, remaining, total_cost
            )
        return total_cost, remaining

    return await asyncio.to_thread(_run)


async def try_send_insufficient_funds_notif(
    user_id: str,
    graph_id: str,
    error: InsufficientBalanceError,
    log_metadata: LogMetadata,
) -> None:
    """Send an insufficient-funds notification, swallowing failures."""
    try:
        await asyncio.to_thread(
            handle_insufficient_funds_notif,
            get_db_client(),
            user_id,
            graph_id,
            error,
        )
    except Exception as notif_error:  # pragma: no cover
        log_metadata.warning(
            f"Failed to send insufficient funds notification: {notif_error}"
        )


async def handle_post_execution_billing(
    node: Node,
    node_exec: NodeExecutionEntry,
    execution_stats: NodeExecutionStats,
    status: ExecutionStatus,
    log_metadata: LogMetadata,
) -> None:
    """Charge extra runtime cost for blocks that opt into per-LLM-call billing.

    The first LLM call is already covered by charge_usage(); each additional
    call costs another base_cost. Skipped for dry runs and failed runs.

    InsufficientBalanceError here is a post-hoc billing leak: the work is
    already done but the user can no longer pay. The run stays COMPLETED and
    the error is logged with ``billing_leak: True`` for alerting.
    """
    extra_iterations = (
        cast(Block, node.block).extra_runtime_cost(execution_stats)
        if status == ExecutionStatus.COMPLETED
        and not node_exec.execution_context.dry_run
        else 0
    )
    if extra_iterations <= 0:
        return

    try:
        extra_cost, remaining_balance = await charge_extra_runtime_cost(
            node_exec,
            extra_iterations,
        )
        if extra_cost > 0:
            execution_stats.extra_cost += extra_cost
            await asyncio.to_thread(
                handle_low_balance,
                get_db_client(),
                node_exec.user_id,
                remaining_balance,
                extra_cost,
            )
    except InsufficientBalanceError as e:
        log_metadata.error(
            "billing_leak: insufficient balance after "
            f"{node.block.name} completed {extra_iterations} "
            f"extra iterations",
            extra={
                "billing_leak": True,
                "user_id": node_exec.user_id,
                "graph_id": node_exec.graph_id,
                "block_id": node_exec.block_id,
                "extra_runtime_cost_count": extra_iterations,
                "error": str(e),
            },
        )
        # Do NOT set execution_stats.error — the node ran to completion,
        # only the post-hoc charge failed. See class-level billing-leak
        # contract documentation.
        await try_send_insufficient_funds_notif(
            node_exec.user_id,
            node_exec.graph_id,
            e,
            log_metadata,
        )
    except Exception as e:
        log_metadata.error(
            f"billing_leak: failed to charge extra iterations for {node.block.name}",
            extra={
                "billing_leak": True,
                "user_id": node_exec.user_id,
                "graph_id": node_exec.graph_id,
                "block_id": node_exec.block_id,
                "extra_runtime_cost_count": extra_iterations,
                "error_type": type(e).__name__,
                "error": str(e),
            },
            exc_info=True,
        )


def handle_agent_run_notif(
    db_client: "DatabaseManagerClient",
    graph_exec: GraphExecutionEntry,
    exec_stats: GraphExecutionStats,
) -> None:
    metadata = db_client.get_graph_metadata(
        graph_exec.graph_id, graph_exec.graph_version
    )
    outputs = db_client.get_node_executions(
        graph_exec.graph_exec_id,
        block_ids=[AgentOutputBlock().id],
    )

    named_outputs = [
        {
            key: value[0] if key == "name" else value
            for key, value in output.output_data.items()
        }
        for output in outputs
    ]

    queue_notification(
        NotificationEventModel(
            user_id=graph_exec.user_id,
            type=NotificationType.AGENT_RUN,
            data=AgentRunData(
                outputs=named_outputs,
                agent_name=metadata.name if metadata else "Unknown Agent",
                credits_used=exec_stats.cost,
                execution_time=exec_stats.walltime,
                graph_id=graph_exec.graph_id,
                node_count=exec_stats.node_count,
            ),
        )
    )


def handle_insufficient_funds_notif(
    db_client: "DatabaseManagerClient",
    user_id: str,
    graph_id: str,
    e: InsufficientBalanceError,
) -> None:
    # Check if we've already sent a notification for this user+agent combo.
    # We only send one notification per user per agent until they top up credits.
    redis_key = f"{INSUFFICIENT_FUNDS_NOTIFIED_PREFIX}:{user_id}:{graph_id}"
    try:
        redis_client = redis.get_redis()
        # SET NX returns True only if the key was newly set (didn't exist)
        is_new_notification = redis_client.set(
            redis_key,
            "1",
            nx=True,
            ex=INSUFFICIENT_FUNDS_NOTIFIED_TTL_SECONDS,
        )
        if not is_new_notification:
            # Already notified for this user+agent, skip all notifications
            logger.debug(
                f"Skipping duplicate insufficient funds notification for "
                f"user={user_id}, graph={graph_id}"
            )
            return
    except Exception as redis_error:
        # If Redis fails, log and continue to send the notification
        # (better to occasionally duplicate than to never notify)
        logger.warning(
            f"Failed to check/set insufficient funds notification flag in Redis: "
            f"{redis_error}"
        )

    shortfall = abs(e.amount) - e.balance
    metadata = db_client.get_graph_metadata(graph_id)
    base_url = settings.config.frontend_base_url or settings.config.platform_base_url

    # Queue user email notification
    queue_notification(
        NotificationEventModel(
            user_id=user_id,
            type=NotificationType.ZERO_BALANCE,
            data=ZeroBalanceData(
                current_balance=e.balance,
                billing_page_link=f"{base_url}/profile/credits",
                shortfall=shortfall,
                agent_name=metadata.name if metadata else "Unknown Agent",
            ),
        )
    )

    # Send Discord system alert
    try:
        user_email = db_client.get_user_email_by_id(user_id)

        alert_message = (
            f"❌ **Insufficient Funds Alert**\n"
            f"User: {user_email or user_id}\n"
            f"Agent: {metadata.name if metadata else 'Unknown Agent'}\n"
            f"Current balance: ${e.balance / 100:.2f}\n"
            f"Attempted cost: ${abs(e.amount) / 100:.2f}\n"
            f"Shortfall: ${abs(shortfall) / 100:.2f}\n"
            f"[View User Details]({base_url}/admin/spending?search={user_email})"
        )

        get_notification_manager_client().discord_system_alert(
            alert_message, DiscordChannel.PRODUCT
        )
    except Exception as alert_error:
        logger.error(f"Failed to send insufficient funds Discord alert: {alert_error}")


def handle_low_balance(
    db_client: "DatabaseManagerClient",
    user_id: str,
    current_balance: int,
    transaction_cost: int,
) -> None:
    """Check and handle low balance scenarios after a transaction"""
    LOW_BALANCE_THRESHOLD = settings.config.low_balance_threshold

    balance_before = current_balance + transaction_cost

    if (
        current_balance < LOW_BALANCE_THRESHOLD
        and balance_before >= LOW_BALANCE_THRESHOLD
    ):
        base_url = (
            settings.config.frontend_base_url or settings.config.platform_base_url
        )
        queue_notification(
            NotificationEventModel(
                user_id=user_id,
                type=NotificationType.LOW_BALANCE,
                data=LowBalanceData(
                    current_balance=current_balance,
                    billing_page_link=f"{base_url}/profile/credits",
                ),
            )
        )

        try:
            user_email = db_client.get_user_email_by_id(user_id)
            alert_message = (
                f"⚠️ **Low Balance Alert**\n"
                f"User: {user_email or user_id}\n"
                f"Balance dropped below ${LOW_BALANCE_THRESHOLD / 100:.2f}\n"
                f"Current balance: ${current_balance / 100:.2f}\n"
                f"Transaction cost: ${transaction_cost / 100:.2f}\n"
                f"[View User Details]({base_url}/admin/spending?search={user_email})"
            )
            get_notification_manager_client().discord_system_alert(
                alert_message, DiscordChannel.PRODUCT
            )
        except Exception as e:
            logger.warning(f"Failed to send low balance Discord alert: {e}")
