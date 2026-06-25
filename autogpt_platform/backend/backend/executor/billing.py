import asyncio
import logging
from typing import TYPE_CHECKING, Any

from backend.blocks import get_block
from backend.blocks._base import Block
from backend.blocks.io import AgentOutputBlock
from backend.data import redis_client as redis
from backend.data.credit import UsageTransactionMetadata
from backend.data.execution import GraphExecutionEntry, NodeExecutionEntry
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
    get_database_manager_async_client,
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
        # Keys here span multiple graph IDs and therefore multiple cluster
        # slots — a bulk DELETE would raise CROSSSLOT, so delete per key.
        deleted = 0
        for key in keys:
            deleted += await redis_client.delete(key)
        return deleted
    except Exception as e:
        logger.warning(
            f"Failed to clear insufficient funds notification flags for user "
            f"{user_id}: {e}"
        )
        return 0


def _block_has_paid_cost_entry(block: Block, input_data: "dict[str, Any]") -> bool:
    """Whether any BLOCK_COSTS entry matches this input — even if pre-flight is 0.

    Used to guard dynamic-cost blocks (SECOND/ITEMS/COST_USD) whose
    pre-flight cost is 0 but whose post-flight reconciliation will debit
    a real amount. A user with non-positive balance must not be allowed
    to start such a block.
    """
    from backend.data.block_cost_config import BLOCK_COSTS

    from .utils import _is_cost_filter_match

    block_costs = BLOCK_COSTS.get(type(block))
    if not block_costs:
        return False
    return any(_is_cost_filter_match(bc.cost_filter, input_data) for bc in block_costs)


def resolve_block_cost(
    node_exec: NodeExecutionEntry,
) -> tuple["Block | None", int, dict[str, Any]]:
    """Look up the block and compute its pre-flight usage cost for an exec.

    Shared by ``charge_usage`` and ``charge_reconciled_usage`` so the
    ``(get_block, block_usage_cost)`` lookup lives in exactly one place.
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
) -> tuple[int, int, int]:
    """Pre-flight charge for a node execution.

    Returns ``(total_cost, remaining_balance, block_pre_flight_cost)``.
    ``block_pre_flight_cost`` is the block-only portion (excludes the
    every-N-blocks ``execution_usage_cost``) so reconciliation can settle
    its delta against the exact value charged here, not a re-fetched one
    that may have drifted if the estimates JSON was hot-swapped between
    the two calls.
    """
    total_cost = 0
    remaining_balance = 0
    block_pre_flight = 0
    db_client = get_db_client()
    block, cost, matching_filter = resolve_block_cost(node_exec)
    if not block:
        return total_cost, 0, block_pre_flight

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
        block_pre_flight = cost
    elif _block_has_paid_cost_entry(block, node_exec.inputs):
        # Dynamic-cost blocks (SECOND/ITEMS/COST_USD) compute 0 pre-flight
        # because the real charge is settled post-flight against stats. Guard
        # execution here: a user with non-positive balance cannot start a
        # paid block even if the pre-flight estimate is zero, otherwise
        # reconciliation leaks real provider spend as an uncollectable debit.
        remaining_balance = db_client.get_credits(user_id=node_exec.user_id)
        if remaining_balance <= 0:
            raise InsufficientBalanceError(
                user_id=node_exec.user_id,
                message=(
                    f"Insufficient balance to run {block.name}: "
                    "dynamic-cost blocks require a positive balance."
                ),
                balance=remaining_balance,
                amount=0,
            )

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

    return total_cost, remaining_balance, block_pre_flight


async def charge_reconciled_usage(
    node_exec: NodeExecutionEntry,
    stats: NodeExecutionStats,
    pre_flight_charge: int | None = None,
) -> tuple[int, int]:
    """Charge the dynamic portion of a block's cost from its execution stats.

    Computes post-flight cost from the execution stats and settles the delta
    against the pre-flight estimate. Positive delta → charge the user (allowed
    to push the balance negative — see below); negative delta → refund the
    overcharge (happens when a TOKENS block's flat MODEL_COST floor exceeds
    the real token-metered cost). Zero delta is a no-op — common for RUN-only
    blocks and any balanced estimate.

    Negative-balance handling: positive deltas are recorded with
    ``fail_insufficient_credits=False`` so a wallet that can't cover the
    shortfall ends up in debt rather than producing a `billing_leak`. The
    block already ran and the provider was already paid in real $$ — failing
    the spend would just hide the cost. The debt is naturally cleared on the
    next top-up (auto or manual), since top-ups simply add to the balance.

    Called once per node execution AFTER the block has finished running and
    stats (walltime, tokens, provider_cost) are populated. Swallows any
    unexpected exception so reconciliation never poisons the success path.

    ``pre_flight_charge`` lets the caller pin the baseline to the exact
    amount that was actually billed in `charge_usage`, instead of re-reading
    the historical-average JSON. Without this, a hot-swap of
    ``block_preflight_estimates.json`` between charge and reconciliation
    would shift the baseline and skew the delta.
    """
    try:
        db_client = get_database_manager_async_client()
        block = get_block(node_exec.block_id)
        if not block:
            return 0, 0

        if pre_flight_charge is None:
            pre_flight, _ = block_usage_cost(block=block, input_data=node_exec.inputs)
        else:
            pre_flight = pre_flight_charge
        post_flight, matching_filter = block_usage_cost(
            block=block, input_data=node_exec.inputs, stats=stats
        )
        delta = post_flight - pre_flight
        if delta == 0:
            return 0, 0

        # spend_credits with a negative cost posts a USAGE transaction whose
        # amount is positive (i.e. credits back to the wallet). We reuse the
        # USAGE type so the refund is attributable to the same graph
        # execution in credit history.
        #
        # `fail_insufficient_credits=False` is passed for every reconciliation
        # spend. For positive deltas this bypasses the balance guard so the
        # spend always lands — the wallet may go negative; that's the point,
        # we'd rather record real debt than leak the cost. For negative deltas
        # (refunds) the flag is moot: `amount = -cost > 0`, so the SQL guard's
        # `$2 >= 0` short-circuit holds regardless.
        remaining_balance = await db_client.spend_credits(
            user_id=node_exec.user_id,
            cost=delta,
            metadata=UsageTransactionMetadata(
                graph_exec_id=node_exec.graph_exec_id,
                graph_id=node_exec.graph_id,
                node_exec_id=node_exec.node_exec_id,
                node_id=node_exec.node_id,
                block_id=node_exec.block_id,
                block=block.name,
                input={**matching_filter, "reconciled_delta": delta},
                reason=(
                    f"Post-flight reconciliation for {block.name}: "
                    f"actual={post_flight} credits, pre-flight={pre_flight}"
                ),
            ),
            fail_insufficient_credits=False,
        )
        # Refunds can't push the balance below the threshold — skip.
        if delta > 0:
            # handle_low_balance is sync + does a blocking RPC; dispatch to
            # thread so we don't block the event loop. Rare path (threshold
            # crossings only).
            await asyncio.to_thread(
                handle_low_balance,
                get_db_client(),
                node_exec.user_id,
                remaining_balance,
                delta,
            )
        return delta, remaining_balance
    except Exception:
        logger.exception(
            f"charge_reconciled_usage failed unexpectedly for block "
            f"{node_exec.block_id}"
        )
        return 0, 0


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
        total_cost, remaining, _ = charge_usage(node_exec, 0)
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
                billing_page_link=f"{base_url}/settings/billing",
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
                    billing_page_link=f"{base_url}/settings/billing",
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
