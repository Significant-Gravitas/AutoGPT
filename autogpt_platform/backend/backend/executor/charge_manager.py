import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from backend.data.block import block_usage_cost, get_block
from backend.data.cost import execution_usage_cost
from backend.executor.execution_data_client import create_execution_data_client
from backend.integrations.credentials_store import UsageTransactionMetadata

logger = logging.getLogger(__name__)


class ChargeManager:
    def __init__(self):
        self._executor: Optional[ThreadPoolExecutor] = None
        self._lock = threading.Lock()

    def get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            with self._lock:
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(
                        max_workers=2, thread_name_prefix="charge-worker"
                    )
        return self._executor

    def charge_async(
        self,
        node_exec,
        execution_count: int,
        execution_stats=None,
        execution_stats_lock=None,
    ):
        executor = self.get_executor()
        executor.submit(
            self._do_charge,
            node_exec,
            execution_count,
            execution_stats,
            execution_stats_lock,
        )

    def _do_charge(
        self,
        node_exec,
        execution_count: int,
        execution_stats=None,
        execution_stats_lock=None,
    ):
        try:
            db_client = create_execution_data_client()
            block = get_block(node_exec.block_id)
            if not block:
                return

            total_cost = 0

            cost, matching_filter = block_usage_cost(
                block=block, input_data=node_exec.inputs
            )
            if cost > 0:
                db_client.spend_credits(
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

            cost, usage_count = execution_usage_cost(execution_count)
            if cost > 0:
                db_client.spend_credits(
                    user_id=node_exec.user_id,
                    cost=cost,
                    metadata=UsageTransactionMetadata(
                        graph_exec_id=node_exec.graph_exec_id,
                        graph_id=node_exec.graph_id,
                        input={
                            "execution_count": usage_count,
                            "charge": "Execution Cost",
                        },
                        reason=f"Execution Cost for {usage_count} blocks",
                    ),
                )
                total_cost += cost

            if execution_stats and execution_stats_lock:
                with execution_stats_lock:
                    execution_stats.cost += total_cost

        except Exception as e:
            logger.error(f"Async charge failed: {e}")

    def shutdown(self):
        if self._executor:
            self._executor.shutdown(wait=True)


_charge_manager: Optional[ChargeManager] = None


def get_charge_manager() -> ChargeManager:
    global _charge_manager
    if _charge_manager is None:
        _charge_manager = ChargeManager()
    return _charge_manager
