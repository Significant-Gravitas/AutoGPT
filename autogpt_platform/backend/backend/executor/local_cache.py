"""
Local SQLite-based caching layer for graph executor to reduce blocking I/O operations.

This module provides:
1. Local SQLite cache for non-critical database operations
2. Thread-safe operations with local locks instead of Redis
3. Background sync mechanism for eventual consistency
4. Local credit tracking with atomic operations

Architecture:
- SQLite for local state persistence (survives process restarts)
- Background thread for syncing pending operations to remote DB
- Thread locks replace Redis distributed locks for single-process execution
- Local balance tracking with periodic sync to prevent overdraft
"""

import logging
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.data.block import BlockInput
from backend.data.credit import UsageTransactionMetadata
from backend.data.execution import ExecutionStatus, NodeExecutionResult
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()


@dataclass
class PendingOperation:
    """Represents a database operation that needs to be synced to remote DB"""

    operation_type: str
    graph_exec_id: str
    node_exec_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    timestamp: float = 0.0
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class LocalExecutionState:
    """Local state for a graph execution"""

    graph_exec_id: str
    user_id: str
    status: ExecutionStatus
    stats: Optional[Dict[str, Any]] = None
    local_balance: Optional[int] = None
    execution_count: int = 0
    last_sync: float = 0.0


class LocalExecutorCache:
    """
    Local SQLite-based cache for graph executor operations.

    Provides non-blocking operations for:
    - Execution output storage
    - Status updates
    - Statistics tracking
    - Credit balance management
    - Execution counting
    """

    def __init__(self, db_path: Optional[str] = None):
        # Use temporary directory if cache_dir is not configured
        cache_dir = getattr(settings.config, "cache_dir", "/tmp/autogpt_cache")
        self.db_path = db_path or str(Path(cache_dir) / "executor_cache.db")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # Threading components
        self._local_locks: Dict[str, threading.RLock] = {}
        self._locks_lock = threading.RLock()
        self._sync_thread: Optional[threading.Thread] = None
        self._sync_stop_event = threading.Event()
        self._sync_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="cache-sync"
        )

        # State tracking
        self._execution_states: Dict[str, LocalExecutionState] = {}
        self._pending_operations: List[PendingOperation] = []
        self._initialized = False

        # Remote DB client (injected)
        self._remote_db_client = None

    def initialize(self, remote_db_client):
        """Initialize the cache with remote DB client"""
        if self._initialized:
            return

        self._remote_db_client = remote_db_client
        self._setup_database()
        self._load_state()
        self._start_sync_thread()
        self._initialized = True
        logger.info(f"LocalExecutorCache initialized with DB at {self.db_path}")

    def cleanup(self):
        """Cleanup resources"""
        if not self._initialized:
            return

        logger.info("Shutting down LocalExecutorCache...")
        self._sync_stop_event.set()

        if self._sync_thread:
            self._sync_thread.join(timeout=30)

        self._sync_executor.shutdown(wait=True)
        self._flush_all_pending()
        logger.info("LocalExecutorCache cleanup completed")

    def _setup_database(self):
        """Setup SQLite database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS execution_states (
                    graph_exec_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stats TEXT,
                    local_balance INTEGER,
                    execution_count INTEGER DEFAULT 0,
                    last_sync REAL DEFAULT 0.0,
                    created_at REAL DEFAULT (strftime('%s', 'now')),
                    updated_at REAL DEFAULT (strftime('%s', 'now'))
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pending_operations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    operation_type TEXT NOT NULL,
                    graph_exec_id TEXT NOT NULL,
                    node_exec_id TEXT,
                    data TEXT,
                    timestamp REAL DEFAULT (strftime('%s', 'now')),
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS execution_outputs (
                    node_exec_id TEXT NOT NULL,
                    output_name TEXT NOT NULL,
                    output_data TEXT NOT NULL,
                    timestamp REAL DEFAULT (strftime('%s', 'now')),
                    synced BOOLEAN DEFAULT FALSE,
                    PRIMARY KEY (node_exec_id, output_name)
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS node_statuses (
                    node_exec_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    execution_data TEXT,
                    stats TEXT,
                    timestamp REAL DEFAULT (strftime('%s', 'now')),
                    synced BOOLEAN DEFAULT FALSE
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS node_executions (
                    node_exec_id TEXT PRIMARY KEY,
                    node_id TEXT NOT NULL,
                    graph_exec_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    block_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    input_data TEXT,
                    output_data TEXT,
                    stats TEXT,
                    created_at REAL DEFAULT (strftime('%s', 'now')),
                    updated_at REAL DEFAULT (strftime('%s', 'now'))
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id TEXT PRIMARY KEY,
                    graph_id TEXT NOT NULL,
                    block_id TEXT NOT NULL,
                    input_default TEXT,
                    input_links TEXT,
                    output_links TEXT,
                    metadata TEXT,
                    cached_at REAL DEFAULT (strftime('%s', 'now'))
                )
            """
            )

            # Indexes for performance
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pending_ops_timestamp ON pending_operations(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_outputs_synced ON execution_outputs(synced)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_statuses_synced ON node_statuses(synced)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_node_executions_graph ON node_executions(graph_exec_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_node_executions_status ON node_executions(status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_node_executions_node ON node_executions(node_id, graph_exec_id)"
            )

            conn.commit()

    def _load_state(self):
        """Load existing state from SQLite"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Load execution states
            for row in conn.execute("SELECT * FROM execution_states"):
                state = LocalExecutionState(
                    graph_exec_id=row["graph_exec_id"],
                    user_id=row["user_id"],
                    status=ExecutionStatus(row["status"]),
                    stats=eval(row["stats"]) if row["stats"] else None,
                    local_balance=row["local_balance"],
                    execution_count=row["execution_count"],
                    last_sync=row["last_sync"],
                )
                self._execution_states[state.graph_exec_id] = state

            # Load pending operations
            for row in conn.execute(
                "SELECT * FROM pending_operations ORDER BY timestamp"
            ):
                op = PendingOperation(
                    operation_type=row["operation_type"],
                    graph_exec_id=row["graph_exec_id"],
                    node_exec_id=row["node_exec_id"],
                    data=eval(row["data"]) if row["data"] else None,
                    timestamp=row["timestamp"],
                    retry_count=row["retry_count"],
                    max_retries=row["max_retries"],
                )
                self._pending_operations.append(op)

    def get_local_lock(self, key: str) -> threading.RLock:
        """Get or create a local thread lock for the given key"""
        with self._locks_lock:
            if key not in self._local_locks:
                self._local_locks[key] = threading.RLock()
            return self._local_locks[key]

    @asynccontextmanager
    async def local_synchronized(self, key: str):
        """Local thread-based synchronization to replace Redis locks"""
        lock = self.get_local_lock(key)
        acquired = lock.acquire(blocking=True, timeout=60)
        if not acquired:
            raise TimeoutError(f"Could not acquire local lock for key: {key}")
        try:
            yield
        finally:
            lock.release()

    def get_local_balance(self, user_id: str, graph_exec_id: str) -> Optional[int]:
        """Get local balance for user"""
        if graph_exec_id in self._execution_states:
            return self._execution_states[graph_exec_id].local_balance
        return None

    def spend_credits_local(
        self,
        user_id: str,
        graph_exec_id: str,
        cost: int,
        metadata: UsageTransactionMetadata,
    ) -> int:
        """
        Spend credits locally and queue for remote sync.
        Returns remaining balance or raises InsufficientBalanceError.
        """
        with self.get_local_lock(f"balance:{user_id}"):
            state = self._execution_states.get(graph_exec_id)
            if not state:
                # Initialize from remote balance - this is still blocking but rare
                remote_balance = self._remote_db_client.get_credits(user_id)
                state = LocalExecutionState(
                    graph_exec_id=graph_exec_id,
                    user_id=user_id,
                    status=ExecutionStatus.RUNNING,
                    local_balance=remote_balance,
                )
                self._execution_states[graph_exec_id] = state

            if state.local_balance < cost:
                from backend.util.exceptions import InsufficientBalanceError

                raise InsufficientBalanceError(
                    user_id=user_id,
                    balance=state.local_balance,
                    amount=cost,
                    message=f"Insufficient local balance: {state.local_balance} < {cost}",
                )

            # Deduct locally
            state.local_balance -= cost

            # Queue for remote sync
            self._queue_operation(
                PendingOperation(
                    operation_type="spend_credits",
                    graph_exec_id=graph_exec_id,
                    data={
                        "user_id": user_id,
                        "cost": cost,
                        "metadata": asdict(metadata),
                    },
                )
            )

            return state.local_balance

    def increment_execution_count_local(self, user_id: str, graph_exec_id: str) -> int:
        """Local execution counter replacement for Redis"""
        with self.get_local_lock(f"exec_count:{user_id}"):
            if graph_exec_id not in self._execution_states:
                self._execution_states[graph_exec_id] = LocalExecutionState(
                    graph_exec_id=graph_exec_id,
                    user_id=user_id,
                    status=ExecutionStatus.RUNNING,
                )

            state = self._execution_states[graph_exec_id]
            state.execution_count += 1
            return state.execution_count

    def upsert_execution_output_local(
        self, node_exec_id: str, output_name: str, output_data: Any
    ):
        """Store execution output locally and queue for remote sync"""
        import json

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO execution_outputs 
                (node_exec_id, output_name, output_data, synced) 
                VALUES (?, ?, ?, FALSE)
            """,
                (node_exec_id, output_name, json.dumps(output_data)),
            )
            conn.commit()

        # Queue for remote sync
        self._queue_operation(
            PendingOperation(
                operation_type="upsert_execution_output",
                graph_exec_id="",  # Will be resolved during sync
                node_exec_id=node_exec_id,
                data={"output_name": output_name, "output_data": output_data},
            )
        )

    def update_node_execution_status_local(
        self,
        exec_id: str,
        status: ExecutionStatus,
        execution_data: Optional[BlockInput] = None,
        stats: Optional[Dict[str, Any]] = None,
    ):
        """Update node execution status locally and queue for remote sync"""
        import json

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO node_statuses 
                (node_exec_id, status, execution_data, stats, synced) 
                VALUES (?, ?, ?, ?, FALSE)
            """,
                (
                    exec_id,
                    status.value,
                    json.dumps(execution_data) if execution_data else None,
                    json.dumps(stats) if stats else None,
                ),
            )
            conn.commit()

        # Queue for remote sync
        self._queue_operation(
            PendingOperation(
                operation_type="update_node_execution_status",
                graph_exec_id="",  # Will be resolved during sync
                node_exec_id=exec_id,
                data={
                    "status": status.value,
                    "execution_data": execution_data,
                    "stats": stats,
                },
            )
        )

    def cache_node_execution(self, node_exec: "NodeExecutionResult"):
        """Cache a node execution result locally"""
        import json

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO node_executions 
                (node_exec_id, node_id, graph_exec_id, user_id, block_id, status, input_data, output_data, stats)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    node_exec.node_exec_id,
                    getattr(node_exec, "node_id", ""),
                    getattr(node_exec, "graph_exec_id", ""),
                    getattr(node_exec, "user_id", ""),
                    getattr(node_exec, "block_id", ""),
                    node_exec.status.value,
                    json.dumps(node_exec.input_data),
                    json.dumps(node_exec.output_data),
                    json.dumps(node_exec.stats) if node_exec.stats else None,
                ),
            )
            conn.commit()

    def cache_node(self, node: "Node"):
        """Cache a node definition locally"""
        import json

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO nodes 
                (node_id, graph_id, block_id, input_default, input_links, output_links, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    node.id,
                    getattr(node, "graph_id", ""),
                    node.block_id,
                    json.dumps(node.input_default) if node.input_default else None,
                    (
                        json.dumps([link.__dict__ for link in node.input_links])
                        if node.input_links
                        else None
                    ),
                    (
                        json.dumps([link.__dict__ for link in node.output_links])
                        if node.output_links
                        else None
                    ),
                    json.dumps(getattr(node, "metadata", {})),
                ),
            )
            conn.commit()

    def get_cached_node_executions(
        self,
        graph_exec_id: str,
        statuses: Optional[List[ExecutionStatus]] = None,
        node_id: Optional[str] = None,
        block_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get cached node executions from SQLite - used for hot path operations"""
        import json

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            query = "SELECT * FROM node_executions WHERE graph_exec_id = ?"
            params = [graph_exec_id]

            if statuses:
                status_placeholders = ",".join("?" for _ in statuses)
                query += f" AND status IN ({status_placeholders})"
                params.extend([status.value for status in statuses])

            if node_id:
                query += " AND node_id = ?"
                params.append(node_id)

            if block_ids:
                block_placeholders = ",".join("?" for _ in block_ids)
                query += f" AND block_id IN ({block_placeholders})"
                params.extend(block_ids)

            query += " ORDER BY created_at"

            results = []
            for row in conn.execute(query, params):
                result = {
                    "node_exec_id": row["node_exec_id"],
                    "node_id": row["node_id"],
                    "graph_exec_id": row["graph_exec_id"],
                    "user_id": row["user_id"],
                    "block_id": row["block_id"],
                    "status": ExecutionStatus(row["status"]),
                    "input_data": (
                        json.loads(row["input_data"]) if row["input_data"] else {}
                    ),
                    "output_data": (
                        json.loads(row["output_data"]) if row["output_data"] else {}
                    ),
                    "stats": json.loads(row["stats"]) if row["stats"] else None,
                }
                results.append(result)

            return results

    def get_cached_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get cached node definition from SQLite"""
        import json

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            row = conn.execute(
                "SELECT * FROM nodes WHERE node_id = ?", (node_id,)
            ).fetchone()
            if not row:
                return None

            return {
                "id": row["node_id"],
                "graph_id": row["graph_id"],
                "block_id": row["block_id"],
                "input_default": (
                    json.loads(row["input_default"]) if row["input_default"] else {}
                ),
                "input_links": (
                    json.loads(row["input_links"]) if row["input_links"] else []
                ),
                "output_links": (
                    json.loads(row["output_links"]) if row["output_links"] else []
                ),
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            }

    def should_use_cache_for_node_executions(self, graph_exec_id: str) -> bool:
        """Determine if we should use cache for node executions based on freshness"""
        # For hot path operations within an active execution, always use cache
        # This assumes the cache is populated during execution start
        if graph_exec_id in self._execution_states:
            return True

        # Check if we have recent data in cache
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                """
                SELECT COUNT(*) as count, MAX(updated_at) as last_update 
                FROM node_executions 
                WHERE graph_exec_id = ?
            """,
                (graph_exec_id,),
            ).fetchone()

            if result and result[0] > 0:
                # If data exists and is less than 60 seconds old, use cache
                if time.time() - result[1] < 60:
                    return True

        return False

    def populate_cache_for_execution(self, graph_exec_id: str):
        """
        Populate cache with initial node executions for a graph execution.
        This is called at the start of execution to warm the cache for hot path operations.
        """
        try:
            # Fetch existing node executions from remote DB and cache them
            if self._remote_db_client:
                node_executions = self._remote_db_client.get_node_executions(
                    graph_exec_id
                )
                for node_exec in node_executions:
                    self.cache_node_execution(node_exec)
                logger.info(
                    f"Populated cache with {len(node_executions)} node executions for {graph_exec_id}"
                )
        except Exception as e:
            logger.error(f"Failed to populate cache for {graph_exec_id}: {e}")

    def _queue_operation(self, operation: PendingOperation):
        """Add operation to pending queue"""
        operation.timestamp = time.time()
        self._pending_operations.append(operation)

        # Persist to SQLite
        with sqlite3.connect(self.db_path) as conn:
            import json

            conn.execute(
                """
                INSERT INTO pending_operations 
                (operation_type, graph_exec_id, node_exec_id, data, timestamp, retry_count, max_retries)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    operation.operation_type,
                    operation.graph_exec_id,
                    operation.node_exec_id,
                    json.dumps(operation.data) if operation.data else None,
                    operation.timestamp,
                    operation.retry_count,
                    operation.max_retries,
                ),
            )
            conn.commit()

    def _start_sync_thread(self):
        """Start background sync thread"""
        self._sync_thread = threading.Thread(
            target=self._sync_worker, name="executor-cache-sync", daemon=True
        )
        self._sync_thread.start()

    def _sync_worker(self):
        """Background worker to sync pending operations"""
        logger.info("Cache sync worker started")

        while not self._sync_stop_event.is_set():
            try:
                self._process_pending_operations()
                self._sync_stop_event.wait(timeout=5.0)  # Sync every 5 seconds
            except Exception as e:
                logger.error(f"Error in sync worker: {e}")
                self._sync_stop_event.wait(timeout=10.0)  # Back off on error

        logger.info("Cache sync worker stopped")

    def _process_pending_operations(self):
        """Process pending operations batch"""
        if not self._pending_operations or not self._remote_db_client:
            return

        # Process in batches to avoid blocking
        batch_size = 10
        operations_to_process = self._pending_operations[:batch_size]

        for op in operations_to_process:
            try:
                success = self._sync_operation(op)
                if success:
                    self._pending_operations.remove(op)
                    self._remove_from_db(op)
                else:
                    op.retry_count += 1
                    if op.retry_count >= op.max_retries:
                        logger.error(f"Operation failed max retries: {op}")
                        self._pending_operations.remove(op)
                        self._remove_from_db(op)

            except Exception as e:
                logger.error(f"Error syncing operation {op}: {e}")
                op.retry_count += 1

    def _sync_operation(self, op: PendingOperation) -> bool:
        """Sync a single operation to remote DB"""
        try:
            if op.operation_type == "spend_credits":
                self._remote_db_client.spend_credits(
                    user_id=op.data["user_id"],
                    cost=op.data["cost"],
                    metadata=UsageTransactionMetadata(**op.data["metadata"]),
                )
                return True

            elif op.operation_type == "upsert_execution_output":
                self._remote_db_client.upsert_execution_output(
                    node_exec_id=op.node_exec_id,
                    output_name=op.data["output_name"],
                    output_data=op.data["output_data"],
                )
                return True

            elif op.operation_type == "update_node_execution_status":
                self._remote_db_client.update_node_execution_status(
                    exec_id=op.node_exec_id,
                    status=ExecutionStatus(op.data["status"]),
                    execution_data=op.data.get("execution_data"),
                    stats=op.data.get("stats"),
                )
                return True

            else:
                logger.warning(f"Unknown operation type: {op.operation_type}")
                return True  # Remove unknown operations

        except Exception as e:
            logger.error(f"Failed to sync {op.operation_type}: {e}")
            return False

    def _remove_from_db(self, op: PendingOperation):
        """Remove synced operation from SQLite"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                DELETE FROM pending_operations 
                WHERE operation_type = ? AND graph_exec_id = ? AND node_exec_id = ? AND timestamp = ?
            """,
                (
                    op.operation_type,
                    op.graph_exec_id,
                    op.node_exec_id or "",
                    op.timestamp,
                ),
            )
            conn.commit()

    def _flush_all_pending(self):
        """Flush all pending operations on shutdown"""
        logger.info(f"Flushing {len(self._pending_operations)} pending operations...")

        # Try to sync remaining operations
        for op in self._pending_operations[:]:
            try:
                if self._sync_operation(op):
                    self._pending_operations.remove(op)
            except Exception as e:
                logger.error(f"Failed to flush operation {op}: {e}")

        if self._pending_operations:
            logger.warning(
                f"Could not flush {len(self._pending_operations)} operations"
            )


# Global instance
_executor_cache: Optional[LocalExecutorCache] = None


def get_executor_cache() -> LocalExecutorCache:
    """Get global executor cache instance"""
    global _executor_cache
    if _executor_cache is None:
        _executor_cache = LocalExecutorCache()
    return _executor_cache


def initialize_executor_cache(remote_db_client):
    """Initialize the global executor cache"""
    cache = get_executor_cache()
    cache.initialize(remote_db_client)
    return cache


def cleanup_executor_cache():
    """Cleanup the global executor cache"""
    global _executor_cache
    if _executor_cache:
        _executor_cache.cleanup()
        _executor_cache = None
