"""
Enhanced DatabaseManager client that uses local caching to reduce blocking operations.

This module provides drop-in replacements for DatabaseManagerClient and DatabaseManagerAsyncClient
that transparently use local caching for non-critical operations while maintaining the same interface.
"""

import logging
from typing import Any, Dict, Optional

from backend.data.block import BlockInput
from backend.data.credit import UsageTransactionMetadata
from backend.data.execution import ExecutionStatus, NodeExecutionResult
from backend.executor.database import DatabaseManagerAsyncClient, DatabaseManagerClient
from backend.executor.local_cache import get_executor_cache, initialize_executor_cache
from backend.util.exceptions import InsufficientBalanceError

logger = logging.getLogger(__name__)


class CachedDatabaseManagerClient(DatabaseManagerClient):
    """
    Enhanced DatabaseManagerClient that uses local caching for non-blocking operations.

    Operations are categorized as:
    1. Critical (blocking): get_node, get_credits, get_graph_execution_meta, etc.
    2. Non-blocking (cached): upsert_execution_output, update_node_execution_status, etc.
    3. Credit operations: spend_credits with local balance tracking
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = get_executor_cache()
        self._original_client = super()

    @classmethod
    def create_with_cache(cls, remote_client: DatabaseManagerClient):
        """Create cached client using existing remote client"""
        instance = cls()
        instance._original_client = remote_client
        instance._cache = initialize_executor_cache(remote_client)
        return instance

    # Critical operations - remain blocking (delegate to original client)
    def get_graph_executions(self, *args, **kwargs):
        return self._original_client.get_graph_executions(*args, **kwargs)

    def get_graph_execution_meta(self, *args, **kwargs):
        return self._original_client.get_graph_execution_meta(*args, **kwargs)

    def get_node_executions(
        self, graph_exec_id: str, statuses=None, node_id=None, block_ids=None
    ):
        """Get node executions - use cache for hot path operations"""
        try:
            if self._cache.should_use_cache_for_node_executions(graph_exec_id):
                cached_results = self._cache.get_cached_node_executions(
                    graph_exec_id=graph_exec_id,
                    statuses=statuses,
                    node_id=node_id,
                    block_ids=block_ids,
                )
                if cached_results:
                    logger.debug(
                        f"Returned {len(cached_results)} cached node executions for {graph_exec_id}"
                    )
                    # Convert to NodeExecutionResult objects
                    from datetime import datetime

                    from backend.data.execution import NodeExecutionResult

                    results = []
                    for cached in cached_results:
                        result = NodeExecutionResult(
                            user_id=cached.get("user_id", ""),
                            graph_id=cached.get("graph_id", ""),
                            graph_version=1,  # Default version
                            graph_exec_id=cached["graph_exec_id"],
                            node_exec_id=cached["node_exec_id"],
                            node_id=cached.get("node_id", ""),
                            block_id=cached.get("block_id", ""),
                            status=cached["status"],
                            input_data=cached["input_data"],
                            output_data=cached["output_data"],
                            add_time=datetime.now(),
                            queue_time=datetime.now(),
                            start_time=datetime.now(),
                            end_time=datetime.now(),
                        )
                        # Add stats if available
                        if cached.get("stats"):
                            result.stats = cached["stats"]
                        results.append(result)
                    return results
        except Exception as e:
            logger.error(f"Failed to get cached node executions: {e}")

        # Fallback to remote call and cache results
        results = self._original_client.get_node_executions(
            graph_exec_id, statuses, node_id, block_ids
        )

        # Cache the results for future use
        try:
            for result in results:
                self._cache.cache_node_execution(result)
        except Exception as e:
            logger.error(f"Failed to cache node execution results: {e}")

        return results

    def get_graph_metadata(self, *args, **kwargs):
        # This is used for notifications - can cache but not critical
        return self._original_client.get_graph_metadata(*args, **kwargs)

    def get_user_email_by_id(self, *args, **kwargs):
        return self._original_client.get_user_email_by_id(*args, **kwargs)

    def get_block_error_stats(self, *args, **kwargs):
        return self._original_client.get_block_error_stats(*args, **kwargs)

    # Non-blocking operations - use local cache
    def upsert_execution_output(
        self, node_exec_id: str, output_name: str, output_data: Any
    ):
        """Store execution output in local cache and queue for remote sync"""
        try:
            self._cache.upsert_execution_output_local(
                node_exec_id, output_name, output_data
            )
            logger.debug(f"Cached execution output for {node_exec_id}:{output_name}")
        except Exception as e:
            logger.error(f"Failed to cache execution output: {e}")
            # Fallback to remote call
            return self._original_client.upsert_execution_output(
                node_exec_id, output_name, output_data
            )

    def update_node_execution_status(
        self,
        exec_id: str,
        status: ExecutionStatus,
        execution_data: Optional[BlockInput] = None,
        stats: Optional[Dict[str, Any]] = None,
    ) -> NodeExecutionResult:
        """Update node execution status in local cache and queue for remote sync"""
        try:
            self._cache.update_node_execution_status_local(
                exec_id, status, execution_data, stats
            )
            logger.debug(f"Cached node execution status update for {exec_id}: {status}")

            # For status updates, we need to return a NodeExecutionResult
            # We'll create a minimal one since the real update happens async
            return NodeExecutionResult(
                node_exec_id=exec_id,
                status=status,
                input_data=execution_data or {},
                output_data={},
                stats=stats,
            )
        except Exception as e:
            logger.error(f"Failed to cache node status update: {e}")
            # Fallback to remote call
            return self._original_client.update_node_execution_status(
                exec_id, status, execution_data, stats
            )

    def update_graph_execution_start_time(self, *args, **kwargs):
        """Graph execution start time updates can be cached"""
        # For now, keep this blocking since it's called only once per execution
        return self._original_client.update_graph_execution_start_time(*args, **kwargs)

    def update_graph_execution_stats(self, *args, **kwargs):
        """Graph execution stats updates can be cached"""
        # For now, keep this blocking since it's called only once per execution
        return self._original_client.update_graph_execution_stats(*args, **kwargs)

    # Credit operations with local balance tracking
    def spend_credits(
        self, user_id: str, cost: int, metadata: UsageTransactionMetadata
    ) -> int:
        """Spend credits using local balance with eventual consistency"""
        try:
            # Extract graph_exec_id from metadata if available
            graph_exec_id = getattr(metadata, "graph_exec_id", None)
            if not graph_exec_id:
                # Fallback to remote if no graph_exec_id
                return self._original_client.spend_credits(user_id, cost, metadata)

            return self._cache.spend_credits_local(
                user_id, graph_exec_id, cost, metadata
            )
        except InsufficientBalanceError:
            # Re-raise balance errors as-is
            raise
        except Exception as e:
            logger.error(f"Failed to spend credits locally: {e}")
            # Fallback to remote call
            return self._original_client.spend_credits(user_id, cost, metadata)

    def get_credits(self, user_id: str) -> int:
        """Get credits - check local balance first, then remote"""
        # For initial balance check, use remote
        # TODO: Could cache this with TTL for better performance
        return self._original_client.get_credits(user_id)

    # Library and Store operations - keep blocking
    def list_library_agents(self, *args, **kwargs):
        return self._original_client.list_library_agents(*args, **kwargs)

    def add_store_agent_to_library(self, *args, **kwargs):
        return self._original_client.add_store_agent_to_library(*args, **kwargs)

    def get_store_agents(self, *args, **kwargs):
        return self._original_client.get_store_agents(*args, **kwargs)

    def get_store_agent_details(self, *args, **kwargs):
        return self._original_client.get_store_agent_details(*args, **kwargs)


class CachedDatabaseManagerAsyncClient(DatabaseManagerAsyncClient):
    """
    Enhanced async DatabaseManagerAsyncClient that uses local caching.

    For async operations, we maintain the same async interface but use local caching
    where appropriate to reduce blocking on remote database calls.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = get_executor_cache()
        self._original_client = super()

    @classmethod
    def create_with_cache(cls, remote_client: DatabaseManagerAsyncClient):
        """Create cached async client using existing remote client"""
        instance = cls()
        instance._original_client = remote_client
        instance._cache = get_executor_cache()  # Should already be initialized
        return instance

    # Critical async operations - remain blocking
    async def get_node(self, node_id: str):
        """Get node definition - use cache for frequent lookups"""
        try:
            cached_node = self._cache.get_cached_node(node_id)
            if cached_node:
                logger.debug(f"Returned cached node {node_id}")
                # Convert cached dict back to Node object
                from backend.data.graph import Link, Node

                # Convert links back to Link objects
                input_links = [Link(**link) for link in cached_node["input_links"]]
                output_links = [Link(**link) for link in cached_node["output_links"]]

                # Create Node object (simplified - may need adjustments based on actual Node class)
                from backend.data.block import get_block

                block = get_block(cached_node["block_id"])
                if block:
                    node = Node(
                        id=cached_node["id"],
                        block_id=cached_node["block_id"],
                        input_default=cached_node["input_default"],
                        input_links=input_links,
                        output_links=output_links,
                    )
                    # Set block property through the constructor or initialization
                    # Note: This may need adjustment based on the actual Node implementation
                    object.__setattr__(node, "block", block)
                    return node
        except Exception as e:
            logger.error(f"Failed to get cached node {node_id}: {e}")

        # Fallback to remote call and cache result
        node = await self._original_client.get_node(node_id)

        # Cache the node for future use
        try:
            if node:
                self._cache.cache_node(node)
        except Exception as e:
            logger.error(f"Failed to cache node {node_id}: {e}")

        return node

    async def get_graph(self, *args, **kwargs):
        return await self._original_client.get_graph(*args, **kwargs)

    async def get_graph_execution_meta(self, *args, **kwargs):
        return await self._original_client.get_graph_execution_meta(*args, **kwargs)

    async def get_node_execution(self, *args, **kwargs):
        return await self._original_client.get_node_execution(*args, **kwargs)

    async def get_node_executions(
        self, graph_exec_id: str, statuses=None, node_id=None, block_ids=None
    ):
        """Get node executions - use cache for hot path operations"""
        try:
            if self._cache.should_use_cache_for_node_executions(graph_exec_id):
                cached_results = self._cache.get_cached_node_executions(
                    graph_exec_id=graph_exec_id,
                    statuses=statuses,
                    node_id=node_id,
                    block_ids=block_ids,
                )
                if cached_results:
                    logger.debug(
                        f"Returned {len(cached_results)} cached async node executions for {graph_exec_id}"
                    )
                    # Convert to NodeExecutionResult objects
                    from datetime import datetime

                    from backend.data.execution import NodeExecutionResult

                    results = []
                    for cached in cached_results:
                        result = NodeExecutionResult(
                            user_id=cached.get("user_id", ""),
                            graph_id=cached.get("graph_id", ""),
                            graph_version=1,  # Default version
                            graph_exec_id=cached["graph_exec_id"],
                            node_exec_id=cached["node_exec_id"],
                            node_id=cached.get("node_id", ""),
                            block_id=cached.get("block_id", ""),
                            status=cached["status"],
                            input_data=cached["input_data"],
                            output_data=cached["output_data"],
                            add_time=datetime.now(),
                            queue_time=datetime.now(),
                            start_time=datetime.now(),
                            end_time=datetime.now(),
                        )
                        # Add stats if available
                        if cached.get("stats"):
                            result.stats = cached["stats"]
                        results.append(result)
                    return results
        except Exception as e:
            logger.error(f"Failed to get cached async node executions: {e}")

        # Fallback to remote call and cache results
        results = await self._original_client.get_node_executions(
            graph_exec_id, statuses, node_id, block_ids
        )

        # Cache the results for future use
        try:
            for result in results:
                self._cache.cache_node_execution(result)
        except Exception as e:
            logger.error(f"Failed to cache async node execution results: {e}")

        return results

    async def get_latest_node_execution(self, *args, **kwargs):
        return await self._original_client.get_latest_node_execution(*args, **kwargs)

    async def upsert_execution_input(self, *args, **kwargs):
        # This is critical for node coordination - keep blocking
        return await self._original_client.upsert_execution_input(*args, **kwargs)

    async def get_user_integrations(self, *args, **kwargs):
        return await self._original_client.get_user_integrations(*args, **kwargs)

    async def update_user_integrations(self, *args, **kwargs):
        return await self._original_client.update_user_integrations(*args, **kwargs)

    async def get_connected_output_nodes(self, *args, **kwargs):
        return await self._original_client.get_connected_output_nodes(*args, **kwargs)

    async def get_graph_metadata(self, *args, **kwargs):
        return await self._original_client.get_graph_metadata(*args, **kwargs)

    # Non-blocking async operations - use local cache
    async def upsert_execution_output(
        self, node_exec_id: str, output_name: str, output_data: Any
    ):
        """Store execution output in local cache and queue for remote sync"""
        try:
            self._cache.upsert_execution_output_local(
                node_exec_id, output_name, output_data
            )
            logger.debug(
                f"Cached async execution output for {node_exec_id}:{output_name}"
            )
        except Exception as e:
            logger.error(f"Failed to cache async execution output: {e}")
            # Fallback to remote call
            await self._original_client.upsert_execution_output(
                node_exec_id, output_name, output_data
            )

    async def update_node_execution_status(
        self,
        exec_id: str,
        status: ExecutionStatus,
        execution_data: Optional[BlockInput] = None,
        stats: Optional[Dict[str, Any]] = None,
    ) -> NodeExecutionResult:
        """Update node execution status in local cache and queue for remote sync"""
        try:
            self._cache.update_node_execution_status_local(
                exec_id, status, execution_data, stats
            )
            logger.debug(
                f"Cached async node execution status update for {exec_id}: {status}"
            )

            # For async status updates, we need to return a NodeExecutionResult
            return NodeExecutionResult(
                node_exec_id=exec_id,
                status=status,
                input_data=execution_data or {},
                output_data={},
                stats=stats,
            )
        except Exception as e:
            logger.error(f"Failed to cache async node status update: {e}")
            # Fallback to remote call
            return await self._original_client.update_node_execution_status(
                exec_id, status, execution_data, stats
            )

    async def update_graph_execution_stats(self, *args, **kwargs):
        # For now, keep this blocking since it's called only once per execution
        return await self._original_client.update_graph_execution_stats(*args, **kwargs)

    # KV data operations
    async def get_execution_kv_data(self, *args, **kwargs):
        return await self._original_client.get_execution_kv_data(*args, **kwargs)

    async def set_execution_kv_data(self, *args, **kwargs):
        return await self._original_client.set_execution_kv_data(*args, **kwargs)

    # User communication operations
    async def get_active_user_ids_in_timerange(self, *args, **kwargs):
        return await self._original_client.get_active_user_ids_in_timerange(
            *args, **kwargs
        )

    async def get_user_email_by_id(self, *args, **kwargs):
        return await self._original_client.get_user_email_by_id(*args, **kwargs)

    async def get_user_email_verification(self, *args, **kwargs):
        return await self._original_client.get_user_email_verification(*args, **kwargs)

    async def get_user_notification_preference(self, *args, **kwargs):
        return await self._original_client.get_user_notification_preference(
            *args, **kwargs
        )

    # Notification operations
    async def create_or_add_to_user_notification_batch(self, *args, **kwargs):
        return await self._original_client.create_or_add_to_user_notification_batch(
            *args, **kwargs
        )

    async def empty_user_notification_batch(self, *args, **kwargs):
        return await self._original_client.empty_user_notification_batch(
            *args, **kwargs
        )

    async def get_all_batches_by_type(self, *args, **kwargs):
        return await self._original_client.get_all_batches_by_type(*args, **kwargs)

    async def get_user_notification_batch(self, *args, **kwargs):
        return await self._original_client.get_user_notification_batch(*args, **kwargs)

    async def get_user_notification_oldest_message_in_batch(self, *args, **kwargs):
        return (
            await self._original_client.get_user_notification_oldest_message_in_batch(
                *args, **kwargs
            )
        )

    # Library operations
    async def list_library_agents(self, *args, **kwargs):
        return await self._original_client.list_library_agents(*args, **kwargs)

    async def add_store_agent_to_library(self, *args, **kwargs):
        return await self._original_client.add_store_agent_to_library(*args, **kwargs)

    # Store operations
    async def get_store_agents(self, *args, **kwargs):
        return await self._original_client.get_store_agents(*args, **kwargs)

    async def get_store_agent_details(self, *args, **kwargs):
        return await self._original_client.get_store_agent_details(*args, **kwargs)

    # Summary data
    async def get_user_execution_summary_data(self, *args, **kwargs):
        return await self._original_client.get_user_execution_summary_data(
            *args, **kwargs
        )


# Convenience functions to create cached clients
def create_cached_db_client(
    original_client: DatabaseManagerClient,
) -> CachedDatabaseManagerClient:
    """Create a cached database client from original client"""
    return CachedDatabaseManagerClient.create_with_cache(original_client)


def create_cached_db_async_client(
    original_client: DatabaseManagerAsyncClient,
) -> CachedDatabaseManagerAsyncClient:
    """Create a cached async database client from original client"""
    return CachedDatabaseManagerAsyncClient.create_with_cache(original_client)
