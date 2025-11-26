import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Callable, Concatenate, ParamSpec, TypeVar, cast

from backend.data import db
from backend.data.credit import UsageTransactionMetadata, get_user_credit_model
from backend.data.execution import (
    create_graph_execution,
    get_block_error_stats,
    get_child_graph_executions,
    get_execution_kv_data,
    get_graph_execution_meta,
    get_graph_executions,
    get_graph_executions_count,
    get_latest_node_execution,
    get_node_execution,
    get_node_executions,
    set_execution_kv_data,
    update_graph_execution_start_time,
    update_graph_execution_stats,
    update_node_execution_status,
    update_node_execution_status_batch,
    upsert_execution_input,
    upsert_execution_output,
)
from backend.data.generate_data import get_user_execution_summary_data
from backend.data.graph import (
    get_connected_output_nodes,
    get_graph,
    get_graph_metadata,
    get_node,
    validate_graph_execution_permissions,
)
from backend.data.notifications import (
    clear_all_user_notification_batches,
    create_or_add_to_user_notification_batch,
    empty_user_notification_batch,
    get_all_batches_by_type,
    get_user_notification_batch,
    get_user_notification_oldest_message_in_batch,
    remove_notifications_from_batch,
)
from backend.data.user import (
    get_active_user_ids_in_timerange,
    get_user_by_id,
    get_user_email_by_id,
    get_user_email_verification,
    get_user_integrations,
    get_user_notification_preference,
    update_user_integrations,
)
from backend.server.v2.library.db import add_store_agent_to_library, list_library_agents
from backend.server.v2.store.db import get_store_agent_details, get_store_agents
from backend.util.service import (
    AppService,
    AppServiceClient,
    UnhealthyServiceError,
    endpoint_to_sync,
    expose,
)
from backend.util.settings import Config

if TYPE_CHECKING:
    from fastapi import FastAPI

config = Config()
logger = logging.getLogger(__name__)
P = ParamSpec("P")
R = TypeVar("R")


async def _spend_credits(
    user_id: str, cost: int, metadata: UsageTransactionMetadata
) -> int:
    user_credit_model = await get_user_credit_model(user_id)
    return await user_credit_model.spend_credits(user_id, cost, metadata)


async def _get_credits(user_id: str) -> int:
    user_credit_model = await get_user_credit_model(user_id)
    return await user_credit_model.get_credits(user_id)


class DatabaseManager(AppService):
    @asynccontextmanager
    async def lifespan(self, app: "FastAPI"):
        async with super().lifespan(app):
            logger.info(f"[{self.service_name}] ⏳ Connecting to Database...")
            await db.connect()

            logger.info(f"[{self.service_name}] ✅ Ready")
            yield

            logger.info(f"[{self.service_name}] ⏳ Disconnecting Database...")
            await db.disconnect()

    async def health_check(self) -> str:
        if not db.is_connected():
            raise UnhealthyServiceError("Database is not connected")

        try:
            # Test actual database connectivity by executing a simple query
            # This will fail if Prisma query engine is not responding
            result = await db.query_raw_with_schema("SELECT 1 as health_check")
            if not result or result[0].get("health_check") != 1:
                raise UnhealthyServiceError("Database query test failed")
        except Exception as e:
            raise UnhealthyServiceError(f"Database health check failed: {e}")

        return await super().health_check()

    @classmethod
    def get_port(cls) -> int:
        return config.database_api_port

    @staticmethod
    def _(
        f: Callable[P, R], name: str | None = None
    ) -> Callable[Concatenate[object, P], R]:
        if name is not None:
            f.__name__ = name
        return cast(Callable[Concatenate[object, P], R], expose(f))

    # Executions
    get_child_graph_executions = _(get_child_graph_executions)
    get_graph_executions = _(get_graph_executions)
    get_graph_executions_count = _(get_graph_executions_count)
    get_graph_execution_meta = _(get_graph_execution_meta)
    create_graph_execution = _(create_graph_execution)
    get_node_execution = _(get_node_execution)
    get_node_executions = _(get_node_executions)
    get_latest_node_execution = _(get_latest_node_execution)
    update_node_execution_status = _(update_node_execution_status)
    update_node_execution_status_batch = _(update_node_execution_status_batch)
    update_graph_execution_start_time = _(update_graph_execution_start_time)
    update_graph_execution_stats = _(update_graph_execution_stats)
    upsert_execution_input = _(upsert_execution_input)
    upsert_execution_output = _(upsert_execution_output)
    get_execution_kv_data = _(get_execution_kv_data)
    set_execution_kv_data = _(set_execution_kv_data)
    get_block_error_stats = _(get_block_error_stats)

    # Graphs
    get_node = _(get_node)
    get_graph = _(get_graph)
    get_connected_output_nodes = _(get_connected_output_nodes)
    get_graph_metadata = _(get_graph_metadata)

    # Credits
    spend_credits = _(_spend_credits, name="spend_credits")
    get_credits = _(_get_credits, name="get_credits")

    # User + User Metadata + User Integrations
    get_user_integrations = _(get_user_integrations)
    update_user_integrations = _(update_user_integrations)

    # User Comms - async
    get_active_user_ids_in_timerange = _(get_active_user_ids_in_timerange)
    get_user_by_id = _(get_user_by_id)
    get_user_email_by_id = _(get_user_email_by_id)
    get_user_email_verification = _(get_user_email_verification)
    get_user_notification_preference = _(get_user_notification_preference)

    # Notifications - async
    clear_all_user_notification_batches = _(clear_all_user_notification_batches)
    create_or_add_to_user_notification_batch = _(
        create_or_add_to_user_notification_batch
    )
    empty_user_notification_batch = _(empty_user_notification_batch)
    remove_notifications_from_batch = _(remove_notifications_from_batch)
    get_all_batches_by_type = _(get_all_batches_by_type)
    get_user_notification_batch = _(get_user_notification_batch)
    get_user_notification_oldest_message_in_batch = _(
        get_user_notification_oldest_message_in_batch
    )

    # Library
    list_library_agents = _(list_library_agents)
    add_store_agent_to_library = _(add_store_agent_to_library)
    validate_graph_execution_permissions = _(validate_graph_execution_permissions)

    # Store
    get_store_agents = _(get_store_agents)
    get_store_agent_details = _(get_store_agent_details)

    # Summary data - async
    get_user_execution_summary_data = _(get_user_execution_summary_data)


class DatabaseManagerClient(AppServiceClient):
    d = DatabaseManager
    _ = endpoint_to_sync

    @classmethod
    def get_service_type(cls):
        return DatabaseManager

    # Executions
    get_graph_executions = _(d.get_graph_executions)
    get_graph_executions_count = _(d.get_graph_executions_count)
    get_graph_execution_meta = _(d.get_graph_execution_meta)
    get_node_executions = _(d.get_node_executions)
    update_node_execution_status = _(d.update_node_execution_status)
    update_graph_execution_start_time = _(d.update_graph_execution_start_time)
    update_graph_execution_stats = _(d.update_graph_execution_stats)
    upsert_execution_output = _(d.upsert_execution_output)

    # Graphs
    get_graph_metadata = _(d.get_graph_metadata)

    # Credits
    spend_credits = _(d.spend_credits)
    get_credits = _(d.get_credits)

    # Block error monitoring
    get_block_error_stats = _(d.get_block_error_stats)

    # User Emails
    get_user_email_by_id = _(d.get_user_email_by_id)

    # Library
    list_library_agents = _(d.list_library_agents)
    add_store_agent_to_library = _(d.add_store_agent_to_library)
    validate_graph_execution_permissions = _(d.validate_graph_execution_permissions)

    # Store
    get_store_agents = _(d.get_store_agents)
    get_store_agent_details = _(d.get_store_agent_details)


class DatabaseManagerAsyncClient(AppServiceClient):
    d = DatabaseManager

    @classmethod
    def get_service_type(cls):
        return DatabaseManager

    create_graph_execution = d.create_graph_execution
    get_child_graph_executions = d.get_child_graph_executions
    get_connected_output_nodes = d.get_connected_output_nodes
    get_latest_node_execution = d.get_latest_node_execution
    get_graph = d.get_graph
    get_graph_metadata = d.get_graph_metadata
    get_graph_execution_meta = d.get_graph_execution_meta
    get_node = d.get_node
    get_node_execution = d.get_node_execution
    get_node_executions = d.get_node_executions
    get_user_by_id = d.get_user_by_id
    get_user_integrations = d.get_user_integrations
    upsert_execution_input = d.upsert_execution_input
    upsert_execution_output = d.upsert_execution_output
    update_graph_execution_stats = d.update_graph_execution_stats
    update_node_execution_status = d.update_node_execution_status
    update_node_execution_status_batch = d.update_node_execution_status_batch
    update_user_integrations = d.update_user_integrations
    get_execution_kv_data = d.get_execution_kv_data
    set_execution_kv_data = d.set_execution_kv_data

    # User Comms
    get_active_user_ids_in_timerange = d.get_active_user_ids_in_timerange
    get_user_email_by_id = d.get_user_email_by_id
    get_user_email_verification = d.get_user_email_verification
    get_user_notification_preference = d.get_user_notification_preference

    # Notifications
    clear_all_user_notification_batches = d.clear_all_user_notification_batches
    create_or_add_to_user_notification_batch = (
        d.create_or_add_to_user_notification_batch
    )
    empty_user_notification_batch = d.empty_user_notification_batch
    remove_notifications_from_batch = d.remove_notifications_from_batch
    get_all_batches_by_type = d.get_all_batches_by_type
    get_user_notification_batch = d.get_user_notification_batch
    get_user_notification_oldest_message_in_batch = (
        d.get_user_notification_oldest_message_in_batch
    )

    # Library
    list_library_agents = d.list_library_agents
    add_store_agent_to_library = d.add_store_agent_to_library
    validate_graph_execution_permissions = d.validate_graph_execution_permissions

    # Store
    get_store_agents = d.get_store_agents
    get_store_agent_details = d.get_store_agent_details

    # Summary data
    get_user_execution_summary_data = d.get_user_execution_summary_data
