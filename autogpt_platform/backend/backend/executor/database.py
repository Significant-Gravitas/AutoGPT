import logging
from typing import Callable, Concatenate, ParamSpec, TypeVar, cast

from backend.data import db
from backend.data.credit import UsageTransactionMetadata, get_user_credit_model
from backend.data.execution import (
    create_graph_execution,
    get_block_error_stats,
    get_execution_kv_data,
    get_graph_execution,
    get_graph_execution_meta,
    get_graph_executions,
    get_latest_node_execution,
    get_node_execution,
    get_node_executions,
    set_execution_kv_data,
    update_graph_execution_start_time,
    update_graph_execution_stats,
    update_node_execution_stats,
    update_node_execution_status,
    update_node_execution_status_batch,
    upsert_execution_input,
    upsert_execution_output,
)
from backend.data.graph import (
    get_connected_output_nodes,
    get_graph,
    get_graph_metadata,
    get_node,
)
from backend.data.notifications import (
    create_or_add_to_user_notification_batch,
    empty_user_notification_batch,
    get_all_batches_by_type,
    get_user_notification_batch,
    get_user_notification_oldest_message_in_batch,
)
from backend.data.user import (
    get_active_user_ids_in_timerange,
    get_user_email_by_id,
    get_user_email_verification,
    get_user_integrations,
    get_user_metadata,
    get_user_notification_preference,
    update_user_integrations,
    update_user_metadata,
)
from backend.util.service import AppService, AppServiceClient, endpoint_to_sync, expose
from backend.util.settings import Config

config = Config()
_user_credit_model = get_user_credit_model()
logger = logging.getLogger(__name__)
P = ParamSpec("P")
R = TypeVar("R")


async def _spend_credits(
    user_id: str, cost: int, metadata: UsageTransactionMetadata
) -> int:
    return await _user_credit_model.spend_credits(user_id, cost, metadata)


async def _get_credits(user_id: str) -> int:
    return await _user_credit_model.get_credits(user_id)


class DatabaseManager(AppService):

    def run_service(self) -> None:
        logger.info(f"[{self.service_name}] ⏳ Connecting to Database...")
        self.run_and_wait(db.connect())
        super().run_service()

    def cleanup(self):
        super().cleanup()
        logger.info(f"[{self.service_name}] ⏳ Disconnecting Database...")
        self.run_and_wait(db.disconnect())

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
    get_graph_execution = _(get_graph_execution)
    get_graph_executions = _(get_graph_executions)
    get_graph_execution_meta = _(get_graph_execution_meta)
    create_graph_execution = _(create_graph_execution)
    get_node_execution = _(get_node_execution)
    get_node_executions = _(get_node_executions)
    get_latest_node_execution = _(get_latest_node_execution)
    update_node_execution_status = _(update_node_execution_status)
    update_node_execution_status_batch = _(update_node_execution_status_batch)
    update_graph_execution_start_time = _(update_graph_execution_start_time)
    update_graph_execution_stats = _(update_graph_execution_stats)
    update_node_execution_stats = _(update_node_execution_stats)
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
    get_user_metadata = _(get_user_metadata)
    update_user_metadata = _(update_user_metadata)
    get_user_integrations = _(get_user_integrations)
    update_user_integrations = _(update_user_integrations)

    # User Comms - async
    get_active_user_ids_in_timerange = _(get_active_user_ids_in_timerange)
    get_user_email_by_id = _(get_user_email_by_id)
    get_user_email_verification = _(get_user_email_verification)
    get_user_notification_preference = _(get_user_notification_preference)

    # Notifications - async
    create_or_add_to_user_notification_batch = _(
        create_or_add_to_user_notification_batch
    )
    empty_user_notification_batch = _(empty_user_notification_batch)
    get_all_batches_by_type = _(get_all_batches_by_type)
    get_user_notification_batch = _(get_user_notification_batch)
    get_user_notification_oldest_message_in_batch = _(
        get_user_notification_oldest_message_in_batch
    )


class DatabaseManagerClient(AppServiceClient):
    d = DatabaseManager
    _ = endpoint_to_sync

    @classmethod
    def get_service_type(cls):
        return DatabaseManager

    # Executions
    get_graph_execution = _(d.get_graph_execution)
    get_graph_executions = _(d.get_graph_executions)
    get_graph_execution_meta = _(d.get_graph_execution_meta)
    create_graph_execution = _(d.create_graph_execution)
    get_node_execution = _(d.get_node_execution)
    get_node_executions = _(d.get_node_executions)
    get_latest_node_execution = _(d.get_latest_node_execution)
    update_node_execution_status = _(d.update_node_execution_status)
    update_node_execution_status_batch = _(d.update_node_execution_status_batch)
    update_graph_execution_start_time = _(d.update_graph_execution_start_time)
    update_graph_execution_stats = _(d.update_graph_execution_stats)
    update_node_execution_stats = _(d.update_node_execution_stats)
    upsert_execution_input = _(d.upsert_execution_input)
    upsert_execution_output = _(d.upsert_execution_output)
    get_execution_kv_data = _(d.get_execution_kv_data)
    set_execution_kv_data = _(d.set_execution_kv_data)

    # Graphs
    get_node = _(d.get_node)
    get_graph = _(d.get_graph)
    get_connected_output_nodes = _(d.get_connected_output_nodes)
    get_graph_metadata = _(d.get_graph_metadata)

    # Credits
    spend_credits = _(d.spend_credits)
    get_credits = _(d.get_credits)

    # User + User Metadata + User Integrations
    get_user_metadata = _(d.get_user_metadata)
    update_user_metadata = _(d.update_user_metadata)
    get_user_integrations = _(d.get_user_integrations)
    update_user_integrations = _(d.update_user_integrations)

    # User Comms - async
    get_active_user_ids_in_timerange = _(d.get_active_user_ids_in_timerange)
    get_user_email_by_id = _(d.get_user_email_by_id)
    get_user_email_verification = _(d.get_user_email_verification)
    get_user_notification_preference = _(d.get_user_notification_preference)

    # Notifications - async
    create_or_add_to_user_notification_batch = _(
        d.create_or_add_to_user_notification_batch
    )
    empty_user_notification_batch = _(d.empty_user_notification_batch)
    get_all_batches_by_type = _(d.get_all_batches_by_type)
    get_user_notification_batch = _(d.get_user_notification_batch)
    get_user_notification_oldest_message_in_batch = _(
        d.get_user_notification_oldest_message_in_batch
    )

    # Block error monitoring
    get_block_error_stats = _(d.get_block_error_stats)


class DatabaseManagerAsyncClient(AppServiceClient):
    d = DatabaseManager

    @classmethod
    def get_service_type(cls):
        return DatabaseManager

    create_graph_execution = d.create_graph_execution
    get_connected_output_nodes = d.get_connected_output_nodes
    get_latest_node_execution = d.get_latest_node_execution
    get_graph = d.get_graph
    get_graph_metadata = d.get_graph_metadata
    get_graph_execution_meta = d.get_graph_execution_meta
    get_node = d.get_node
    get_node_execution = d.get_node_execution
    get_node_executions = d.get_node_executions
    get_user_integrations = d.get_user_integrations
    upsert_execution_input = d.upsert_execution_input
    upsert_execution_output = d.upsert_execution_output
    update_graph_execution_stats = d.update_graph_execution_stats
    update_node_execution_stats = d.update_node_execution_stats
    update_node_execution_status = d.update_node_execution_status
    update_node_execution_status_batch = d.update_node_execution_status_batch
    update_user_integrations = d.update_user_integrations
    get_execution_kv_data = d.get_execution_kv_data
    set_execution_kv_data = d.set_execution_kv_data
    get_block_error_stats = d.get_block_error_stats
