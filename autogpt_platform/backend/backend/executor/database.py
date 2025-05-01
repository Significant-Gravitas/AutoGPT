import logging

from backend.data import db
from backend.data.credit import UsageTransactionMetadata, get_user_credit_model
from backend.data.execution import (
    create_graph_execution,
    get_graph_execution,
    get_incomplete_node_executions,
    get_latest_node_execution,
    get_node_execution_results,
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
from backend.util.service import AppService, exposed_run_and_wait
from backend.util.settings import Config

config = Config()
_user_credit_model = get_user_credit_model()
logger = logging.getLogger(__name__)


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

    # Executions
    get_graph_execution = exposed_run_and_wait(get_graph_execution)
    create_graph_execution = exposed_run_and_wait(create_graph_execution)
    get_node_execution_results = exposed_run_and_wait(get_node_execution_results)
    get_incomplete_node_executions = exposed_run_and_wait(
        get_incomplete_node_executions
    )
    get_latest_node_execution = exposed_run_and_wait(get_latest_node_execution)
    update_node_execution_status = exposed_run_and_wait(update_node_execution_status)
    update_node_execution_status_batch = exposed_run_and_wait(
        update_node_execution_status_batch
    )
    update_graph_execution_start_time = exposed_run_and_wait(
        update_graph_execution_start_time
    )
    update_graph_execution_stats = exposed_run_and_wait(update_graph_execution_stats)
    update_node_execution_stats = exposed_run_and_wait(update_node_execution_stats)
    upsert_execution_input = exposed_run_and_wait(upsert_execution_input)
    upsert_execution_output = exposed_run_and_wait(upsert_execution_output)

    # Graphs
    get_node = exposed_run_and_wait(get_node)
    get_graph = exposed_run_and_wait(get_graph)
    get_connected_output_nodes = exposed_run_and_wait(get_connected_output_nodes)
    get_graph_metadata = exposed_run_and_wait(get_graph_metadata)

    # Credits
    spend_credits = exposed_run_and_wait(_spend_credits)
    get_credits = exposed_run_and_wait(_get_credits)

    # User + User Metadata + User Integrations
    get_user_metadata = exposed_run_and_wait(get_user_metadata)
    update_user_metadata = exposed_run_and_wait(update_user_metadata)
    get_user_integrations = exposed_run_and_wait(get_user_integrations)
    update_user_integrations = exposed_run_and_wait(update_user_integrations)

    # User Comms - async
    get_active_user_ids_in_timerange = exposed_run_and_wait(
        get_active_user_ids_in_timerange
    )
    get_user_email_by_id = exposed_run_and_wait(get_user_email_by_id)
    get_user_email_verification = exposed_run_and_wait(get_user_email_verification)
    get_user_notification_preference = exposed_run_and_wait(
        get_user_notification_preference
    )

    # Notifications - async
    create_or_add_to_user_notification_batch = exposed_run_and_wait(
        create_or_add_to_user_notification_batch
    )
    empty_user_notification_batch = exposed_run_and_wait(empty_user_notification_batch)
    get_all_batches_by_type = exposed_run_and_wait(get_all_batches_by_type)
    get_user_notification_batch = exposed_run_and_wait(get_user_notification_batch)
    get_user_notification_oldest_message_in_batch = exposed_run_and_wait(
        get_user_notification_oldest_message_in_batch
    )
