import logging
import sys
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from types import ModuleType
from typing import TYPE_CHECKING, Any, Concatenate, ParamSpec, TypeVar, cast

from prisma.enums import ContentType

from backend.api.features.experts import credentials as experts_credentials
from backend.api.features.experts import experts_db
from backend.api.features.experts import kickoff as experts_kickoff
from backend.api.features.experts import routine_jobs as experts_routine_jobs
from backend.api.features.experts import routines as experts_routines
from backend.api.features.experts import scheduling as experts_scheduling
from backend.api.features.experts import spend_approval as experts_spend_approval
from backend.api.features.library import db as library_db
from backend.api.features.library import triggers as library_triggers
from backend.api.features.onboarding_dump import db as onboarding_dump_db
from backend.api.features.orgs import db as orgs_db
from backend.api.features.search import embeddings as search_embeddings
from backend.api.features.search import hybrid_search as search_hybrid_search
from backend.api.features.store import db as store_db
from backend.api.features.store import embeddings as store_embeddings
from backend.api.features.store import skill_db as store_skill_db
from backend.blocks import get_blocks
from backend.copilot import background_task_db as copilot_background_task_db
from backend.copilot import db as copilot_db
from backend.copilot.scheduled_chats import forget as copilot_scheduled_chats_forget
from backend.copilot.scheduled_chats import service as copilot_scheduled_chats_service
from backend.copilot.sharing import db as copilot_sharing_db
from backend.data import (
    account_deletion,
    activity_event,
    alerts,
    analytics,
    auth_ip_events,
    block,
    bot_analytics,
    bot_installs,
    briefing,
    briefing_data,
    credit,
    db,
    dream_pass,
    execution,
    feedback_prompts,
)
from backend.data import graph as graph_db
from backend.data import (
    human_review,
    memory_schedule,
    onboarding,
    org_credit,
    platform_cost,
    posthog_lifecycle_sync,
    pro_activation,
    product_mail_audience,
    push_subscription,
    skill_deletion,
    skill_learning,
    skill_publication,
    skill_reviews,
    skill_revocations,
    skill_use,
    skill_versions,
    stripe_reconciliation,
    subscription_trial,
    understanding,
    user,
    workspace,
    workspace_folder,
    workspace_scope,
    workspace_skill,
)
from backend.data.account_erasure import request as account_erasure_request
from backend.data.account_erasure import sweep as account_erasure_sweep
from backend.data.auth import oauth as auth_oauth
from backend.data.auth import session as auth_session
from backend.data.db_endpoints import db_endpoint, declared_endpoints
from backend.notifications import trial as notifications_trial
from backend.platform_linking import db as platform_linking_db
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

# Each module declares its own routes with ``@db_endpoint``; listing it here serves them.
DB_ENDPOINT_MODULES: tuple[ModuleType, ...] = (
    experts_credentials,
    experts_db,
    experts_kickoff,
    experts_routine_jobs,
    experts_routines,
    experts_scheduling,
    experts_spend_approval,
    library_db,
    library_triggers,
    onboarding_dump_db,
    orgs_db,
    search_embeddings,
    search_hybrid_search,
    store_db,
    store_embeddings,
    store_skill_db,
    copilot_background_task_db,
    copilot_db,
    copilot_scheduled_chats_forget,
    copilot_scheduled_chats_service,
    copilot_sharing_db,
    account_deletion,
    account_erasure_request,
    account_erasure_sweep,
    activity_event,
    alerts,
    analytics,
    auth_oauth,
    auth_session,
    auth_ip_events,
    block,
    bot_analytics,
    bot_installs,
    briefing,
    briefing_data,
    credit,
    dream_pass,
    execution,
    feedback_prompts,
    graph_db,
    human_review,
    memory_schedule,
    onboarding,
    org_credit,
    platform_cost,
    posthog_lifecycle_sync,
    pro_activation,
    product_mail_audience,
    push_subscription,
    skill_deletion,
    skill_learning,
    skill_publication,
    skill_reviews,
    skill_revocations,
    skill_use,
    skill_versions,
    stripe_reconciliation,
    subscription_trial,
    understanding,
    user,
    workspace,
    workspace_folder,
    workspace_scope,
    workspace_skill,
    notifications_trial,
    platform_linking_db,
    # The adapters below, for functions whose own signature can't cross the RPC.
    sys.modules[__name__],
)


@db_endpoint
async def spend_credits(
    user_id: str,
    cost: int,
    metadata: credit.UsageTransactionMetadata,
    fail_insufficient_credits: bool = True,
) -> int:
    user_credit_model = await credit.get_user_credit_model(user_id)
    return await user_credit_model.spend_credits(
        user_id, cost, metadata, fail_insufficient_credits=fail_insufficient_credits
    )


@db_endpoint
async def get_credits(user_id: str) -> int:
    user_credit_model = await credit.get_user_credit_model(user_id)
    return await user_credit_model.get_credits(user_id)


@db_endpoint
async def store_content_embedding(
    content_type: ContentType,
    content_id: str,
    embedding: list[float],
    searchable_text: str,
    metadata: dict | None = None,
    user_id: str | None = None,
) -> bool:
    # Without the ``tx`` parameter, which can't cross the RPC boundary.
    return await search_embeddings.store_content_embedding(
        content_type=content_type,
        content_id=content_id,
        embedding=embedding,
        searchable_text=searchable_text,
        metadata=metadata,
        user_id=user_id,
    )


@db_endpoint
async def spend_org_credits(
    org_id: str,
    user_id: str,
    cost: int,
    metadata: credit.UsageTransactionMetadata,
    team_id: str | None = None,
    fail_insufficient_credits: bool = True,
) -> int:
    return await org_credit.spend_org_credits(
        org_id=org_id,
        user_id=user_id,
        amount=cost,
        team_id=team_id,
        metadata=metadata.model_dump(),
        fail_insufficient_credits=fail_insufficient_credits,
    )


@db_endpoint
async def get_org_credits(org_id: str) -> int:
    return await org_credit.get_org_credits(org_id)


# ``db_accessors.credit_db()`` hands out this module in a process with Prisma, so
# the billing functions its callers reach are here by name, beside the adapters.
get_personal_org_owner = org_credit.get_personal_org_owner
get_subscription_trial = subscription_trial.get_subscription_trial
list_trials_ending_unrenewed = subscription_trial.list_trials_ending_unrenewed
record_subscription_trial_cost = subscription_trial.record_subscription_trial_cost
reconcile_stripe_tier_for_user = credit.reconcile_stripe_tier_for_user
sync_subscription_from_stripe = credit.sync_subscription_from_stripe


class DatabaseManager(AppService):
    """Database connection pooling service.

    This service connects to the Prisma engine and exposes database
    operations via RPC endpoints. It acts as a centralized connection pool
    for all services that need database access.
    """

    @asynccontextmanager
    async def lifespan(self, app: "FastAPI"):
        async with super().lifespan(app):
            logger.info(f"[{self.service_name}] ⏳ Connecting to Database...")
            await db.connect()

            # Load the block registry before serving. The first RPC that needs a
            # block loads it otherwise, on this event loop, and every RPC waits.
            load_started = time.monotonic()
            block_count = len(get_blocks())
            logger.info(
                f"[{self.service_name}] Loaded {block_count} blocks in "
                f"{time.monotonic() - load_started:.1f}s"
            )

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

    @classmethod
    def serve(cls, module: ModuleType) -> None:
        """Expose the endpoints ``module`` declares, each under its function name."""
        for route, endpoint in declared_endpoints(module).items():
            if hasattr(cls, route):
                raise RuntimeError(
                    f"{module.__name__}.{route} is declared a database endpoint, "
                    f"but the database manager already has a {route!r}"
                )
            setattr(cls, route, expose(endpoint))


for _module in DB_ENDPOINT_MODULES:
    DatabaseManager.serve(_module)


def _method(f: Callable[P, R]) -> Callable[Concatenate[Any, P], R]:
    """Types a declared endpoint as a client method. The client only reads its
    name and signature to build the request, and never calls it."""
    return cast(Callable[Concatenate[Any, P], R], f)


def _blocking_method(f: Callable[P, Awaitable[R]]) -> Callable[Concatenate[Any, P], R]:
    return endpoint_to_sync(_method(f))


class DatabaseManagerClient(AppServiceClient):
    _ = _blocking_method

    @classmethod
    def get_service_type(cls):
        return DatabaseManager

    # Executions
    get_graph_executions = _(execution.get_graph_executions)
    get_graph_executions_count = _(execution.get_graph_executions_count)
    get_graph_execution = _(execution.get_graph_execution)
    get_graph_execution_meta = _(execution.get_graph_execution_meta)
    get_node_executions = _(execution.get_node_executions)
    update_node_execution_status = _(execution.update_node_execution_status)
    update_graph_execution_start_time = _(execution.update_graph_execution_start_time)
    update_graph_execution_stats = _(execution.update_graph_execution_stats)
    upsert_execution_output = _(execution.upsert_execution_output)

    # Graphs
    get_graph_metadata = _(graph_db.get_graph_metadata)
    get_node = _(graph_db.get_node)

    # Credits
    spend_credits = _(spend_credits)
    get_credits = _(get_credits)
    spend_org_credits = _(spend_org_credits)
    get_org_credits = _(get_org_credits)
    get_personal_org_owner = _(org_credit.get_personal_org_owner)

    # Orgs
    resolve_member_tenancy = _(orgs_db.resolve_member_tenancy)

    # Block error monitoring
    get_block_error_stats = _(execution.get_block_error_stats)
    # Execution accuracy monitoring
    get_accuracy_trends_and_alerts = _(analytics.get_accuracy_trends_and_alerts)
    # Auth identity invariant monitoring
    heal_orphaned_auth_identities = _(user.heal_orphaned_auth_identities)
    # Account erasures that started and did not finish
    stalled_account_erasures = _(account_erasure_request.stalled_account_erasures)
    get_frequently_executed_graphs = _(execution.get_frequently_executed_graphs)
    get_marketplace_graphs_for_monitoring = _(
        analytics.get_marketplace_graphs_for_monitoring
    )

    # Human In The Loop
    has_pending_reviews_for_graph_exec = _(
        human_review.has_pending_reviews_for_graph_exec
    )

    # User Emails
    get_user_email_by_id = _(user.get_user_email_by_id)

    # Alerts
    raise_alert_condition = _(alerts.raise_alert_condition)
    resolve_alert_condition = _(alerts.resolve_alert_condition)
    count_scheduled_agents = _(briefing_data.count_scheduled_agents)
    get_recent_daily_spend = _(credit.get_recent_daily_spend)

    # Library
    list_library_agents = _(library_db.list_library_agents)
    add_store_agent_to_library = _(library_db.add_store_agent_to_library)
    validate_graph_execution_permissions = _(
        graph_db.validate_graph_execution_permissions
    )

    # Expert run posts (executor completion hook)
    append_expert_run_message = _(copilot_db.append_expert_run_message)
    get_library_agent_id_by_graph_id = _(library_db.get_library_agent_id_by_graph_id)

    # Activity events (executor completion hook)
    create_activity_event = _(activity_event.create_activity_event)

    # Morning briefing (scheduler cron; runs Prisma-less)
    append_plain_session_message = _(copilot_db.append_plain_session_message)
    create_briefing = _(briefing.create_briefing)
    get_briefing_for_date = _(briefing.get_briefing_for_date)
    get_latest_briefings = _(briefing.get_latest_briefings)
    mark_briefing_delivered = _(briefing.mark_briefing_delivered)
    update_briefing_content = _(briefing.update_briefing_content)
    get_library_agent_refs_by_graph_ids = _(
        library_db.get_library_agent_refs_by_graph_ids
    )

    # Store
    get_store_agents = _(store_db.get_store_agents)
    get_store_agent_details = _(store_db.get_store_agent_details)

    # Store Embeddings
    get_embedding_stats = _(search_embeddings.get_embedding_stats)
    backfill_missing_embeddings = _(store_embeddings.backfill_missing_embeddings)
    cleanup_orphaned_embeddings = _(search_embeddings.cleanup_orphaned_embeddings)

    # Block Descriptions
    get_blocks_needing_optimization = _(block.get_blocks_needing_optimization)
    update_block_optimized_description = _(block.update_block_optimized_description)


class DatabaseManagerAsyncClient(AppServiceClient):
    _ = _method
    is_auth_session_active = _(auth_session.is_auth_session_active)

    @classmethod
    def get_service_type(cls):
        return DatabaseManager

    # ============ Graph Executions ============ #
    create_graph_execution = _(execution.create_graph_execution)
    get_child_graph_executions = _(execution.get_child_graph_executions)
    get_connected_output_nodes = _(graph_db.get_connected_output_nodes)
    get_latest_node_execution = _(execution.get_latest_node_execution)
    get_graph_execution = _(execution.get_graph_execution)
    get_graph_execution_meta = _(execution.get_graph_execution_meta)
    get_graph_executions = _(execution.get_graph_executions)
    get_node_execution = _(execution.get_node_execution)
    get_node_executions = _(execution.get_node_executions)
    update_graph_execution_stats = _(execution.update_graph_execution_stats)
    update_node_execution_status = _(execution.update_node_execution_status)
    update_node_execution_status_batch = _(execution.update_node_execution_status_batch)
    mark_node_executions_moderated = _(execution.mark_node_executions_moderated)
    upsert_execution_input = _(execution.upsert_execution_input)
    upsert_execution_output = _(execution.upsert_execution_output)
    get_execution_outputs_by_node_exec_id = _(
        execution.get_execution_outputs_by_node_exec_id
    )
    get_execution_kv_data = _(execution.get_execution_kv_data)
    set_execution_kv_data = _(execution.set_execution_kv_data)

    # ============ Graphs ============ #
    get_graph = _(graph_db.get_graph)
    get_copied_graph_ids = _(graph_db.get_copied_graph_ids)
    get_graph_metadata = _(graph_db.get_graph_metadata)
    get_graph_settings = _(graph_db.get_graph_settings)
    is_graph_version_owner = _(graph_db.is_graph_version_owner)
    get_node = _(graph_db.get_node)
    get_store_listed_graphs = _(graph_db.get_store_listed_graphs)

    # ============ User + Integrations ============ #
    get_user_by_id = _(user.get_user_by_id)
    get_user_default_chat_route = _(user.get_user_default_chat_route)
    get_user_subscription_tier = _(user.get_user_subscription_tier)
    get_usage_activation_state = _(pro_activation.get_usage_activation_state)
    get_subscription_trial = _(subscription_trial.get_subscription_trial)
    list_trials_ending_unrenewed = _(subscription_trial.list_trials_ending_unrenewed)
    sync_subscription_from_stripe = _(credit.sync_subscription_from_stripe)
    record_subscription_trial_cost = _(
        subscription_trial.record_subscription_trial_cost
    )
    get_auth_user_flag_fields = _(user.get_auth_user_flag_fields)
    is_auth_user_banned = _(user.is_auth_user_banned)
    is_account_closed = _(account_deletion.is_account_closed)
    heal_orphaned_auth_identities = _(user.heal_orphaned_auth_identities)
    get_user_integrations = _(user.get_user_integrations)
    update_user_integrations = _(user.update_user_integrations)
    get_near_expiry_oauth_credentials = _(user.get_near_expiry_oauth_credentials)
    get_user_credentials = _(user.get_user_credentials)
    set_user_credentials = _(user.set_user_credentials)

    # ============ Human In The Loop ============ #
    cancel_pending_reviews_for_execution = _(
        human_review.cancel_pending_reviews_for_execution
    )
    check_approval = _(human_review.check_approval)
    claim_approved_review = _(human_review.claim_approved_review)
    delete_review_as_read = _(human_review.delete_review_as_read)
    delete_review_by_node_exec_id = _(human_review.delete_review_by_node_exec_id)
    get_or_create_human_review = _(human_review.get_or_create_human_review)
    get_pending_reviews_for_execution = _(
        human_review.get_pending_reviews_for_execution
    )
    get_pending_reviews_for_chat_session = _(
        human_review.get_pending_reviews_for_chat_session
    )
    get_pending_reviews_for_user = _(human_review.get_pending_reviews_for_user)
    get_reviews_by_node_exec_ids = _(human_review.get_reviews_by_node_exec_ids)
    process_all_reviews_for_execution = _(
        human_review.process_all_reviews_for_execution
    )
    record_gate_approval = _(human_review.record_gate_approval)
    release_review_claim = _(human_review.release_review_claim)
    reopen_review = _(human_review.reopen_review)
    update_review_processed_status = _(human_review.update_review_processed_status)

    # ============ User Comms ============ #
    get_active_user_ids_in_timerange = _(user.get_active_user_ids_in_timerange)
    get_user_email_by_id = _(user.get_user_email_by_id)
    get_user_email_verification = _(user.get_user_email_verification)
    get_product_mail_standing = _(product_mail_audience.get_product_mail_standing)
    get_user_notification_preference = _(user.get_user_notification_preference)

    # ============ Alerts ============ #
    raise_alert_condition = _(alerts.raise_alert_condition)
    resolve_alert_condition = _(alerts.resolve_alert_condition)
    count_scheduled_agents = _(briefing_data.count_scheduled_agents)
    get_recent_daily_spend = _(credit.get_recent_daily_spend)

    # ============ Briefing / Alert engine (notification service) ============ #
    # The NotificationManager owns no Prisma connection, so its scheduled
    # passes read and write exclusively through these.
    get_users_with_matured_alerts = _(alerts.get_users_with_matured_alerts)
    get_pending_alert_conditions = _(alerts.get_pending_alert_conditions)
    count_alerts_sent_since = _(alerts.count_alerts_sent_since)
    mark_alert_conditions_sent = _(alerts.mark_alert_conditions_sent)
    mark_alert_conditions_deferred = _(alerts.mark_alert_conditions_deferred)
    get_briefing_alert_conditions = _(alerts.get_briefing_alert_conditions)
    mark_alert_conditions_briefed = _(alerts.mark_alert_conditions_briefed)
    get_agent_period_stats = _(briefing_data.get_agent_period_stats)
    get_top_scored_runs = _(briefing_data.get_top_scored_runs)
    count_active_agents = _(briefing_data.count_active_agents)
    get_briefing_credit_balance = _(briefing_data.get_briefing_credit_balance)
    get_briefing_candidates = _(user.get_briefing_candidates)
    get_briefing_candidate = _(user.get_briefing_candidate)
    set_last_briefing_at = _(user.set_last_briefing_at)
    get_billing_email_recipient = _(user.get_billing_email_recipient)
    claim_welcome_email = _(user.claim_welcome_email)
    is_marketing_opted_out = _(user.is_marketing_opted_out)
    release_welcome_email = _(user.release_welcome_email)

    # ============ Morning Briefing ============ #
    append_plain_session_message = _(copilot_db.append_plain_session_message)
    append_session_message = _(copilot_db.append_session_message)
    record_scheduled_run = _(copilot_scheduled_chats_service.record_scheduled_run)
    forget_scheduled_chat = _(copilot_scheduled_chats_forget.forget_scheduled_chat)
    create_briefing = _(briefing.create_briefing)
    get_briefing_for_date = _(briefing.get_briefing_for_date)
    get_latest_briefings = _(briefing.get_latest_briefings)
    mark_briefing_delivered = _(briefing.mark_briefing_delivered)
    update_briefing_content = _(briefing.update_briefing_content)

    # ============ Library ============ #
    list_library_agents = _(library_db.list_library_agents)
    get_library_agent_refs_by_graph_ids = _(
        library_db.get_library_agent_refs_by_graph_ids
    )
    add_store_agent_to_library = _(library_db.add_store_agent_to_library)
    create_graph_in_library = _(library_db.create_graph_in_library)
    create_library_agent = _(library_db.create_library_agent)
    get_library_agent = _(library_db.get_library_agent)
    get_library_agent_by_graph_id = _(library_db.get_library_agent_by_graph_id)
    update_library_agent = _(library_db.update_library_agent)
    update_graph_in_library = _(library_db.update_graph_in_library)
    validate_graph_execution_permissions = _(
        graph_db.validate_graph_execution_permissions
    )
    setup_triggered_preset = _(library_triggers.setup_triggered_preset)
    update_triggered_preset = _(library_triggers.update_triggered_preset)
    delete_preset_with_webhook_cleanup = _(
        library_triggers.delete_preset_with_webhook_cleanup
    )
    get_preset = _(library_db.get_preset)
    create_preset = _(library_db.create_preset)
    list_presets = _(library_db.list_presets)
    list_trigger_agents = _(library_db.list_trigger_agents)

    # ============ Library Folders ============ #
    create_folder = _(library_db.create_folder)
    list_folders = _(library_db.list_folders)
    get_folder = _(library_db.get_folder)
    get_folder_tree = _(library_db.get_folder_tree)
    update_folder = _(library_db.update_folder)
    move_folder = _(library_db.move_folder)
    delete_folder = _(library_db.delete_folder)
    bulk_move_agents_to_folder = _(library_db.bulk_move_agents_to_folder)
    get_folder_agents_map = _(library_db.get_folder_agents_map)
    get_root_agent_summaries = _(library_db.get_root_agent_summaries)

    # ============ Onboarding ============ #
    increment_onboarding_runs = _(onboarding.increment_onboarding_runs)
    clear_brain_dump_audio_paths = _(onboarding_dump_db.clear_brain_dump_audio_paths)

    # ============ OAuth ============ #
    cleanup_expired_oauth_tokens = _(auth_oauth.cleanup_expired_oauth_tokens)

    # ============ Sign-in IP log ============ #
    delete_expired_ip_events = _(auth_ip_events.delete_expired_ip_events)

    # ============ Store ============ #
    get_store_agents = _(store_db.get_store_agents)
    get_store_agent_details = _(store_db.get_store_agent_details)
    get_store_agent_by_version_id = _(store_db.get_store_agent_by_version_id)
    get_agent = _(store_db.get_agent)
    get_available_graph = _(store_db.get_available_graph)

    # ============ Search ============ #
    unified_hybrid_search = _(search_hybrid_search.unified_hybrid_search)
    get_content_embedding = _(search_embeddings.get_content_embedding)
    store_content_embedding = _(store_content_embedding)
    delete_content_embedding = _(search_embeddings.delete_content_embedding)

    # ============ Marketplace skills ============ #
    get_active_versions = _(store_skill_db.get_active_versions)
    get_version_packages = _(store_skill_db.get_version_packages)
    find_version_by_hash = _(store_skill_db.find_version_by_hash)

    # ============ Chat Sharing ============ #
    link_new_execution_to_chat_share = _(
        copilot_sharing_db.link_new_execution_to_chat_share
    )

    # ============ Workspace ============ #
    count_workspace_files = _(workspace.count_workspace_files)
    create_workspace_file = _(workspace.create_workspace_file)
    publish_workspace_skill_file = _(workspace_skill.publish_workspace_skill_file)
    skill_root_has_room = _(workspace_skill.skill_root_has_room)
    get_or_create_workspace = _(workspace.get_or_create_workspace)
    get_workspace_file = _(workspace.get_workspace_file)
    get_workspace_file_by_path = _(workspace.get_workspace_file_by_path)
    get_workspace_total_size = _(workspace.get_workspace_total_size)
    list_workspace_files = _(workspace.list_workspace_files)
    list_workspace_folders = _(workspace_folder.list_workspace_folders)
    soft_delete_workspace_file = _(workspace.soft_delete_workspace_file)
    resolve_expert_workspace_scope = _(workspace_scope.resolve_expert_workspace_scope)

    # ============ Credits ============ #
    spend_credits = _(spend_credits)
    get_credits = _(get_credits)
    spend_org_credits = _(spend_org_credits)
    get_org_credits = _(get_org_credits)
    get_personal_org_owner = _(org_credit.get_personal_org_owner)

    # ============ Understanding ============ #
    get_business_understanding = _(understanding.get_business_understanding)
    upsert_business_understanding = _(understanding.upsert_business_understanding)

    # ============ Block Descriptions ============ #
    get_blocks_needing_optimization = _(block.get_blocks_needing_optimization)

    # ============ Platform Cost Tracking ============ #
    log_platform_cost = _(platform_cost.log_platform_cost)

    # ============ Activity Events ============ #
    create_activity_event = _(activity_event.create_activity_event)

    # ============ Feedback Prompts ============ #
    claim_feedback_prompt = _(feedback_prompts.claim_feedback_prompt)
    count_feedback_prompts = _(feedback_prompts.count_feedback_prompts)

    # ============ Push Notifications ============ #
    get_user_push_subscriptions = _(push_subscription.get_user_push_subscriptions)
    delete_push_subscription = _(push_subscription.delete_push_subscription)
    increment_push_fail_count = _(push_subscription.increment_push_fail_count)
    cleanup_failed_push_subscriptions = _(
        push_subscription.cleanup_failed_push_subscriptions
    )

    # ============ Subscription Reconciliation ============ #
    reconcile_all_stripe_tiers = _(stripe_reconciliation.reconcile_all_stripe_tiers)
    reconcile_stripe_tier_for_user = _(credit.reconcile_stripe_tier_for_user)
    start_posthog_lifecycle_sweep = _(
        posthog_lifecycle_sync.start_posthog_lifecycle_sweep
    )
    start_account_erasure_sweep = _(account_erasure_sweep.start_account_erasure_sweep)
    request_posthog_lifecycle_sync = _(
        posthog_lifecycle_sync.request_posthog_lifecycle_sync
    )
    send_trial_last_calls = _(notifications_trial.send_trial_last_calls)

    # ============ Platform Linking ============ #
    find_server_link_owner = _(platform_linking_db.find_server_link_owner)
    get_user_default_team = _(orgs_db.get_user_default_team)
    is_active_member = _(orgs_db.is_active_member)
    resolve_default_tenancy = _(orgs_db.resolve_default_tenancy)
    resolve_member_tenancy = _(orgs_db.resolve_member_tenancy)
    find_user_link_owner = _(platform_linking_db.find_user_link_owner)
    resolve_server_link = _(platform_linking_db.resolve_server_link)
    resolve_user_link = _(platform_linking_db.resolve_user_link)
    create_server_link_token = _(platform_linking_db.create_server_link_token)
    create_user_link_token = _(platform_linking_db.create_user_link_token)
    get_link_token_status = _(platform_linking_db.get_link_token_status)
    get_link_token_info = _(platform_linking_db.get_link_token_info)
    confirm_server_link = _(platform_linking_db.confirm_server_link)
    confirm_user_link = _(platform_linking_db.confirm_user_link)
    list_server_links = _(platform_linking_db.list_server_links)
    list_user_links = _(platform_linking_db.list_user_links)
    refresh_server_link_name = _(platform_linking_db.refresh_server_link_name)
    delete_server_link = _(platform_linking_db.delete_server_link)
    delete_user_link = _(platform_linking_db.delete_user_link)
    cleanup_expired_platform_link_tokens = _(
        platform_linking_db.cleanup_expired_platform_link_tokens
    )
    fetch_workspace_artifact = _(platform_linking_db.fetch_workspace_artifact)

    # ============ Bot Installs ============ #
    get_bot_install = _(bot_installs.get_bot_install)
    is_install_revoked = _(bot_installs.is_install_revoked)
    upsert_bot_install = _(bot_installs.upsert_bot_install)
    revoke_bot_install = _(bot_installs.revoke_bot_install)

    # ============ Memory Scope Schedules ============ #
    get_scope_schedule = _(memory_schedule.get_scope_schedule)
    claim_scope_schedule = _(memory_schedule.claim_scope_schedule)
    record_scope_jobs = _(memory_schedule.record_scope_jobs)
    set_scope_state = _(memory_schedule.set_scope_state)
    forget_scope_job = _(memory_schedule.forget_scope_job)
    record_scope_run = _(memory_schedule.record_scope_run)
    list_due_community_scopes = _(memory_schedule.list_due_community_scopes)
    list_user_scope_schedules = _(memory_schedule.list_user_scope_schedules)
    list_scope_schedules = _(memory_schedule.list_scope_schedules)
    existing_user_ids = _(memory_schedule.existing_user_ids)

    # ============ Bot Analytics ============ #
    record_bot_event = _(bot_analytics.record_bot_event)
    record_guild_joined = _(bot_analytics.record_guild_joined)
    mark_guild_left = _(bot_analytics.mark_guild_left)
    sync_guild_presence = _(bot_analytics.sync_guild_presence)

    # ============ Experts ============ #
    get_expert = _(experts_db.get_expert)
    expert_row_exists = _(experts_db.expert_row_exists)
    resolve_attributable_expert = _(experts_db.resolve_attributable_expert)
    list_experts = _(experts_db.list_experts)
    resolve_private_expert_tenancy = _(experts_db.resolve_private_expert_tenancy)
    check_private_expert = _(experts_db.check_private_expert)
    get_autopilot_mode = _(experts_db.get_autopilot_mode)
    list_expert_identities = _(experts_db.list_expert_identities)
    owns_active_expert = _(experts_db.owns_active_expert)
    get_routine = _(experts_routines.get_routine)
    record_routine_thread = _(experts_routine_jobs.record_routine_thread)
    record_routine_fired = _(experts_routine_jobs.record_routine_fired)
    mark_routine_unscheduled = _(experts_routine_jobs.mark_routine_unscheduled)
    create_routine = _(experts_routines.create_routine)
    list_routines = _(experts_routines.list_routines)
    enable_routine = _(experts_routines.enable_routine)
    disable_routine = _(experts_routines.disable_routine)
    enforce_expert_run_budget = _(experts_scheduling.enforce_expert_run_budget)
    spend_approval_required = _(experts_spend_approval.spend_approval_required)
    park_execution_for_spend_approval = _(
        experts_spend_approval.park_execution_for_spend_approval
    )
    parked_spend_decision = _(experts_spend_approval.parked_spend_decision)
    open_chat_spend_review = _(experts_spend_approval.open_chat_spend_review)
    expert_allowed_credential_ids = _(experts_credentials.expert_allowed_credential_ids)
    seeded_expert_credential_ids = _(experts_credentials.seeded_expert_credential_ids)
    settle_credential_seed = _(experts_credentials.settle_credential_seed)
    update_soul = _(experts_db.update_soul)
    update_soul_if_current = _(experts_db.update_soul_if_current)
    update_soul_fields = _(experts_db.update_soul_fields)
    update_soul_fields_if_current = _(experts_db.update_soul_fields_if_current)
    add_expert_skill_name = _(experts_db.add_expert_skill_name)
    add_expert_skill_names = _(experts_db.add_expert_skill_names)
    remove_expert_skill_name = _(experts_db.remove_expert_skill_name)
    set_expert_learning_paused = _(experts_db.set_expert_learning_paused)

    # ============ Skill learning ============ #
    upsert_source_revision = _(skill_learning.upsert_source_revision)
    get_source = _(skill_learning.get_source)
    get_source_by_ref = _(skill_learning.get_source_by_ref)
    list_pending_sources = _(skill_learning.list_pending_sources)
    list_pending_owner_keys = _(skill_learning.list_pending_owner_keys)
    get_review_for_revision = _(skill_reviews.get_review_for_revision)
    reserve_model_review = _(skill_reviews.reserve_model_review)
    checkpoint_review_usage = _(skill_reviews.checkpoint_review_usage)
    set_review_replay = _(skill_reviews.set_review_replay)
    require_owned_source = _(skill_learning.require_owned_source)
    set_source_eligibility = _(skill_learning.set_source_eligibility)
    set_source_approval = _(skill_learning.set_source_approval)
    mark_source_attempted = _(skill_learning.mark_source_attempted)
    advance_source_cursor = _(skill_learning.advance_source_cursor)
    upsert_review = _(skill_reviews.upsert_review)
    get_review = _(skill_reviews.get_review)
    list_reviews = _(skill_reviews.list_reviews)
    summarize_learning = _(skill_reviews.summarize_learning)
    get_head = _(skill_versions.get_head)
    ensure_head = _(skill_versions.ensure_head)
    list_heads = _(skill_versions.list_heads)
    list_use_paused_skill_names = _(skill_versions.list_use_paused_skill_names)
    update_head_policy = _(skill_versions.update_head_policy)
    pause_use_if_current = _(skill_versions.pause_use_if_current)
    publish_version_if_current = _(skill_versions.publish_version_if_current)
    commit_version_safe = _(skill_publication.commit_version_safe)
    complete_publication = _(skill_publication.complete_publication)
    abandon_publication = _(skill_publication.abandon_publication)
    list_pending_publications = _(skill_publication.list_pending_publications)
    get_pending_publication_for_skill = _(
        skill_publication.get_pending_publication_for_skill
    )
    create_version = _(skill_versions.create_version)
    set_version_state = _(skill_versions.set_version_state)
    get_version = _(skill_versions.get_version)
    list_versions = _(skill_versions.list_versions)
    list_recent_versions = _(skill_versions.list_recent_versions)
    list_open_decisions = _(skill_versions.list_open_decisions)
    list_versions_in_states = _(skill_versions.list_versions_in_states)
    invalidate_source_versions = _(skill_revocations.invalidate_source_versions)
    latest_independent_version_id = _(skill_revocations.latest_independent_version_id)
    record_skill_deletion = _(skill_deletion.record_skill_deletion)
    add_suppression = _(skill_use.add_suppression)
    find_suppression = _(skill_use.find_suppression)
    list_suppressions = _(skill_use.list_suppressions)
    record_use_event = _(skill_use.record_use_event)
    list_use_events = _(skill_use.list_use_events)

    install_workflow = _(experts_db.install_workflow)
    remove_workflow = _(experts_db.remove_workflow)
    get_workflow_label = _(experts_db.get_workflow_label)
    grant_expert_credentials = _(experts_credentials.grant_expert_credentials)
    revoke_expert_credential = _(experts_credentials.revoke_expert_credential)
    list_expert_credentials = _(experts_credentials.list_expert_credentials)
    list_templates = _(experts_db.list_templates)
    get_template = _(experts_db.get_template)
    with_bundled_skills = _(experts_db.with_bundled_skills)
    hire_expert = _(experts_db.hire_expert)
    start_expert_kickoff = _(experts_kickoff.start_expert_kickoff)
    expert_setup_status = _(experts_db.expert_setup_status)
    create_raised_expert = _(experts_db.create_raised_expert)
    count_active_experts = _(experts_db.count_active_experts)
    active_expert_ids = _(experts_db.active_expert_ids)
    count_raised_experts = _(experts_db.count_raised_experts)

    # ============ CoPilot Chat Sessions ============ #
    # NOTE: no eager-load `get_chat_session` here — callers go through
    # `get_chat_messages_paginated` (with `limit=MAX_LOADED_CHAT_MESSAGES`) so
    # tool-pair boundary expansion + visibility guarantees apply uniformly
    # across the LLM-context and UI scroll paths.
    get_chat_session_metadata = _(copilot_db.get_chat_session_metadata)
    get_background_task = _(copilot_background_task_db.get_background_task)
    get_session_background_state = _(
        copilot_background_task_db.get_session_background_state
    )
    create_background_task = _(copilot_background_task_db.create_background_task)
    claim_background_turn = _(copilot_background_task_db.claim_background_turn)
    finish_background_turn = _(copilot_background_task_db.finish_background_turn)
    complete_background_task = _(copilot_background_task_db.complete_background_task)
    list_background_tasks = _(copilot_background_task_db.list_background_tasks)
    enqueue_background_followup = _(
        copilot_background_task_db.enqueue_background_followup
    )
    suppress_background_followup = _(
        copilot_background_task_db.suppress_background_followup
    )
    release_deleted_background_turn = _(
        copilot_background_task_db.release_deleted_background_turn
    )
    release_abandoned_background_turn = _(
        copilot_background_task_db.release_abandoned_background_turn
    )
    background_child_result = _(copilot_background_task_db.background_child_result)
    has_open_handoffs = _(copilot_background_task_db.has_open_handoffs)
    cancel_background_tasks = _(copilot_background_task_db.cancel_background_tasks)
    cancel_queued_background_followup = _(
        copilot_background_task_db.cancel_queued_background_followup
    )
    cancel_orphaned_background_tasks = _(
        copilot_background_task_db.cancel_orphaned_background_tasks
    )
    get_chat_messages_paginated = _(copilot_db.get_chat_messages_paginated)
    create_chat_session = _(copilot_db.create_chat_session)
    update_chat_session = _(copilot_db.update_chat_session)
    add_chat_message = _(copilot_db.add_chat_message)
    add_chat_messages_batch = _(copilot_db.add_chat_messages_batch)
    get_chat_messages_by_ids = _(copilot_db.get_chat_messages_by_ids)
    append_expert_run_message = _(copilot_db.append_expert_run_message)
    get_library_agent_id_by_graph_id = _(library_db.get_library_agent_id_by_graph_id)
    get_user_chat_sessions = _(copilot_db.get_user_chat_sessions)
    list_recent_chat_sessions = _(copilot_db.list_recent_chat_sessions)
    get_user_chat_session_groups = _(copilot_db.get_user_chat_session_groups)
    set_session_pending_question = _(copilot_db.set_session_pending_question)
    clear_session_pending_question = _(copilot_db.clear_session_pending_question)
    get_sessions_with_pending_question = _(
        copilot_db.get_sessions_with_pending_question
    )
    get_user_session_count = _(copilot_db.get_user_session_count)
    delete_chat_session = _(copilot_db.delete_chat_session)
    get_next_sequence = _(copilot_db.get_next_sequence)
    update_tool_message_content = _(copilot_db.update_tool_message_content)
    update_message_content_by_sequence = _(
        copilot_db.update_message_content_by_sequence
    )
    add_dream_narrative = _(copilot_db.add_dream_narrative)
    blank_dream_narratives = _(copilot_db.blank_dream_narratives)
    update_chat_message_stamps = _(copilot_db.update_chat_message_stamps)
    update_chat_message_tool_calls = _(copilot_db.update_chat_message_tool_calls)
    update_chat_session_title = _(copilot_db.update_chat_session_title)
    update_chat_session_llm_route = _(copilot_db.update_chat_session_llm_route)
    update_chat_session_autopilot_mode = _(
        copilot_db.update_chat_session_autopilot_mode
    )
    update_chat_session_pinned = _(copilot_db.update_chat_session_pinned)
    set_turn_duration = _(copilot_db.set_turn_duration)
    count_chat_sessions_by_status = _(copilot_db.count_chat_sessions_by_status)
    list_chat_sessions_by_status = _(copilot_db.list_chat_sessions_by_status)
    update_chat_session_status = _(copilot_db.update_chat_session_status)
    admit_chat_session_turn = _(copilot_db.admit_chat_session_turn)
    record_chat_session_turn = _(copilot_db.record_chat_session_turn)
    get_chat_session_status = _(copilot_db.get_chat_session_status)
    get_latest_user_message_in_session = _(
        copilot_db.get_latest_user_message_in_session
    )
    add_chat_message = _(copilot_db.add_chat_message)

    # ============ Dream Passes ============ #
    create_dream_pass = _(dream_pass.create_dream_pass)
    update_dream_pass = _(dream_pass.update_dream_pass)
    get_dream_pass = _(dream_pass.get_dream_pass)
    get_dream_pass_for_user = _(dream_pass.get_dream_pass_for_user)
    list_open_dream_passes = _(dream_pass.list_open_dream_passes)
    list_dream_passes = _(dream_pass.list_dream_passes)
    list_expired_dream_passes = _(dream_pass.list_expired_dream_passes)
    list_dream_pass_cleanups = _(dream_pass.list_dream_pass_cleanups)
    delete_old_dream_passes = _(dream_pass.delete_old_dream_passes)
    erase_dream_passes = _(dream_pass.erase_dream_passes)
