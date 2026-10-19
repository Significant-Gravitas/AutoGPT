from backend.util.service import EXPOSED_FLAG

from .db_manager import DatabaseManager, DatabaseManagerAsyncClient


def test_async_client_exposes_chat_methods() -> None:
    assert hasattr(DatabaseManagerAsyncClient, "delete_chat_session")
    assert hasattr(DatabaseManagerAsyncClient, "set_turn_duration")
    assert hasattr(DatabaseManager, "update_chat_session_llm_route")
    assert hasattr(DatabaseManagerAsyncClient, "update_chat_session_llm_route")


def test_bot_analytics_methods_registered() -> None:
    for method in (
        "record_bot_event",
        "record_guild_joined",
        "mark_guild_left",
        "sync_guild_presence",
    ):
        assert hasattr(DatabaseManager, method)
        assert hasattr(DatabaseManagerAsyncClient, method)


def test_add_store_agent_rpc_request_schema_is_constructible() -> None:
    manager = DatabaseManager()
    manager._create_fastapi_endpoint(manager.add_store_agent_to_library)


def test_every_exposed_method_has_a_constructible_rpc_schema() -> None:
    """The service builds them all as it starts. One it can't build stops the
    database manager, and every run and call waiting on it hangs."""
    manager = DatabaseManager()
    failures = []
    for name, method in vars(DatabaseManager).items():
        if not getattr(method, EXPOSED_FLAG, False):
            continue
        try:
            manager._create_fastapi_endpoint(method)
        except Exception as error:
            failures.append(f"{name}: {type(error).__name__}: {error}")

    assert not failures, "\n".join(failures)


def test_memory_schedule_methods_registered() -> None:
    """The scheduler and copilot-executor have no Prisma client; the
    schedule registry reaches its rows through these."""
    for method in (
        "get_scope_schedule",
        "claim_scope_schedule",
        "record_scope_jobs",
        "set_scope_state",
        "forget_scope_job",
        "record_scope_run",
        "list_user_scope_schedules",
        "list_scope_schedules",
        "existing_user_ids",
    ):
        assert hasattr(DatabaseManager, method)
        assert hasattr(DatabaseManagerAsyncClient, method)


def test_memory_schedule_rpc_request_schemas_are_constructible() -> None:
    manager = DatabaseManager()
    manager._create_fastapi_endpoint(manager.claim_scope_schedule)
    manager._create_fastapi_endpoint(manager.record_scope_run)
    manager._create_fastapi_endpoint(manager.existing_user_ids)


def test_the_audience_consumer_can_read_the_opt_out() -> None:
    assert hasattr(DatabaseManager, "is_marketing_opted_out")
    assert hasattr(DatabaseManagerAsyncClient, "is_marketing_opted_out")
