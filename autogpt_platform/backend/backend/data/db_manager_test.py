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


def test_the_scheduler_can_post_a_note_into_a_chat() -> None:
    """The scheduler has no Prisma of its own; a skipped run's note goes
    through this RPC."""
    assert hasattr(DatabaseManagerAsyncClient, "append_session_message")
    manager = DatabaseManager()
    manager._create_fastapi_endpoint(manager.append_session_message)
