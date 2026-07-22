from .db_manager import (
    DatabaseManager,
    DatabaseManagerAsyncClient,
    DatabaseManagerClient,
)


def test_async_client_exposes_chat_methods() -> None:
    assert hasattr(DatabaseManagerAsyncClient, "delete_chat_session")
    assert hasattr(DatabaseManagerAsyncClient, "set_turn_duration")


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


def test_owner_grant_validator_is_registered_on_rpc_service_and_client() -> None:
    assert hasattr(DatabaseManager, "validate_execution_credentials_owner")
    assert hasattr(DatabaseManagerAsyncClient, "validate_execution_credentials_owner")
    manager = DatabaseManager()
    manager._create_fastapi_endpoint(manager.validate_execution_credentials_owner)


def test_shared_memory_access_is_registered_on_rpc_clients() -> None:
    for method in (
        "get_shared_memory_org_access",
        "list_shared_memory_team_access",
        "get_shared_memory_hold_buffer",
    ):
        assert method in DatabaseManager.__dict__
        assert method in DatabaseManagerClient.__dict__
        assert method in DatabaseManagerAsyncClient.__dict__
