from backend.util.service import EXPOSED_FLAG

from .db_manager import DatabaseManager, DatabaseManagerAsyncClient


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


def test_all_rpc_request_schemas_are_constructible() -> None:
    manager = DatabaseManager()

    for name, attr in vars(DatabaseManager).items():
        if not getattr(attr, EXPOSED_FLAG, False):
            continue

        try:
            manager._create_fastapi_endpoint(attr)
        except Exception as exc:
            raise AssertionError(f"RPC request schema failed for {name}") from exc
