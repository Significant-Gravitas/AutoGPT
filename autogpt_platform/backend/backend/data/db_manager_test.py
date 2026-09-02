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


def test_exposed_attribute_names_match_function_names() -> None:
    """The AppService registers each exposed method's route under its
    *attribute* name, but the generated clients call the wrapped *function's*
    ``__name__`` — a mismatch 404s every RPC call to that method (this is how
    the task-spine writers silently failed to create receipts)."""
    mismatches = [
        (attr, fn.__name__)
        for attr, fn in vars(DatabaseManager).items()
        if getattr(fn, EXPOSED_FLAG, False) and attr != fn.__name__
    ]
    assert mismatches == []
