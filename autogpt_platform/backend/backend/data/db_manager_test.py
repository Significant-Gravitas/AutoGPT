from .db_manager import DatabaseManagerAsyncClient


def test_async_client_exposes_memory_episode_log_method() -> None:
    assert hasattr(DatabaseManagerAsyncClient, "create_memory_episode_log")
