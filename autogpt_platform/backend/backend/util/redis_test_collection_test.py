import runpy
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.data import redis_client


@pytest.mark.parametrize(
    "test_module",
    [
        "api/conn_manager_integration_test.py",
        "data/e2e_redis_rabbit_test.py",
        "data/event_bus_test.py",
        "data/redis_client_test.py",
    ],
)
def test_collection_does_not_call_retrying_redis_client(test_module: str) -> None:
    with (
        patch.object(
            redis_client, "connect", side_effect=ConnectionError("Redis unavailable")
        ) as connect,
        patch("socket.create_connection", side_effect=OSError("port closed")),
    ):
        runpy.run_path(str(Path(__file__).parents[1] / test_module))

    connect.assert_not_called()
