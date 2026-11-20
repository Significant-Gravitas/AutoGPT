"""Dict keys starting with "_sa" cross the internal RPC both ways.

FastAPI's ``jsonable_encoder`` drops every such key by default, at any depth.
The client encoded a call's arguments with it, and FastAPI encoded each
method's result with it, so the executor's calls to the database manager lost
them from block data such as ``{"_sales_total": 42}``.
"""

import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI

from backend.util import service_auth
from backend.util.service import (
    AppService,
    AppServiceClient,
    endpoint_to_sync,
    expose,
    get_service_client,
)
from backend.util.settings import Settings

REPORT = {"_sales_total": 42, "rows": [{"_sa_region": "EMEA"}]}


class EchoService(AppService):
    port = 0

    @classmethod
    def get_port(cls) -> int:
        return cls.port

    @expose
    def echo(self, report: dict[str, Any]) -> dict[str, Any]:
        return report

    @expose
    async def echo_async(self, report: dict[str, Any]) -> dict[str, Any]:
        return report


class EchoServiceClient(AppServiceClient):
    @classmethod
    def get_service_type(cls):
        return EchoService

    echo = EchoService.echo
    echo_async = endpoint_to_sync(EchoService.echo_async)


def test_keys_starting_with_sa_cross_the_rpc_both_ways(
    monkeypatch: pytest.MonkeyPatch,
):
    settings = Settings()
    settings.secrets.internal_api_key = "internal-key-for-tests-0123456789abcdef"
    monkeypatch.setattr(service_auth, "Settings", lambda: settings)
    monkeypatch.setenv(f"{EchoService.__name__.upper()}_HOST", "127.0.0.1")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        monkeypatch.setattr(EchoService, "port", probe.getsockname()[1])

    with _serving(EchoService().build_fastapi_app()):
        client = get_service_client(EchoServiceClient)
        try:
            assert client.echo(REPORT) == REPORT
            assert client.echo_async(REPORT) == REPORT
        finally:
            client.close()


@contextmanager
def _serving(app: FastAPI) -> Iterator[None]:
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=EchoService.port,
            lifespan="off",
            log_config=None,
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError("test RPC server did not start")
        time.sleep(0.02)
    try:
        yield
    finally:
        server.should_exit = True
        thread.join(timeout=10)
