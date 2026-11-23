"""The options the REST API runs under uvicorn with, apart from the app.

``AgentServer.run`` serves the API with them, and ``uvicorn_options_test.py``
serves a stub app with the same ones, so what that test proves holds for the
API.
"""

import platform
from typing import Literal, TypedDict

from backend.util.settings import Config
from backend.util.trusted_proxies import uvicorn_forwarded_allow_ips


class UvicornOptions(TypedDict):
    host: str
    port: int
    log_config: None
    http: Literal["httptools"]
    loop: Literal["uvloop", "auto"]
    ws: Literal["websockets-sansio"]
    forwarded_allow_ips: str | None
    timeout_keep_alive: int


def rest_api_uvicorn_options(config: Config) -> UvicornOptions:
    return {
        "host": config.agent_api_host,
        "port": config.agent_api_port,
        "log_config": None,
        "http": "httptools",
        # uvloop doesn't run on Windows.
        "loop": "uvloop" if platform.system() != "Windows" else "auto",
        "ws": "websockets-sansio",
        "forwarded_allow_ips": uvicorn_forwarded_allow_ips(config.trusted_proxies),
        # uvicorn's default, 5 s, closes a connection the load balancer still
        # counts on reusing.
        "timeout_keep_alive": config.server_keep_alive_timeout_seconds,
    }
