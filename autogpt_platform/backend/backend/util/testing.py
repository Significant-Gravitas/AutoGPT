"""Lightweight helpers for tests that must inspect local infrastructure."""

import socket
from functools import cache


@cache
def is_tcp_port_reachable(
    host: str,
    port: int,
    *,
    timeout: float = 1.0,
) -> bool:
    """Return whether a TCP endpoint accepts a connection within ``timeout``.

    This deliberately avoids service clients with retry policies so it is safe
    to call from module-level pytest skip markers during collection. Results
    are cached so modules that share an endpoint only wait for one probe.
    """
    try:
        connection = socket.create_connection((host, port), timeout=timeout)
    except OSError:
        return False

    connection.close()
    return True
