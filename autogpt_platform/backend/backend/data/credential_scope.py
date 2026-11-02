"""What a user/password login may be used for, defined per provider.

A login's ``scope`` is a URI whose meaning depends on its provider. Blocks that
connect to a server named in their inputs only send a login to the server its
scope names.
"""

import re

from pydantic import BaseModel

from backend.integrations.providers import ProviderName

_SERVER_URI = re.compile(
    r"(?P<scheme>[a-z][a-z0-9+.-]*)://"
    r"(?:\[(?P<ipv6>[0-9a-f:.]+)\]|(?P<host>[a-z0-9._-]+))"
    r"(?::(?P<port>[0-9]{1,5}))?"
)


class ServerTarget(BaseModel):
    """The server a block is about to send a login to."""

    scheme: str
    host: str
    port: int | None = None


class ServerScope(BaseModel):
    """A login for one server: ``<scheme>://host[:port]``. It matches only the
    same scheme and the exact host, and the port too when the scope names one.
    No wildcards, paths or credentials in the URI."""

    schemes: frozenset[str]
    example_host: str
    example_port: int

    def parse(self, scope: str) -> ServerTarget | None:
        match = _SERVER_URI.fullmatch(scope.strip().lower())
        if not match or match["scheme"] not in self.schemes:
            return None
        port = int(match["port"]) if match["port"] else None
        if port is not None and not 1 <= port <= 65535:
            return None
        return ServerTarget(
            scheme=match["scheme"], host=match["ipv6"] or match["host"], port=port
        )

    def matches(self, scope: str, target: ServerTarget) -> bool:
        bound = self.parse(scope)
        if bound is None or bound.scheme != target.scheme:
            return False
        if bound.host != _normalize_host(target.host):
            return False
        return bound.port is None or bound.port == target.port


SCOPE_RULES: dict[str, ServerScope] = {
    ProviderName.SMTP.value: ServerScope(
        schemes=frozenset({"smtp"}),
        example_host="smtp.example.com",
        example_port=587,
    ),
    ProviderName.DATABASE.value: ServerScope(
        schemes=frozenset({"postgresql", "mysql", "mssql"}),
        example_host="db.example.com",
        example_port=5432,
    ),
}
"""Providers whose logins a block sends to a server named by the graph. A new
login for one of them has to say which server it is for."""


def scope_rule(provider: str) -> ServerScope | None:
    return SCOPE_RULES.get(ProviderName(provider).value)


def scope_error(provider: str, scope: str | None) -> str | None:
    """Why a user/password login for `provider` can't be saved with `scope`."""
    rule = scope_rule(provider)
    if rule is None:
        if scope is None:
            return None
        return f"Logins for {provider} don't take a scope"
    # People only ever see a "Server" field, so these speak of the server.
    example = f"{rule.example_host} or {rule.example_host}:{rule.example_port}"
    if scope is None or not scope.strip():
        return f"Enter the server this login is for, like {example}"
    if rule.parse(scope) is None:
        return f"Enter a valid server, like {example} (port 1-65535)"
    return None


def _normalize_host(host: str) -> str:
    host = host.strip().lower()
    if host.startswith("[") and host.endswith("]"):
        return host[1:-1]
    return host
