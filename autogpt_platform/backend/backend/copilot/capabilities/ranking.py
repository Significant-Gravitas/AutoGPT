"""Connection state and the class/connection weighting applied on top of BM25.

Order of preference, from the plan: connected service > unconnected service
> primitive with credentials for the host > bare primitive.  It decides
between entries that match the query equally well; it never lifts a weak
lexical match over a strong one (see ``index._ranked``).
"""

from pydantic import BaseModel, ConfigDict, Field

from .models import CapabilityEntry

WEIGHT_CONNECTED_SERVICE = 1.25
WEIGHT_SERVICE = 1.0
WEIGHT_PRIMITIVE_WITH_CREDENTIALS = 0.85
WEIGHT_PRIMITIVE = 0.75


class ConnectionState(BaseModel):
    """The user's credentials, keyed the three ways capabilities need them."""

    model_config = ConfigDict(frozen=True)

    providers: frozenset[str] = Field(default_factory=frozenset)
    server_urls: frozenset[str] = Field(default_factory=frozenset)
    hosts: frozenset[str] = Field(default_factory=frozenset)


def resolve_connected(
    entry: CapabilityEntry, state: ConnectionState | None
) -> bool | None:
    """Whether *entry* can run for this user, or ``None`` when unknowable here.

    Host-keyed primitives (authenticated HTTP) depend on the request URL, and
    multi-provider blocks (the LLM blocks) on the provider the user picks, so
    both stay ``None`` until describe/run time.
    """
    connection = entry.connection
    if not connection.required or state is None or connection.key is None:
        return None
    if connection.key_type == "provider":
        return connection.key in state.providers
    if connection.key_type == "server_url":
        return normalize_server_url(connection.key) in {
            normalize_server_url(url) for url in state.server_urls
        }
    return None


def tier(entry: CapabilityEntry, connected: bool | None) -> int:
    """0 = connected service ... 3 = bare primitive; used for tie-breaks."""
    if entry.kind == "skill":
        # The owner's own procedure: nothing to connect, written for this
        # user, so it ranks with the services they have connected.
        return 0
    if entry.klass == "service":
        return 0 if connected else 1
    # A host-keyed primitive resolves its credential from the request URL at
    # call time, so ``connected`` is None rather than False. Reading that as
    # "no credentials" ranked it below primitives that genuinely have none.
    if connected is None and entry.connection.key_type == "host":
        return 2
    return 2 if connected else 3


def class_weight(entry: CapabilityEntry, connected: bool | None) -> float:
    return {
        0: WEIGHT_CONNECTED_SERVICE,
        1: WEIGHT_SERVICE,
        2: WEIGHT_PRIMITIVE_WITH_CREDENTIALS,
        3: WEIGHT_PRIMITIVE,
    }[tier(entry, connected)]


def normalize_server_url(url: str) -> str:
    """Compare MCP server URLs the way the catalog and the user write them.

    Catalog keys are stored both ways (``https://mcp.miro.com/`` alongside
    ``https://mcp.linear.app/mcp``), so a raw ``==`` against whatever the
    model passes decides "is this a catalog server" on a trailing slash.
    """
    return url.strip().lower().rstrip("/")
