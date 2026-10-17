"""Turn what the model hands us into a registry entry, and load the
connection state that decides how entries rank for this user."""

import logging
from urllib.parse import urlsplit

from backend.copilot.capabilities.index import CapabilityIndex
from backend.copilot.capabilities.models import CapabilityEntry
from backend.copilot.capabilities.ranking import ConnectionState, normalize_server_url
from backend.copilot.capabilities.text import normalize_name
from backend.data.model import Credentials, HostScopedCredentials, OAuth2Credentials
from backend.integrations.credentials_store import canonical_provider
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.integrations.providers import ProviderName

logger = logging.getLogger(__name__)


def resolve_entry(index: CapabilityIndex, capability_id: str) -> CapabilityEntry | None:
    """Accept an entry id, a bare block uuid or tool name, a block class name,
    an MCP host, or an MCP server URL."""
    key = (capability_id or "").strip()
    if not key:
        return None
    entry = index.get(key)
    if entry is None and "://" not in key:
        entry = index.get(key.lower())
    if entry is not None:
        return entry
    if "://" in key:
        parsed = urlsplit(key)
        if (
            parsed.scheme != "https"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            return None
        host = parsed.hostname
        wanted_url = normalize_server_url(key)
        by_host = index.get(f"mcp:{host}") if host else None
        # Two catalog presets can share a host, so those entries are keyed by
        # slug. Prefer an exact endpoint before the legacy host fallback;
        # session resolution keeps stored custom URLs out of that fallback.
        return next(
            (
                e
                for e in index.entries
                if e.connection.key
                and normalize_server_url(e.connection.key) == wanted_url
            ),
            by_host,
        )
    if not key.startswith(("tool:", "block:", "mcp:")):
        entry = index.get(f"mcp:{key.lower()}")
        if entry is not None:
            return entry
    wanted = normalize_name(key.split(":", 1)[-1])
    return next(
        (e for e in index.entries if normalize_name(e.name) == wanted),
        None,
    )


async def load_connection_state(
    user_id: str, expert_id: str | None = None
) -> ConnectionState:
    """Providers, MCP server URLs and hosts this session holds credentials for.

    One store read per call; the credentials manager already caches, so a
    ``find_capability`` turn costs no extra round-trips.

    With an *expert_id* the state is narrowed to the expert's grants — the
    same filter a run applies — so ``connected`` means "this session can run
    it", and the account credentials the expert lacks are kept aside as
    ``ungranted``. A grant lookup that fails leaves the expert with nothing
    usable, matching the run-time gate rather than overstating what it can do.
    """
    try:
        credentials = await IntegrationCredentialsManager().store.get_all_creds(user_id)
    except Exception:
        logger.warning("Could not load credentials for ranking", exc_info=True)
        return ConnectionState()
    if expert_id is None:
        return connection_state_from(credentials)

    from backend.copilot.tools.utils import scope_credentials_to_expert

    try:
        granted = await scope_credentials_to_expert(user_id, expert_id, credentials)
    except Exception:
        logger.warning("Could not load expert credential grants", exc_info=True)
        return ConnectionState(ungranted=connection_state_from(credentials))
    granted_ids = {c.id for c in granted}
    ungranted = [c for c in credentials if c.id not in granted_ids]
    return connection_state_from(granted).model_copy(
        update={"ungranted": connection_state_from(ungranted)}
    )


def connection_state_from(credentials: list[Credentials]) -> ConnectionState:
    providers: set[str] = set()
    server_urls: set[str] = set()
    hosts: set[str] = set()
    for credential in credentials:
        # Credentials stored under Python 3.13 carry ``"ProviderName.MCP"``
        # where they mean ``"mcp"``; ranking compares against the canonical
        # value, so an un-normalised one reads as a service the user never
        # connected.
        provider = canonical_provider(credential.provider)
        providers.add(provider)
        if isinstance(credential, HostScopedCredentials):
            hosts.add(credential.host.lower())
        if provider == ProviderName.MCP.value and isinstance(
            credential, OAuth2Credentials
        ):
            url = (credential.metadata or {}).get("mcp_server_url")
            if isinstance(url, str) and url:
                server_urls.add(url)
    return ConnectionState(
        providers=frozenset(providers),
        server_urls=frozenset(server_urls),
        hosts=frozenset(hosts),
    )
