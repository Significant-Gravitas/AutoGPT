"""Shared MCP helpers used by blocks, copilot tools, and API routes."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from backend.data.model import OAuth2Credentials

logger = logging.getLogger(__name__)


def normalize_mcp_url(url: str) -> str:
    """Normalize an MCP server URL for consistent credential matching.

    Strips leading/trailing whitespace and a single trailing slash so that
    ``https://mcp.example.com/`` and ``https://mcp.example.com`` resolve to
    the same stored credential.
    """
    return url.strip().rstrip("/")


def server_host(server_url: str) -> str:
    """Extract the hostname from a server URL for display purposes.

    Uses ``parsed.hostname`` (never ``netloc``) to strip any embedded
    username/password before surfacing the value in UI messages.
    """
    try:
        parsed = urlparse(server_url)
        return parsed.hostname or server_url
    except Exception:
        return server_url


def parse_mcp_content(content: list[dict[str, Any]]) -> Any:
    """Parse MCP tool response content into a plain Python value.

    - text items: parsed as JSON when possible, kept as str otherwise
    - image items: kept as ``{type, data, mimeType}`` dict for frontend rendering
    - resource items: unwrapped to their resource payload dict

    Single-item responses are unwrapped from the list; multiple items are
    returned as a list; empty content returns ``None``.
    """
    output_parts: list[Any] = []
    for item in content:
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text", "")
            try:
                output_parts.append(json.loads(text))
            except (json.JSONDecodeError, ValueError):
                output_parts.append(text)
        elif item_type == "image":
            output_parts.append(
                {
                    "type": "image",
                    "data": item.get("data"),
                    "mimeType": item.get("mimeType"),
                }
            )
        elif item_type == "resource":
            output_parts.append(item.get("resource", {}))

    if len(output_parts) == 1:
        return output_parts[0]
    return output_parts or None


async def auto_lookup_mcp_credential(
    user_id: str, server_url: str
) -> OAuth2Credentials | None:
    """Look up the best stored MCP credential for *server_url*.

    The caller should pass a **normalized** URL (via :func:`normalize_mcp_url`)
    so the comparison with ``mcp_server_url`` in credential metadata matches.

    Returns the credential with the latest ``access_token_expires_at``, refreshed
    if needed, or ``None`` when no match is found.
    """
    from backend.data.model import OAuth2Credentials
    from backend.integrations.creds_manager import IntegrationCredentialsManager
    from backend.integrations.providers import ProviderName

    try:
        mgr = IntegrationCredentialsManager()
        mcp_creds = await mgr.store.get_creds_by_provider(
            user_id, ProviderName.MCP.value
        )
        # Collect all matching credentials and pick the best one.
        # Primary sort: latest access_token_expires_at (tokens with expiry
        # are preferred over non-expiring ones).  Secondary sort: last in
        # iteration order, which corresponds to the most recently created
        # row — this acts as a tiebreaker when multiple bearer tokens have
        # no expiry (e.g. after a failed old-credential cleanup).
        best: OAuth2Credentials | None = None
        for cred in mcp_creds:
            if (
                isinstance(cred, OAuth2Credentials)
                and (cred.metadata or {}).get("mcp_server_url") == server_url
            ):
                if best is None or (
                    (cred.access_token_expires_at or 0)
                    >= (best.access_token_expires_at or 0)
                ):
                    best = cred
        if best:
            best = await mgr.refresh_if_needed(user_id, best)
            logger.info("Auto-resolved MCP credential %s for %s", best.id, server_url)
        return best
    except Exception:
        logger.warning("Auto-lookup MCP credential failed", exc_info=True)
        return None
