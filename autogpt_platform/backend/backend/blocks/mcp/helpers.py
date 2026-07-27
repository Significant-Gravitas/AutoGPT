"""Shared MCP helpers used by blocks, copilot tools, and API routes."""

from __future__ import annotations

import json
import logging
from typing import Any, TypeGuard
from urllib.parse import urlparse

from backend.data.model import APIKeyCredentials, Credentials, OAuth2Credentials
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.integrations.providers import ProviderName

logger = logging.getLogger(__name__)

# MCP credentials can be either a full OAuth2 token or a static API key /
# bearer token entered by the user. Both are sent to the server as
# ``Authorization: Bearer <token>`` (see ``MCPClient._build_headers``).
MCPCredential = OAuth2Credentials | APIKeyCredentials


def normalize_mcp_url(url: str) -> str:
    """Normalize an MCP server URL for consistent credential matching.

    Strips leading/trailing whitespace and a single trailing slash so that
    ``https://mcp.example.com/`` and ``https://mcp.example.com`` resolve to
    the same stored credential.
    """
    return url.strip().rstrip("/")


def mcp_auth_token(cred: MCPCredential) -> str:
    """Extract the bearer token string from an MCP credential.

    OAuth2 credentials carry it in ``access_token``; static API-key / bearer
    tokens carry it in ``api_key``. The MCP client adds the ``Bearer`` prefix.
    """
    if isinstance(cred, APIKeyCredentials):
        return cred.api_key.get_secret_value()
    return cred.access_token.get_secret_value()


def is_mcp_credential_for_server(
    cred: Credentials, server_url: str
) -> TypeGuard[MCPCredential]:
    """True if *cred* is an MCP credential stored for *server_url*.

    Matches both OAuth2 and API-key credentials so bearer-token and OAuth
    flows share the same lookup/cleanup logic. *server_url* should be
    normalized (via :func:`normalize_mcp_url`).
    """
    if not isinstance(cred, (OAuth2Credentials, APIKeyCredentials)):
        return False
    stored = (cred.metadata or {}).get("mcp_server_url", "")
    return normalize_mcp_url(stored) == server_url


def _mcp_credential_expiry(cred: MCPCredential) -> int:
    """Unix expiry timestamp of an MCP credential, or 0 if it never expires."""
    if isinstance(cred, APIKeyCredentials):
        return cred.expires_at or 0
    return cred.access_token_expires_at or 0


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


async def invalidate_mcp_credential(user_id: str, credential_id: str) -> None:
    """Delete a stored MCP credential that the server just rejected.

    Called from the copilot's ``run_mcp_tool`` path when an MCP server
    returns 401/403 with a credential we *do* have on file — meaning the
    token was revoked or expired server-side without our local
    ``access_token_expires_at`` knowing.  Removing the dead row prevents
    ``auto_lookup_mcp_credential`` from feeding the same stale token back
    on the next attempt and lets the user re-auth cleanly via the setup
    card.  Failures are swallowed (best-effort) — the worst case is a
    second loop through the same code path, which still surfaces the
    setup card.
    """
    try:
        mgr = IntegrationCredentialsManager()
        # Go through ``mgr.delete`` (not ``store.delete_creds_by_id``) so the
        # per-credential lock + ``_invoke_creds_changed_hook`` fire — the hook
        # evicts any cached provider token for the user.
        await mgr.delete(user_id, credential_id)
        logger.info("Invalidated stale MCP credential %s", credential_id)
    except ValueError:
        # ``mgr.delete`` raises ``ValueError`` when the credential is
        # already gone (e.g. the user deleted it manually in Settings
        # between the ``auto_lookup_mcp_credential`` call and now).  Not
        # a problem — the goal was "this row should not exist" and it
        # doesn't.  Demote to debug so we don't spam warnings on retries.
        logger.debug("MCP credential %s already gone during invalidate", credential_id)
    except Exception:
        logger.warning(
            "Failed to invalidate stale MCP credential %s",
            credential_id,
            exc_info=True,
        )


async def auto_lookup_mcp_credential(
    user_id: str, server_url: str
) -> MCPCredential | None:
    """Look up the best stored MCP credential for *server_url*.

    The caller should pass a **normalized** URL (via :func:`normalize_mcp_url`)
    so the comparison with ``mcp_server_url`` in credential metadata matches.

    Matches both OAuth2 credentials and static API-key / bearer tokens.
    Returns the credential with the latest expiry, refreshed if needed (OAuth2
    only), or ``None`` when no match is found.
    """
    try:
        mgr = IntegrationCredentialsManager()
        mcp_creds = await mgr.store.get_creds_by_provider(
            user_id, ProviderName.MCP.value
        )
        # Collect all matching credentials and pick the best one.
        # Primary sort: latest expiry (tokens with expiry are preferred over
        # non-expiring ones).  Secondary sort: last in iteration order, which
        # corresponds to the most recently created row — this acts as a
        # tiebreaker when multiple bearer tokens have no expiry (e.g. after a
        # failed old-credential cleanup).
        best: MCPCredential | None = None
        for cred in mcp_creds:
            if is_mcp_credential_for_server(cred, server_url):
                if best is None or (
                    _mcp_credential_expiry(cred) >= _mcp_credential_expiry(best)
                ):
                    best = cred
        if best is not None and isinstance(best, OAuth2Credentials):
            best = await mgr.refresh_if_needed(user_id, best)
        if best is not None:
            logger.info("Auto-resolved MCP credential %s for %s", best.id, server_url)
        return best
    except Exception:
        logger.warning("Auto-lookup MCP credential failed", exc_info=True)
        return None
