"""Shared MCP helpers used by blocks, copilot tools, and API routes."""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import urlparse

from backend.data.model import OAuth2Credentials
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.integrations.providers import ProviderName

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


def is_manual_mcp_credential(credentials: OAuth2Credentials) -> bool:
    """Whether an MCP credential was pasted by the user rather than obtained via OAuth.

    Manual credentials are stored as ``OAuth2Credentials`` for compatibility
    with the existing credential plumbing, but they carry no refresh token and
    none of the OAuth client metadata the refresh path needs.  Callers use this
    to avoid treating the two kinds as interchangeable — rotating a manual
    credential in place is safe, doing the same to an OAuth row would discard
    its refresh token and client registration.
    """
    metadata = credentials.metadata or {}
    return (
        credentials.refresh_token is None
        and not metadata.get("mcp_token_url")
        and not metadata.get("mcp_client_id")
    )


def mcp_authorization_header(credentials: OAuth2Credentials) -> str:
    """Build the Authorization value to send for a *stored* MCP credential.

    Deliberately never inspects the secret.  Credentials stored since Basic
    support landed carry ``mcp_auth_scheme`` and hold an already-canonical
    ``"<Scheme> <credential>"`` in ``access_token``; anything older is a raw
    bare token that was always sent as Bearer.  The metadata says which, so a
    multi-word secret cannot be re-read as a scheme word plus a remainder --
    the failure that turned a stored ``"Bearer orgid api-key"`` into
    ``"Bearer Bearer orgid api-key"`` when it was normalized a second time.
    """
    token = credentials.access_token.get_secret_value()
    if (credentials.metadata or {}).get("mcp_auth_scheme"):
        return token
    return f"Bearer {token}"


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

    Called wherever a stored credential turns out to be unusable: the copilot's
    ``run_mcp_tool`` probe and the discovery route on a 401/403, and the MCP
    block when the stored value cannot be sent as an Authorization header at
    all — meaning the token was revoked or expired server-side without our local
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
) -> OAuth2Credentials | None:
    """Look up the best stored MCP credential for *server_url*.

    The caller should pass a **normalized** URL (via :func:`normalize_mcp_url`)
    so the comparison with ``mcp_server_url`` in credential metadata matches.

    Returns the credential with the latest ``access_token_expires_at``, refreshed
    if it can expire and needs it, or ``None`` when no match is found.
    """
    try:
        mgr = IntegrationCredentialsManager()
        mcp_creds = await mgr.store.get_creds_by_provider(
            user_id, ProviderName.MCP.value
        )

        # Collect all matching credentials and pick the best one.
        #
        # Primary sort: a manually pasted credential outranks an OAuth row.
        # A manual credential only exists because the user explicitly pasted
        # one for this server, and it never carries an expiry — ranking by
        # `access_token_expires_at or 0` alone made it lose to *any* surviving
        # OAuth row, so a stale grant kept being sent while all three UIs
        # reported "Connected" from a probe of the credential the user had
        # just entered.
        #
        # Secondary sort: latest access_token_expires_at (tokens with expiry
        # are preferred over non-expiring ones).  Tertiary: last in iteration
        # order, a tiebreaker when several rows compare equal (e.g. after a
        # failed old-credential cleanup).
        def rank(cred: OAuth2Credentials) -> tuple[int, float]:
            return (
                1 if is_manual_mcp_credential(cred) else 0,
                cred.access_token_expires_at or 0,
            )

        best: OAuth2Credentials | None = None
        for cred in mcp_creds:
            if (
                isinstance(cred, OAuth2Credentials)
                and (cred.metadata or {}).get("mcp_server_url") == server_url
            ):
                if best is None or rank(cred) >= rank(best):
                    best = cred
        # Manually entered MCP credentials are represented as OAuth2Credentials
        # for compatibility with the existing credential plumbing, but they have
        # no OAuth token endpoint. Trying to refresh one would construct an MCP
        # OAuth handler and reject the otherwise valid token.  Ask that question
        # directly rather than through expiry, which is only a proxy for it: an
        # OAuth credential whose token endpoint omitted ``expires_in`` has no
        # expiry either, and would never be refreshed again.
        if best and not is_manual_mcp_credential(best):
            best = await mgr.refresh_if_needed(user_id, best)
        if best:
            logger.info("Auto-resolved MCP credential %s for %s", best.id, server_url)
        return best
    except Exception:
        logger.warning("Auto-lookup MCP credential failed", exc_info=True)
        return None
