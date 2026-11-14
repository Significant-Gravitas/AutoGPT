"""
Stripe Link CLI — Credential definitions for AutoGPT blocks.

Link CLI uses OAuth 2.0 Device Code Grant (RFC 8628), which produces standard
access_token + refresh_token pairs stored as OAuth2Credentials. The device-code
acquisition flow is handled by ``StripeLinkDeviceAuthHandler`` in
``backend/integrations/oauth/stripe_link.py``.
"""

import logging
from typing import Any, Literal

import httpx
from pydantic import SecretStr

from backend.data.model import CredentialsField, CredentialsMetaInput, OAuth2Credentials

# Owned by the integrations layer; re-exported so blocks import from their own
# package. Every other arrow in this codebase runs blocks -> integrations, and
# the oauth package builds its registries at import, so importing integrations
# from here would drag every block in eagerly.
from backend.integrations.oauth.stripe_link import (  # noqa: E402
    LINK_API_BASE_URL,
    LINK_HTTP_TIMEOUT,
)
from backend.integrations.providers import ProviderName

LINK_DEFAULT_SCOPES = ["userinfo:read", "payment_methods.agentic"]

logger = logging.getLogger(__name__)

# Upstream error text reaches a block `error` output, which is persisted.
MAX_ERROR_DETAIL_CHARS = 500

StripeLinkCredentials = OAuth2Credentials

# `credentials_types` carries two different things at once: the *shape* of the
# stored credential and the *method* used to acquire it. For every other
# provider those coincide, but a device-code grant yields an ordinary OAuth2
# token pair — so the block has to accept `oauth2` (or saved credentials stop
# matching) while still advertising `device_code` so connect UIs offer the
# device flow instead of an authorization-code redirect the provider has no
# client secret for.
StripeLinkCredentialsInput = CredentialsMetaInput[
    Literal[ProviderName.STRIPE_LINK],  # type: ignore[index]
    Literal["oauth2", "device_code"],
]


def StripeLinkCredentialsField() -> StripeLinkCredentialsInput:
    """
    Creates a Stripe Link credentials input on a block.

    All Link blocks require the same `payment_methods.agentic` scope.
    """
    return CredentialsField(
        required_scopes=set(LINK_DEFAULT_SCOPES),
        description=(
            "Connect your Stripe Link account to enable the agent to request "
            "secure, one-time-use payment credentials from your Link wallet. "
            "You'll approve each spend request via the Link app."
        ),
    )


# ---------------------------------------------------------------------------
# Test credentials for block testing
# ---------------------------------------------------------------------------
TEST_CREDENTIALS = OAuth2Credentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="stripe_link",
    access_token=SecretStr("mock-link-access-token"),
    refresh_token=SecretStr("mock-link-refresh-token"),
    access_token_expires_at=None,
    scopes=LINK_DEFAULT_SCOPES,
    title="Mock Stripe Link credentials",
    username="test@example.com",
)

TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


async def link_api_request(
    credentials: StripeLinkCredentials,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Make an authenticated request to the Link API.

    Uses the access_token from OAuth2Credentials as a Bearer token.

    Refresh is deliberately not handled here: `IntegrationCredentialsManager`
    already refreshes on acquire (`_refresh_locked`), under a per-credential
    lock, and persists the rotated tokens. Refreshing inside the block would
    bypass both and let concurrent nodes stampede the token endpoint.
    """
    headers = {
        "Authorization": f"Bearer {credentials.access_token.get_secret_value()}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=LINK_HTTP_TIMEOUT) as client:
        response = await client.request(
            method=method,
            url=f"{LINK_API_BASE_URL}{path}",
            headers=headers,
            json=body,
        )
        # `is_success`, not `not is_error`: the latter is 4xx/5xx only, so a
        # 3xx would fall straight through to `.json()` — and redirects are not
        # followed. A moved endpoint would read as an empty wallet, or raise a
        # bare KeyError with the redirect invisible.
        if not response.is_success:
            # Link explains itself in a structured `error.message`; surface
            # that rather than a bare "400 Bad Request" with the explanation
            # discarded. That is how the SPT merchant-field constraint stayed
            # hidden during development.
            try:
                detail = response.json().get("error", {}).get("message")
            # ValueError: not JSON. AttributeError/TypeError: JSON, but not
            # the object shape we index into. Anything else is our bug, and
            # masking it as "API text" would hide it.
            except (ValueError, AttributeError, TypeError):
                detail = None

            if detail:
                # Bounded like the raw-body log below: this string becomes a
                # persisted block output, and a multi-KB message from Link or
                # an intercepting gateway has no business there either.
                raise RuntimeError(
                    f"Link API error ({response.status_code}): "
                    f"{str(detail)[:MAX_ERROR_DETAIL_CHARS]}"
                )

            # No usable message. The raw body goes to the logs rather than
            # into the exception, because that string becomes a block `error`
            # output — persisted with the execution and surfaced in agent
            # transcripts — and an arbitrary upstream body has no business
            # there.
            logger.error(
                "Link API %s %s failed: %s %s",
                method,
                path,
                response.status_code,
                response.text[:500],
            )
            raise RuntimeError(f"Link API error ({response.status_code})")

        return response.json()
