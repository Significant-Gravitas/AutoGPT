"""
Stripe Link CLI — Credential definitions for AutoGPT blocks.

Link CLI uses OAuth 2.0 Device Code Grant (RFC 8628), which produces standard
access_token + refresh_token pairs stored as OAuth2Credentials. The device-code
acquisition flow is handled by ``StripeLinkDeviceAuthHandler`` in
``backend/integrations/oauth/stripe_link.py``.
"""

from typing import Literal

from pydantic import SecretStr

from backend.data.model import CredentialsField, CredentialsMetaInput, OAuth2Credentials
from backend.integrations.providers import ProviderName

LINK_API_BASE_URL = "https://api.link.com"

# Every Link call is awaited inline by a block or an API handler, so an
# upstream that accepts the connection and then stalls would hold a worker for
# as long as it likes. Bound them all.
LINK_HTTP_TIMEOUT = 15.0
LINK_DEFAULT_SCOPES = ["userinfo:read", "payment_methods.agentic"]

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
