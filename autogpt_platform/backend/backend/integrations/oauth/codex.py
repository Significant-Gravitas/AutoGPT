"""Device-auth handler for ChatGPT (Codex) subscriptions.

Sign-in and refresh both run over plain HTTP against ``auth.openai.com``; no
Codex binary is involved. See ``backend.integrations.codex.chatgpt_auth`` for
the endpoint-level details and for why this provider is *not* RFC 8628.

Registering here is what puts Codex refreshes behind the credentials manager's
Redis mutex. OpenAI rotates the refresh token on every exchange, so two workers
replaying the same one can get the whole grant revoked -- hence
``ROTATES_REFRESH_TOKEN``.
"""

import logging
from typing import ClassVar

from backend.data.model import OAuth2Credentials
from backend.integrations.codex.auth_bundle import CodexAuthBundleError
from backend.integrations.codex.chatgpt_auth import (
    CodexAuthError,
    bundle_from_tokens,
    exchange_authorization_code,
    poll_device_code,
    refresh_access_token,
    request_device_code,
)
from backend.integrations.codex.credential_codec import (
    bundle_from_credentials,
    checkpoint_credentials_from_bundle,
    credentials_from_bundle,
)
from backend.integrations.oauth.device_base import (
    BaseDeviceAuthHandler,
    DeviceAuthInitiation,
    DeviceAuthPollResult,
)
from backend.integrations.providers import ProviderName

logger = logging.getLogger(__name__)

# ChatGPT's poll is keyed by a *pair*, while the base contract carries a single
# opaque ``device_code`` string. That is fine here rather than a smuggle: the
# router stores this value server-side and passes it back verbatim without ever
# interpreting it or exposing it to the browser. The separator cannot collide --
# device auth ids are `deviceauth_<hex>` and user codes are `XXXX-XXXXX`.
_KEY_SEPARATOR = "|"


def _pack_poll_key(device_auth_id: str, user_code: str) -> str:
    return f"{device_auth_id}{_KEY_SEPARATOR}{user_code}"


def _unpack_poll_key(device_code: str) -> tuple[str, str]:
    device_auth_id, separator, user_code = device_code.partition(_KEY_SEPARATOR)
    if not separator or not device_auth_id or not user_code:
        raise CodexAuthError("ChatGPT device poll key is malformed")
    return device_auth_id, user_code


class CodexDeviceAuthHandler(BaseDeviceAuthHandler):
    PROVIDER_NAME: ClassVar[ProviderName | str] = ProviderName.CODEX

    # OpenAI returns a new refresh token on every exchange and treats replay of
    # the previous one as compromise, so refreshes must be serialized.
    ROTATES_REFRESH_TOKEN: ClassVar[bool] = True

    # Scope is fixed by the provider; ChatGPT ignores anything we ask for here.
    DEFAULT_SCOPES: ClassVar[list[str]] = []

    async def initiate_device_auth(self, scopes: list[str]) -> DeviceAuthInitiation:
        device = await request_device_code()
        remaining = device.seconds_remaining()
        return DeviceAuthInitiation(
            device_code=_pack_poll_key(device.device_auth_id, device.user_code),
            user_code=device.user_code,
            verification_url=device.verification_url,
            # ChatGPT has no pre-filled variant: the code is typed by hand.
            verification_url_complete=None,
            expires_in=int(remaining) if remaining and remaining > 0 else 900,
            interval=device.interval,
        )

    async def poll_for_tokens(self, device_code: str) -> DeviceAuthPollResult:
        device_auth_id, user_code = _unpack_poll_key(device_code)
        result = await poll_device_code(device_auth_id, user_code)

        if result.status != "approved":
            return DeviceAuthPollResult(status=result.status)

        # An approved poll yields an authorization code, not tokens: ChatGPT
        # still requires the ordinary PKCE exchange to complete sign-in.
        assert result.authorization_code and result.code_verifier
        tokens = await exchange_authorization_code(
            result.authorization_code, result.code_verifier
        )
        try:
            credentials = credentials_from_bundle(bundle_from_tokens(tokens))
        except CodexAuthBundleError as exc:
            raise CodexAuthError("ChatGPT returned invalid token data") from exc
        return DeviceAuthPollResult(status="approved", credentials=credentials)

    async def _refresh_tokens(
        self, credentials: OAuth2Credentials
    ) -> OAuth2Credentials:
        if credentials.refresh_token is None:
            raise CodexAuthError("ChatGPT credential has no refresh token")
        current_id_token = None
        try:
            current_id_token = bundle_from_credentials(credentials).tokens.id_token
        except CodexAuthBundleError:
            # A fresh ID token can repair legacy or damaged provider state. If
            # the provider omits one too, refresh_access_token reports a normal
            # auth failure instead of leaking a Pydantic validation error.
            pass
        tokens = await refresh_access_token(
            credentials.refresh_token,
            current_id_token=current_id_token,
        )
        # Refresh in place so the credential keeps its id, title and any other
        # caller-owned fields; only the token material is replaced.
        try:
            bundle = bundle_from_tokens(tokens)
        except CodexAuthBundleError as exc:
            raise CodexAuthError("ChatGPT returned invalid token data") from exc
        return checkpoint_credentials_from_bundle(credentials, bundle)

    async def revoke_tokens(self, credentials: OAuth2Credentials) -> bool:
        # OpenAI publishes no revocation endpoint for this client. Returning
        # False tells the caller to drop the stored credential locally rather
        # than report a revocation that never happened.
        return False
