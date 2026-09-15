"""ChatGPT device-code authentication over plain HTTP.

Replaces driving the Codex CLI for sign-in. Every endpoint and response shape
below was verified against ``auth.openai.com`` rather than inferred from
another client's source.

This flow is *not* RFC 8628 despite resembling it, which is why the generic
device-auth helpers do not fit:

- the poll is keyed by ``(device_auth_id, user_code)``, not one ``device_code``;
- ``interval`` arrives as a **string**, and the deadline as an ISO timestamp in
  ``expires_at`` rather than an ``expires_in`` count of seconds;
- there is no ``verification_uri`` in the payload -- it is a fixed URL;
- a successful poll returns an ``authorization_code`` that still has to be
  exchanged at the ordinary OAuth token endpoint.

The client id is OpenAI's public Codex client. It is an identifier, not a
secret: no ``client_secret`` is ever sent, and PKCE is what protects the
exchange. ``originator`` identifies us honestly as AutoGPT rather than
impersonating the Codex CLI, which the backend also accepts.
"""

import logging
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, SecretStr, field_validator

from backend.integrations.codex.auth_bundle import (
    CodexAuthBundleV1,
    CodexAuthTokensV1,
    decode_jwt_claims,
)
from backend.util.request import Requests

logger = logging.getLogger(__name__)

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTH_BASE = "https://auth.openai.com"
DEVICE_CODE_URL = f"{AUTH_BASE}/api/accounts/deviceauth/usercode"
DEVICE_TOKEN_URL = f"{AUTH_BASE}/api/accounts/deviceauth/token"
OAUTH_TOKEN_URL = f"{AUTH_BASE}/oauth/token"
DEVICE_VERIFY_URL = f"{AUTH_BASE}/codex/device"
DEVICE_REDIRECT_URI = f"{AUTH_BASE}/deviceauth/callback"

# Identifies AutoGPT to OpenAI. Deliberately not "codex_cli_rs": the backend
# serves any originator, so sending theirs would be impersonation for no gain.
ORIGINATOR = "autogpt"
USER_AGENT = "autogpt/1.0 (+https://agpt.co)"

# Recorded on the bundle in place of a CLI version now that no binary is involved.
RUNTIME_VERSION = "http"

_DEFAULT_POLL_INTERVAL = 5
_MAX_POLL_INTERVAL = 60
_AUTH_RETRY_ATTEMPTS = 3

# Classify on the machine-readable ``code``, never the HTTP status: a pending
# authorization answers 403 and an unknown or expired one answers 404, so
# switching on status alone would either poll a dead login until it times out
# or abort a live one the user is still approving.
_PENDING_CODES = frozenset({"deviceauth_authorization_pending"})
_SLOW_DOWN_CODES = frozenset({"deviceauth_slow_down", "slow_down"})
_DENIED_CODES = frozenset({"deviceauth_access_denied", "access_denied"})
_EXPIRED_CODES = frozenset(
    {"deviceauth_not_found", "deviceauth_expired_token", "expired_token"}
)


class CodexAuthError(RuntimeError):
    """Sign-in failed in a way that repeating the same poll will not fix."""


class ChatGPTDeviceCode(BaseModel):
    model_config = ConfigDict(extra="ignore")

    device_auth_id: str
    user_code: str
    interval: int = _DEFAULT_POLL_INTERVAL
    expires_at: datetime | None = None

    @field_validator("interval", mode="before")
    @classmethod
    def _coerce_interval(cls, value: object) -> int:
        """Arrives as the string "5"; a junk value must not become a busy loop."""
        try:
            seconds = int(float(str(value)))
        except (TypeError, ValueError):
            return _DEFAULT_POLL_INTERVAL
        if seconds <= 0:
            return _DEFAULT_POLL_INTERVAL
        return min(seconds, _MAX_POLL_INTERVAL)

    @property
    def verification_url(self) -> str:
        return DEVICE_VERIFY_URL

    def seconds_remaining(self, *, now: datetime | None = None) -> float | None:
        if self.expires_at is None:
            return None
        return (self.expires_at - (now or datetime.now(timezone.utc))).total_seconds()


class ChatGPTPollResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["pending", "slow_down", "approved", "denied", "expired"]
    authorization_code: str | None = None
    code_verifier: str | None = None


class ChatGPTTokens(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id_token: SecretStr
    access_token: SecretStr
    refresh_token: SecretStr


def _headers(*, content_type: str = "application/json") -> dict[str, str]:
    return {
        "content-type": content_type,
        "accept": "application/json",
        "originator": ORIGINATOR,
        "user-agent": USER_AGENT,
    }


def _error_code(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if isinstance(error, dict):
        code = error.get("code") or error.get("type")
        return str(code) if code else None
    if isinstance(error, str):
        return error
    return None


async def request_device_code() -> ChatGPTDeviceCode:
    """Start a sign-in and return the code the user enters at the verify URL."""
    response = await Requests(retry_max_attempts=_AUTH_RETRY_ATTEMPTS).post(
        DEVICE_CODE_URL, headers=_headers(), json={"client_id": CLIENT_ID}
    )
    return ChatGPTDeviceCode.model_validate(response.json())


async def poll_device_code(device_auth_id: str, user_code: str) -> ChatGPTPollResult:
    """Poll once. The caller owns the loop, the sleeping, and the deadline."""
    # This function is already one iteration of the caller-owned polling
    # loop. Return a provider 429/5xx promptly so that loop can apply the
    # advertised interval instead of nesting another retry loop inside it.
    response = await Requests(raise_for_status=False, retry_max_attempts=1).post(
        DEVICE_TOKEN_URL,
        headers=_headers(),
        json={
            "client_id": CLIENT_ID,
            "device_auth_id": device_auth_id,
            "user_code": user_code,
        },
    )

    try:
        payload = response.json()
    except ValueError:
        payload = None

    if response.status == 200 and isinstance(payload, dict):
        authorization_code = payload.get("authorization_code")
        code_verifier = payload.get("code_verifier")
        if not authorization_code or not code_verifier:
            raise CodexAuthError(
                "ChatGPT approved the device but returned no authorization code"
            )
        return ChatGPTPollResult(
            status="approved",
            authorization_code=str(authorization_code),
            code_verifier=str(code_verifier),
        )

    code = _error_code(payload)
    if code in _PENDING_CODES:
        return ChatGPTPollResult(status="pending")
    if code in _SLOW_DOWN_CODES:
        return ChatGPTPollResult(status="slow_down")
    if code in _DENIED_CODES:
        return ChatGPTPollResult(status="denied")
    if code in _EXPIRED_CODES:
        return ChatGPTPollResult(status="expired")

    # An unrecognised 429 is still worth backing off on rather than declaring
    # the login dead.
    if response.status == 429:
        return ChatGPTPollResult(status="slow_down")

    raise CodexAuthError(
        f"ChatGPT device authorization failed (HTTP {response.status}, code={code})"
    )


async def exchange_authorization_code(
    authorization_code: str, code_verifier: str
) -> ChatGPTTokens:
    response = await Requests(
        raise_for_status=False, retry_max_attempts=_AUTH_RETRY_ATTEMPTS
    ).post(
        OAUTH_TOKEN_URL,
        headers=_headers(content_type="application/x-www-form-urlencoded"),
        data={
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "code": authorization_code,
            "code_verifier": code_verifier,
            "redirect_uri": DEVICE_REDIRECT_URI,
        },
    )
    if response.status != 200:
        try:
            payload = response.json()
        except ValueError:
            payload = None
        raise CodexAuthError(
            "ChatGPT token exchange failed "
            f"(HTTP {response.status}, code={_error_code(payload)})"
        )
    return ChatGPTTokens.model_validate(response.json())


async def refresh_access_token(
    refresh_token: SecretStr,
    *,
    current_id_token: SecretStr | None = None,
) -> ChatGPTTokens:
    """Exchange a refresh token. OpenAI rotates it, so persist what comes back."""
    response = await Requests(
        raise_for_status=False, retry_max_attempts=_AUTH_RETRY_ATTEMPTS
    ).post(
        OAUTH_TOKEN_URL,
        headers=_headers(content_type="application/x-www-form-urlencoded"),
        data={
            "grant_type": "refresh_token",
            "client_id": CLIENT_ID,
            "refresh_token": refresh_token.get_secret_value(),
        },
    )
    if response.status != 200:
        raise CodexAuthError(
            f"ChatGPT refused to refresh the credential (HTTP {response.status})"
        )
    payload = response.json()
    # A refresh that omits a new refresh token means the existing one stands.
    if not payload.get("refresh_token"):
        payload["refresh_token"] = refresh_token.get_secret_value()
    # OIDC refresh responses are allowed to omit a new ID token. Keep the
    # current one in that case; it carries the ChatGPT account identity needed
    # to rebuild the persisted auth bundle.
    if not payload.get("id_token"):
        if current_id_token is None:
            raise CodexAuthError("ChatGPT refresh response omitted the ID token")
        payload["id_token"] = current_id_token.get_secret_value()
    return ChatGPTTokens.model_validate(payload)


def bundle_from_tokens(tokens: ChatGPTTokens) -> CodexAuthBundleV1:
    account_id = decode_jwt_claims(tokens.id_token).chatgpt_account_id
    return CodexAuthBundleV1(
        tokens=CodexAuthTokensV1(
            id_token=tokens.id_token,
            access_token=tokens.access_token,
            refresh_token=tokens.refresh_token,
            account_id=account_id,
        ),
        last_refresh=datetime.now(timezone.utc),
        codex_runtime_version=RUNTIME_VERSION,
    )
