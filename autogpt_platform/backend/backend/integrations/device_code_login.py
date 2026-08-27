"""RFC 8628 device authorization grant, as a protocol client.

The flow a provider runs when a user signs in on a machine that cannot host
a browser redirect: ask for a code, show the user a short code and a URL,
then poll until they have approved it somewhere else.

Deliberately just the protocol. No Redis, no credential store, no route --
those are orchestration and they differ per provider (whether a login is
exclusive, where the tokens are written, what a cancel does). What does
*not* differ is RFC 8628 itself, and that is what lives here, so a second
provider adopting the flow supplies four endpoints rather than a second
implementation of the polling loop and its error codes.

The error codes are the part worth centralising. ``authorization_pending``
and ``slow_down`` are not failures -- they are the normal shape of a flow
where the user is still typing a code into their phone -- and a client that
treats them as errors abandons a sign-in that was going fine. That is
exactly the bug this module exists to not have written twice.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, SecretStr

from backend.util.request import Requests

logger = logging.getLogger(__name__)

# RFC 8628 section 3.5. Neither is a failure: the user is still approving.
_PENDING = "authorization_pending"
_SLOW_DOWN = "slow_down"
# How much to add to the interval when the server says we are too fast.
# RFC 8628 section 3.5 specifies 5 seconds.
_SLOW_DOWN_INCREMENT_SECONDS = 5
# Floor for the server-supplied interval, in case a provider omits it.
_DEFAULT_INTERVAL_SECONDS = 5


class DeviceCodeConfig(BaseModel):
    """Everything about one provider's device flow that is not the protocol."""

    model_config = ConfigDict(frozen=True)

    device_authorization_url: str
    token_url: str
    client_id: str
    scopes: tuple[str, ...] = ()
    # Some providers want the client secret on the token call; a public
    # client, which is the usual case for a device flow, has none.
    client_secret: SecretStr | None = None
    # Overall wall-clock ceiling. The server sends its own ``expires_in``;
    # this is the cap applied regardless, so a provider answering with an
    # implausible expiry cannot hold a request open indefinitely.
    max_wait_seconds: int = 15 * 60


class DeviceCodeGrant(BaseModel):
    """What the user needs in order to approve, and what we poll with."""

    model_config = ConfigDict(frozen=True)

    device_code: SecretStr
    # The short code the user types. Not secret in the same way -- it is
    # meant to be read off a screen -- but it is single-use, so it is still
    # not something to log.
    user_code: str
    verification_uri: str
    # The same URL with the code already in it, when the provider offers
    # one. Worth preferring: a user who follows it does not have to
    # transcribe anything, which is where this flow usually goes wrong.
    verification_uri_complete: str | None = None
    interval_seconds: int = _DEFAULT_INTERVAL_SECONDS
    expires_in_seconds: int | None = None


class DeviceCodeTokens(BaseModel):
    model_config = ConfigDict(frozen=True)

    access_token: SecretStr
    refresh_token: SecretStr | None = None
    # Absolute epoch seconds, resolved from the relative ``expires_in`` the
    # provider sends. Stored absolute because a relative lifetime is only
    # meaningful next to the moment it was issued, and that moment is not
    # carried anywhere once the credential is persisted.
    access_token_expires_at: int | None = None
    scopes: tuple[str, ...] = ()


class DeviceCodeError(Exception):
    """A device flow that ended without tokens.

    Carries the provider's own error code where there is one, because the
    difference between "the user declined" and "the code expired" is the
    difference between two very different things to say on screen.
    """

    def __init__(self, code: str, description: str | None = None) -> None:
        self.code = code
        self.description = description
        super().__init__(description or code)


class DeviceCodeExpired(DeviceCodeError):
    """The user did not approve in time."""


class DeviceCodeDenied(DeviceCodeError):
    """The user declined, or the provider refused the client."""


async def request_device_code(
    config: DeviceCodeConfig, requests: Requests | None = None
) -> DeviceCodeGrant:
    """Ask the provider for a code to show the user."""
    http = requests or Requests()
    body: dict[str, str] = {"client_id": config.client_id}
    if config.scopes:
        body["scope"] = " ".join(config.scopes)

    response = await http.post(
        config.device_authorization_url,
        data=body,
        headers={"Accept": "application/json"},
    )
    payload: dict[str, Any] = response.json()
    if "device_code" not in payload:
        raise DeviceCodeError(
            payload.get("error", "invalid_device_authorization_response"),
            payload.get("error_description"),
        )

    return DeviceCodeGrant(
        device_code=payload["device_code"],
        user_code=payload.get("user_code", ""),
        # Providers differ on the spelling: RFC 8628 says verification_uri,
        # several ship verification_url. Reading only one is a flow that
        # sends the user nowhere.
        verification_uri=payload.get("verification_uri")
        or payload.get("verification_url", ""),
        verification_uri_complete=payload.get("verification_uri_complete")
        or payload.get("verification_url_complete"),
        interval_seconds=int(payload.get("interval") or _DEFAULT_INTERVAL_SECONDS),
        expires_in_seconds=(
            int(payload["expires_in"]) if payload.get("expires_in") else None
        ),
    )


async def poll_for_tokens(
    config: DeviceCodeConfig,
    grant: DeviceCodeGrant,
    requests: Requests | None = None,
    sleep=asyncio.sleep,
    now=time.time,
) -> DeviceCodeTokens:
    """Poll until the user approves, declines, or the code expires.

    ``sleep`` and ``now`` are injected so the polling behaviour -- backing
    off on ``slow_down``, giving up at the deadline -- can be tested without
    a test that actually waits fifteen minutes.
    """
    http = requests or Requests()
    interval = max(grant.interval_seconds, 1)
    deadline = now() + min(
        config.max_wait_seconds,
        grant.expires_in_seconds or config.max_wait_seconds,
    )

    while True:
        if now() >= deadline:
            raise DeviceCodeExpired(
                "expired_token", "The sign-in code expired before it was approved."
            )

        await sleep(interval)
        payload = await _request_tokens(config, grant, http)
        error = payload.get("error")

        if error is None:
            return _tokens_from(payload, now=now)
        if error == _PENDING:
            continue
        if error == _SLOW_DOWN:
            # Not a failure. The provider is asking for a slower cadence and
            # RFC 8628 says to add five seconds and keep going -- a client
            # that gave up here would abandon a sign-in that was fine.
            interval += _SLOW_DOWN_INCREMENT_SECONDS
            continue
        if error == "expired_token":
            raise DeviceCodeExpired(error, payload.get("error_description"))
        if error in ("access_denied", "unauthorized_client"):
            raise DeviceCodeDenied(error, payload.get("error_description"))
        raise DeviceCodeError(error, payload.get("error_description"))


async def _request_tokens(
    config: DeviceCodeConfig, grant: DeviceCodeGrant, http: Requests
) -> dict[str, Any]:
    body = {
        "client_id": config.client_id,
        "device_code": grant.device_code.get_secret_value(),
        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
    }
    if config.client_secret is not None:
        body["client_secret"] = config.client_secret.get_secret_value()

    response = await http.post(
        config.token_url, data=body, headers={"Accept": "application/json"}
    )
    return response.json()


def _tokens_from(payload: dict[str, Any], now) -> DeviceCodeTokens:
    expires_in = payload.get("expires_in")
    scope = payload.get("scope") or ""
    return DeviceCodeTokens(
        access_token=payload["access_token"],
        refresh_token=payload.get("refresh_token"),
        access_token_expires_at=(int(now()) + int(expires_in)) if expires_in else None,
        scopes=tuple(scope.split()) if scope else (),
    )
