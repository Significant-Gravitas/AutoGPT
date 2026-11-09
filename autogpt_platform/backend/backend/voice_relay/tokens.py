"""Signed call start tokens and resume tickets.

A token is ``base64url(claims).base64url(hmac_sha256(key, base64url(claims)))``
under ``VOICE_RELAY_SIGNING_KEY``. Only the REST API and the relay hold the
key, so the relay trusts the user a token names without ever seeing the
platform session:

- a start token, minted by the REST API's start route, names one user, one
  call and its audio leg, lives ``TOKEN_TTL_S``, and is spent by the first
  socket that presents it (:func:`spend`);
- a resume ticket, which the relay hands the browser in ``ready``, lets the
  same user re-attach a new socket to the same live call until the call cap
  has passed.

A token is checked whole before any of it is believed: the signature first,
in constant time, then its kind, its claims and its expiry.
"""

import base64
import binascii
import hashlib
import hmac
import json
import re
import time
import uuid
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict

from backend.data.redis_client import AsyncRedisClient
from backend.util.settings import Settings
from backend.voice_relay.config import TOKEN_SPENT_TTL_S, TOKEN_TTL_S

settings = Settings()

# The legs a call's audio can take: LiveKit in front of the relay, or the
# control socket itself.
Leg = Literal["livekit", "websocket"]
LEGS: tuple[Leg, ...] = ("livekit", "websocket")

# A shorter key is refused: it would not hold up as an HMAC key.
MIN_KEY_CHARS = 32
# Anything longer is not a token this module minted.
MAX_TOKEN_CHARS = 1_024
MAX_USER_ID_CHARS = 128
CALL_ID = re.compile(r"[0-9a-f]{32}")

_START = "start"
_RESUME = "resume"


class TokenError(Exception):
    """A token the relay refuses. The message is a short code, safe to send."""


class StartClaims(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: str
    call_id: str
    leg: Leg
    expires_at: float


class ResumeClaims(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: str
    call_id: str
    expires_at: float


def signing_key() -> bytes | None:
    """The configured signing key, or None while it is blank or too short."""
    key = settings.secrets.voice_relay_signing_key
    if len(key) < MIN_KEY_CHARS:
        return None
    return key.encode()


def is_configured() -> bool:
    return signing_key() is not None


def new_call_id() -> str:
    return uuid.uuid4().hex


def mint_start(
    user_id: str,
    call_id: str,
    leg: Leg,
    *,
    ttl_s: float = TOKEN_TTL_S,
    now: float | None = None,
) -> str:
    """A start token for *user_id*'s call *call_id* on *leg*."""
    issued = time.time() if now is None else now
    return _sign(
        {
            "kind": _START,
            "uid": user_id,
            "call_id": call_id,
            "leg": leg,
            "exp": issued + ttl_s,
        }
    )


def mint_resume(
    user_id: str, call_id: str, ttl_s: float, *, now: float | None = None
) -> str:
    """A ticket that re-attaches *user_id* to their live call *call_id*."""
    issued = time.time() if now is None else now
    return _sign(
        {"kind": _RESUME, "uid": user_id, "call_id": call_id, "exp": issued + ttl_s}
    )


def verify_start(token: object, *, now: float | None = None) -> StartClaims:
    """The claims of a valid, unexpired start token. Raises :class:`TokenError`."""
    claims = _verify(token, _START, now)
    leg = claims.get("leg")
    if leg not in LEGS:
        raise TokenError("token_invalid")
    return StartClaims(
        user_id=claims["uid"],
        call_id=claims["call_id"],
        leg=cast(Leg, leg),
        expires_at=claims["exp"],
    )


def verify_resume(token: object, *, now: float | None = None) -> ResumeClaims:
    """The claims of a valid, unexpired resume ticket. Raises :class:`TokenError`."""
    claims = _verify(token, _RESUME, now)
    return ResumeClaims(
        user_id=claims["uid"], call_id=claims["call_id"], expires_at=claims["exp"]
    )


def spent_key(call_id: str) -> str:
    return f"voice:tok:{call_id}"


async def unspend(redis: AsyncRedisClient, claims: StartClaims) -> None:
    """Make a spent start token good again, for a start refused with "try
    again" (1013), so the browser can retry with the same token."""
    await redis.delete(spent_key(claims.call_id))


async def spend(redis: AsyncRedisClient, claims: StartClaims) -> None:
    """Spend a start token. Raises :class:`TokenError` when it was spent already."""
    fresh = await redis.set(
        spent_key(claims.call_id), "1", nx=True, ex=TOKEN_SPENT_TTL_S
    )
    if not fresh:
        raise TokenError("token_used")


def _sign(claims: dict[str, object]) -> str:
    key = signing_key()
    if key is None:
        raise TokenError("not_configured")
    body = _b64(json.dumps(claims, separators=(",", ":")).encode())
    return f"{body}.{_mac(key, body)}"


def _verify(token: object, kind: str, now: float | None) -> dict:
    key = signing_key()
    if key is None:
        raise TokenError("not_configured")
    if not isinstance(token, str) or len(token) > MAX_TOKEN_CHARS:
        raise TokenError("token_invalid")
    body, dot, signature = token.partition(".")
    # As bytes: compare_digest refuses a str with anything but ASCII in it.
    if not dot or not hmac.compare_digest(_mac(key, body).encode(), signature.encode()):
        raise TokenError("token_invalid")
    try:
        claims = json.loads(_unb64(body))
    except (ValueError, binascii.Error):
        raise TokenError("token_invalid") from None
    if not isinstance(claims, dict) or claims.get("kind") != kind:
        raise TokenError("token_invalid")
    user_id, call_id, expires_at = (
        claims.get("uid"),
        claims.get("call_id"),
        claims.get("exp"),
    )
    if (
        not isinstance(user_id, str)
        or not 0 < len(user_id) <= MAX_USER_ID_CHARS
        or not isinstance(call_id, str)
        or not CALL_ID.fullmatch(call_id)
        or not isinstance(expires_at, (int, float))
        or isinstance(expires_at, bool)
    ):
        raise TokenError("token_invalid")
    if expires_at < (time.time() if now is None else now):
        raise TokenError("token_expired")
    return claims


def _mac(key: bytes, body: str) -> str:
    return _b64(hmac.new(key, body.encode(), hashlib.sha256).digest())


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
