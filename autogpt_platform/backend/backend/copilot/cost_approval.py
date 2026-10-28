"""Approval token helpers for high-cost CoPilot requests."""

import base64
import binascii
import hashlib
import hmac
import json
import logging
import time
from typing import Literal

from backend.util.settings import Settings

logger = logging.getLogger(__name__)

CopilotMode = Literal["fast", "extended_thinking"]
CopilotLlmModel = Literal["standard", "advanced"]

# Cost estimation prices this many trailing messages. The approval token
# binds the full session length instead, so a message that arrives after
# the estimate invalidates the token even when this window is already full.
HISTORY_MESSAGE_WINDOW = 30

settings = Settings()
_warned_missing_signing_key = False
_DEV_FALLBACK_SECRET = b"autogpt-cost-approval-dev-fallback"


async def generate_cost_approval_token(
    *,
    user_id: str,
    session_id: str,
    message: str,
    is_user_message: bool,
    context: dict[str, str] | None,
    file_ids: list[str] | None,
    mode: CopilotMode | None,
    model: CopilotLlmModel | None,
    message_count: int,
    ttl_seconds: int,
) -> str:
    """Generate a signed approval token for high-cost estimates."""
    payload = {
        "u": user_id,
        "s": session_id,
        "h": build_cost_approval_fingerprint(
            message=message,
            is_user_message=is_user_message,
            context=context,
            file_ids=file_ids,
            mode=mode,
            model=model,
            message_count=message_count,
        ),
        "e": int(time.time()) + ttl_seconds,
    }
    encoded = _urlsafe_encode_json(payload)
    signature = _sign_payload(encoded)
    return f"{encoded}.{signature}"


async def validate_cost_approval_token(
    *,
    token: str,
    user_id: str,
    session_id: str,
    message: str,
    is_user_message: bool,
    context: dict[str, str] | None,
    file_ids: list[str] | None,
    mode: CopilotMode | None,
    model: CopilotLlmModel | None,
    message_count: int,
) -> bool:
    """Validate approval token signature, expiry, and request fingerprint."""
    try:
        encoded, signature = token.split(".", 1)
    except ValueError:
        return False
    expected_signature = _sign_payload(encoded)
    if not hmac.compare_digest(signature, expected_signature):
        return False
    try:
        payload = _decode_urlsafe_json(encoded)
    except (binascii.Error, json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False
    try:
        expires_at = int(payload.get("e", 0))
    except (TypeError, ValueError):
        return False
    if expires_at < int(time.time()):
        return False
    if payload.get("u") != user_id or payload.get("s") != session_id:
        return False
    expected_hash = build_cost_approval_fingerprint(
        message=message,
        is_user_message=is_user_message,
        context=context,
        file_ids=file_ids,
        mode=mode,
        model=model,
        message_count=message_count,
    )
    return payload.get("h") == expected_hash


def build_cost_approval_fingerprint(
    *,
    message: str,
    is_user_message: bool,
    context: dict[str, str] | None,
    file_ids: list[str] | None,
    mode: CopilotMode | None,
    model: CopilotLlmModel | None,
    message_count: int,
) -> str:
    """Bind the request and the session length at estimate time.

    Pricing uses the last ``HISTORY_MESSAGE_WINDOW`` messages. ``message_count``
    is the full session length, so a message that arrives before confirmation
    invalidates the approval even when that window is already full. The
    session id is bound separately on the token payload.
    """
    payload = {
        "message": message,
        "is_user_message": is_user_message,
        "context": context,
        "file_ids": file_ids or [],
        "mode": mode,
        "model": model,
        "message_count": message_count,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _approval_secret() -> bytes:
    """Return a signing key shared by every worker in the deployment.

    A dedicated key is preferred. When it is unset, the key is derived from
    ``encryption_key`` so workers do not each invent a secret at import time.
    """
    dedicated = settings.secrets.cost_approval_signing_key
    if dedicated:
        return dedicated.encode("utf-8")
    encryption_key = settings.secrets.encryption_key
    if encryption_key:
        return hashlib.sha256(
            b"autogpt-cost-approval-v1:" + encryption_key.encode("utf-8")
        ).digest()
    global _warned_missing_signing_key
    if not _warned_missing_signing_key:
        logger.warning(
            "[SECURITY] No cost_approval_signing_key or encryption_key configured: "
            "cost-approval tokens will use a shared development fallback. "
            "Set COST_APPROVAL_SIGNING_KEY for production deployments."
        )
        _warned_missing_signing_key = True
    return _DEV_FALLBACK_SECRET


def _sign_payload(encoded_payload: str) -> str:
    signature = hmac.new(
        _approval_secret(), encoded_payload.encode("utf-8"), hashlib.sha256
    )
    return signature.hexdigest()


def _urlsafe_encode_json(payload: dict[str, str | int]) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return base64.urlsafe_b64encode(body.encode("utf-8")).decode("ascii").rstrip("=")


def _decode_urlsafe_json(encoded_payload: str) -> object:
    padding = "=" * (-len(encoded_payload) % 4)
    decoded = base64.urlsafe_b64decode(encoded_payload + padding)
    return json.loads(decoded.decode("utf-8"))
