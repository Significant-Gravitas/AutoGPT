"""Approval token helpers for high-cost CoPilot requests."""

import base64
import binascii
import hashlib
import hmac
import json
import secrets
import time
from typing import Literal

from backend.util.settings import Settings

CopilotMode = Literal["fast", "extended_thinking"]
CopilotLlmModel = Literal["standard", "advanced"]

settings = Settings()
_ephemeral_secret = secrets.token_bytes(32)


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
) -> str:
    """Build a stable hash for cost-approval token binding."""
    payload = {
        "message": message,
        "is_user_message": is_user_message,
        "context": context,
        "file_ids": file_ids or [],
        "mode": mode,
        "model": model,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _approval_secret() -> bytes:
    candidates = (
        settings.secrets.encryption_key,
        settings.secrets.unsubscribe_secret_key,
        settings.secrets.supabase_service_role_key,
    )
    for value in candidates:
        if value:
            return value.encode("utf-8")
    return _ephemeral_secret


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
