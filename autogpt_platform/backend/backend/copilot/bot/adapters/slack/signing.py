"""Slack request signature verification (HMAC-SHA256)."""

from slack_sdk.signature import SignatureVerifier

from . import config


def verify(body: bytes, timestamp: str, signature: str) -> bool:
    """Validate a Slack request against the configured signing secret.

    Fails closed on any malformed input (missing/empty headers, a non-numeric
    timestamp) rather than raising — a garbage request becomes a clean 401, not
    a 500/400 leaking an ``int('')`` error.
    """
    if not timestamp or not signature:
        return False
    try:
        return SignatureVerifier(config.get_signing_secret()).is_valid(
            body=body, timestamp=timestamp, signature=signature
        )
    except Exception:
        return False
