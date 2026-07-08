"""Slack request signature verification (HMAC-SHA256)."""

from slack_sdk.signature import SignatureVerifier

from . import config


def verify(body: bytes, timestamp: str, signature: str) -> bool:
    """Validate a Slack request against the configured signing secret."""
    return SignatureVerifier(config.get_signing_secret()).is_valid(
        body=body, timestamp=timestamp, signature=signature
    )
