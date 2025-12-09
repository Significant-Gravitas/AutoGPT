"""
Webhook Notification System for External API.

Sends webhook notifications to external applications for execution events.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import weakref
from datetime import datetime, timezone
from typing import Any, Coroutine, Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

# Webhook delivery settings
WEBHOOK_TIMEOUT_SECONDS = 30
WEBHOOK_MAX_RETRIES = 3
WEBHOOK_RETRY_DELAYS = [5, 30, 300]  # seconds: 5s, 30s, 5min


class WebhookDeliveryError(Exception):
    """Raised when webhook delivery fails."""

    pass


def sign_webhook_payload(payload: dict[str, Any], secret: str) -> str:
    """
    Create HMAC-SHA256 signature for webhook payload.

    Args:
        payload: The webhook payload to sign
        secret: The webhook secret key

    Returns:
        Hex-encoded HMAC-SHA256 signature
    """
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    signature = hmac.new(
        secret.encode(),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()
    return signature


def verify_webhook_signature(
    payload: dict[str, Any],
    signature: str,
    secret: str,
) -> bool:
    """
    Verify a webhook signature.

    Args:
        payload: The webhook payload
        signature: The signature to verify
        secret: The webhook secret key

    Returns:
        True if signature is valid
    """
    expected = sign_webhook_payload(payload, secret)
    return hmac.compare_digest(expected, signature)


def validate_webhook_url(url: str, allowed_domains: list[str]) -> bool:
    """
    Validate that a webhook URL is allowed.

    Args:
        url: The webhook URL to validate
        allowed_domains: List of allowed domains (from OAuth client config)

    Returns:
        True if URL is valid and allowed
    """
    from backend.util.url import hostname_matches_any_domain

    try:
        parsed = urlparse(url)

        # Must be HTTPS (except for localhost in development)
        if parsed.scheme != "https":
            if not (
                parsed.scheme == "http"
                and parsed.hostname in ["localhost", "127.0.0.1"]
            ):
                return False

        # Must have a host
        if not parsed.hostname:
            return False

        # Check against allowed domains
        return hostname_matches_any_domain(parsed.hostname, allowed_domains)

    except Exception:
        return False


async def send_webhook(
    url: str,
    payload: dict[str, Any],
    secret: Optional[str] = None,
    timeout: int = WEBHOOK_TIMEOUT_SECONDS,
) -> bool:
    """
    Send a webhook notification.

    Args:
        url: Webhook URL
        payload: Payload to send
        secret: Optional secret for signature
        timeout: Request timeout in seconds

    Returns:
        True if webhook was delivered successfully
    """
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "AutoGPT-Webhook/1.0",
        "X-Webhook-Timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if secret:
        signature = sign_webhook_payload(payload, secret)
        headers["X-Webhook-Signature"] = f"sha256={signature}"

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                json=payload,
                headers=headers,
            )

            if response.status_code >= 200 and response.status_code < 300:
                logger.debug(f"Webhook delivered successfully to {url}")
                return True
            else:
                logger.warning(
                    f"Webhook delivery failed: {url} returned {response.status_code}"
                )
                return False

    except httpx.TimeoutException:
        logger.warning(f"Webhook delivery timed out: {url}")
        return False
    except Exception as e:
        logger.error(f"Webhook delivery error: {url} - {str(e)}")
        return False


async def send_webhook_with_retry(
    url: str,
    payload: dict[str, Any],
    secret: Optional[str] = None,
    max_retries: int = WEBHOOK_MAX_RETRIES,
) -> bool:
    """
    Send a webhook with automatic retries.

    Args:
        url: Webhook URL
        payload: Payload to send
        secret: Optional secret for signature
        max_retries: Maximum number of retry attempts

    Returns:
        True if webhook was eventually delivered successfully
    """
    for attempt in range(max_retries + 1):
        if await send_webhook(url, payload, secret):
            return True

        if attempt < max_retries:
            delay = WEBHOOK_RETRY_DELAYS[min(attempt, len(WEBHOOK_RETRY_DELAYS) - 1)]
            logger.info(
                f"Webhook delivery failed, retrying in {delay}s (attempt {attempt + 1})"
            )
            await asyncio.sleep(delay)

    logger.error(f"Webhook delivery failed after {max_retries} retries: {url}")
    return False


# Track pending webhook tasks to prevent garbage collection
# Using WeakSet so tasks are automatically removed when they complete and are dereferenced
_pending_webhook_tasks: weakref.WeakSet[asyncio.Task[Any]] = weakref.WeakSet()


def _create_tracked_task(coro: Coroutine[Any, Any, bool]) -> asyncio.Task[bool]:
    """Create a task that is tracked to prevent garbage collection."""
    task = asyncio.create_task(coro)
    _pending_webhook_tasks.add(task)
    # No explicit done callback needed - WeakSet automatically removes
    # references when tasks are garbage collected after completion
    return task


class WebhookNotifier:
    """
    Service for sending webhook notifications to external applications.
    """

    def __init__(self):
        pass

    async def notify_execution_started(
        self,
        execution_id: str,
        agent_id: str,
        client_id: str,
        webhook_url: str,
        webhook_secret: Optional[str] = None,
    ) -> None:
        """
        Notify external app that an execution has started.
        """
        payload = {
            "event": "execution.started",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "execution_id": execution_id,
                "agent_id": agent_id,
                "status": "running",
            },
        }

        _create_tracked_task(
            send_webhook_with_retry(webhook_url, payload, webhook_secret)
        )

    async def notify_execution_completed(
        self,
        execution_id: str,
        agent_id: str,
        client_id: str,
        webhook_url: str,
        outputs: dict[str, Any],
        webhook_secret: Optional[str] = None,
    ) -> None:
        """
        Notify external app that an execution has completed successfully.
        """
        payload = {
            "event": "execution.completed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "execution_id": execution_id,
                "agent_id": agent_id,
                "status": "completed",
                "outputs": outputs,
            },
        }

        _create_tracked_task(
            send_webhook_with_retry(webhook_url, payload, webhook_secret)
        )

    async def notify_execution_failed(
        self,
        execution_id: str,
        agent_id: str,
        client_id: str,
        webhook_url: str,
        error: str,
        webhook_secret: Optional[str] = None,
    ) -> None:
        """
        Notify external app that an execution has failed.
        """
        payload = {
            "event": "execution.failed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "execution_id": execution_id,
                "agent_id": agent_id,
                "status": "failed",
                "error": error,
            },
        }

        _create_tracked_task(
            send_webhook_with_retry(webhook_url, payload, webhook_secret)
        )

    async def notify_grant_revoked(
        self,
        grant_id: str,
        credential_id: str,
        provider: str,
        client_id: str,
        webhook_url: str,
        webhook_secret: Optional[str] = None,
    ) -> None:
        """
        Notify external app that a credential grant has been revoked.
        """
        payload = {
            "event": "grant.revoked",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "grant_id": grant_id,
                "credential_id": credential_id,
                "provider": provider,
            },
        }

        _create_tracked_task(
            send_webhook_with_retry(webhook_url, payload, webhook_secret)
        )


# Module-level singleton
_webhook_notifier: Optional[WebhookNotifier] = None


def get_webhook_notifier() -> WebhookNotifier:
    """Get the singleton webhook notifier instance."""
    global _webhook_notifier
    if _webhook_notifier is None:
        _webhook_notifier = WebhookNotifier()
    return _webhook_notifier
