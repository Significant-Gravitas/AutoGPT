"""Sardis API client helpers used by the Sardis block suite."""

import asyncio
import hashlib
import logging
import uuid
from collections import OrderedDict
from typing import Any

from backend.blocks.sardis._auth import SardisCredentials
from backend.util.request import Requests, Response

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client cache — bounded LRU with async lock
# ---------------------------------------------------------------------------

_lock = asyncio.Lock()
_clients: OrderedDict[str, "SardisClient"] = OrderedDict()
_MAX_CLIENTS = 32


async def get_client(credentials: SardisCredentials) -> "SardisClient":
    """Return a cached client keyed on the API key hash (bounded LRU)."""
    cache_key = hashlib.sha256(
        credentials.api_key.get_secret_value().encode()
    ).hexdigest()[:16]
    async with _lock:
        if cache_key in _clients:
            _clients.move_to_end(cache_key)
            return _clients[cache_key]
        client = SardisClient(credentials)
        _clients[cache_key] = client
        if len(_clients) > _MAX_CLIENTS:
            _clients.popitem(last=False)
        return client


class SardisClient:
    """Client for the Sardis API."""

    API_URL = "https://api.sardis.sh/api/v2"

    def __init__(self, credentials: SardisCredentials):
        """Store Sardis credentials and configure tolerant HTTP handling.

        Two HTTP clients are used:
        * ``_requests_safe`` -- retries enabled (3 attempts) for idempotent
          operations (balance queries, policy checks).
        * ``_requests_no_retry`` -- **no** retries, used exclusively for
          ``send_payment`` to avoid duplicate financial transactions.

        The API key is kept behind ``SecretStr`` and extracted only at
        request time via ``_auth_headers`` to avoid long-lived plaintext
        copies in memory.
        """
        self._credentials = credentials
        self._requests_safe = Requests(
            raise_for_status=False,
            retry_max_attempts=3,
            extra_headers={"Content-Type": "application/json"},
        )
        self._requests_no_retry = Requests(
            raise_for_status=False,
            retry_max_attempts=1,
            extra_headers={"Content-Type": "application/json"},
        )

    def _auth_headers(self) -> dict[str, str]:
        """Return auth headers, extracting the secret at call time."""
        return {"X-API-Key": self._credentials.api_key.get_secret_value()}

    async def send_payment(
        self,
        wallet_id: str,
        to: str,
        amount: str,
        token: str = "USDC",
        chain: str = "base",
        purpose: str = "Payment",
        idempotency_key: str = "",
    ) -> dict[str, Any]:
        """Execute a policy-controlled payment.

        ``amount`` is a decimal *string* — never a float — so the exact value
        the caller entered reaches the API without IEEE 754 rounding.

        ``idempotency_key`` should be derived from the execution context
        (e.g. ``node_exec_id``) so that retrying the same logical payment
        does not create a duplicate charge.  Falls back to a random UUID
        when no key is provided (e.g. in tests).
        """
        key = idempotency_key or str(uuid.uuid4())
        response = await self._requests_no_retry.post(
            f"{self.API_URL}/wallets/{wallet_id}/transfer",
            headers={**self._auth_headers(), "Idempotency-Key": key},
            json={
                "destination": to,
                "amount": amount,
                "token": token,
                "chain": chain,
                "purpose": purpose,
            },
        )
        return self._normalize_response(
            response,
            default_error="Sardis payment request failed",
        )

    async def get_balance(
        self,
        wallet_id: str,
        token: str = "USDC",
    ) -> dict[str, Any]:
        """Get wallet balance."""
        response = await self._requests_safe.get(
            f"{self.API_URL}/wallets/{wallet_id}/balance",
            headers=self._auth_headers(),
            params={"token": token},
        )
        return self._normalize_response(
            response,
            default_error="Sardis balance request failed",
        )

    async def check_policy(
        self,
        wallet_id: str,
        amount: str,
        destination: str,
        token: str = "USDC",
    ) -> dict[str, Any]:
        """Check if a payment would be allowed by spending policy.

        ``amount`` is a decimal *string* to preserve precision.
        """
        response = await self._requests_safe.post(
            f"{self.API_URL}/policies/check",
            headers=self._auth_headers(),
            json={
                "wallet_id": wallet_id,
                "amount": amount,
                "destination": destination,
                "token": token,
            },
        )
        return self._normalize_response(
            response,
            default_error="Sardis policy check failed",
        )

    @staticmethod
    def _normalize_response(
        response: Response,
        *,
        default_error: str,
    ) -> dict[str, Any]:
        """Convert Sardis responses into a consistent dict error contract."""
        try:
            payload = response.json()
        except Exception:
            body = (response.text() or "")[:200]
            return {
                "error": f"{default_error}: HTTP {response.status} {body}",
                "status": response.status,
            }

        if response.ok:
            if isinstance(payload, dict):
                return payload
            return {
                "error": f"{default_error}: unexpected response body type {type(payload).__name__}",
                "status": response.status,
            }

        if isinstance(payload, dict):
            normalized_payload = dict(payload)
            normalized_payload.setdefault(
                "error",
                payload.get("error")
                or payload.get("message")
                or f"{default_error}: HTTP {response.status}",
            )
            normalized_payload.setdefault("status", response.status)
            return normalized_payload

        body = (response.text() or "")[:200]
        return {
            "error": f"{default_error}: HTTP {response.status} {body}",
            "status": response.status,
        }
