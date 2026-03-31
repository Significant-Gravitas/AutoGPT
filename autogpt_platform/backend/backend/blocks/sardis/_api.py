"""Sardis API client helpers used by the Sardis block suite."""

import logging
import uuid
from typing import Any

from backend.blocks.sardis._auth import SardisCredentials
from backend.util.request import Requests, Response

logger = logging.getLogger(__name__)


async def get_client(credentials: SardisCredentials) -> "SardisClient":
    """Create a fresh client per call, consistent with other block integrations."""
    return SardisClient(credentials)


class SardisClient:
    """Client for the Sardis API."""

    API_URL = "https://api.sardis.sh/api/v2"

    def __init__(self, credentials: SardisCredentials):
        """Configure tolerant HTTP handling with two retry strategies."""
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

        ``amount`` is a decimal *string* -- never a float -- so the exact value
        the caller entered reaches the API without IEEE 754 rounding.
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
        """Check if a payment would be allowed by spending policy."""
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
        """Convert Sardis responses into a consistent dict error contract.

        Error paths never leak raw server responses to the end user --
        full bodies are logged at debug level only.
        """
        try:
            payload = response.json()
        except Exception:
            body = (response.text() or "")[:200]
            logger.debug(
                "Non-JSON response from Sardis: HTTP %s %s", response.status, body
            )
            return {
                "error": f"{default_error}: HTTP {response.status}",
                "status": response.status,
            }

        if response.ok:
            if isinstance(payload, dict):
                return payload
            return {
                "error": f"{default_error}: unexpected response type",
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

        logger.debug(
            "Non-dict error from Sardis: HTTP %s %s",
            response.status,
            str(payload)[:200],
        )
        return {
            "error": f"{default_error}: HTTP {response.status}",
            "status": response.status,
        }
