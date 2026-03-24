"""Sardis API client helpers used by the Sardis block suite."""

import logging
from typing import Any

from backend.blocks.sardis._auth import SardisCredentials
from backend.util.request import Requests, Response

logger = logging.getLogger(__name__)


class SardisClient:
    """Client for the Sardis API."""

    API_URL = "https://api.sardis.sh/api/v2"

    def __init__(self, credentials: SardisCredentials):
        """Store Sardis credentials and configure tolerant HTTP handling."""
        self.credentials = credentials
        self.requests = Requests(raise_for_status=False)

    def _get_headers(self) -> dict[str, str]:
        """Build authenticated Sardis API headers."""
        return {
            "X-API-Key": self.credentials.api_key.get_secret_value(),
            "Content-Type": "application/json",
        }

    async def send_payment(
        self,
        wallet_id: str,
        to: str,
        amount: float,
        token: str = "USDC",
        chain: str = "base",
        purpose: str = "Payment",
    ) -> dict[str, Any]:
        """Execute a policy-controlled payment."""
        response = await self.requests.post(
            f"{self.API_URL}/wallets/{wallet_id}/transfer",
            headers=self._get_headers(),
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
        response = await self.requests.get(
            f"{self.API_URL}/wallets/{wallet_id}/balance",
            headers=self._get_headers(),
            params={"token": token},
        )
        return self._normalize_response(
            response,
            default_error="Sardis balance request failed",
        )

    async def check_policy(
        self,
        wallet_id: str,
        amount: float,
        destination: str,
        token: str = "USDC",
    ) -> dict[str, Any]:
        """Check if a payment would be allowed by spending policy."""
        response = await self.requests.post(
            f"{self.API_URL}/policies/check",
            headers=self._get_headers(),
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
            return {
                "error": f"{default_error}: HTTP {response.status} {response.text()}",
                "status": response.status,
            }

        if response.ok:
            if isinstance(payload, dict):
                return payload
            return {"data": payload}

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

        return {
            "error": f"{default_error}: HTTP {response.status} {response.text()}",
            "status": response.status,
        }
