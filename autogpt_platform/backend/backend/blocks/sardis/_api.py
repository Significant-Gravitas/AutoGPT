import logging
from typing import Any

from backend.blocks.sardis._auth import SardisCredentials
from backend.util.request import Requests

logger = logging.getLogger(__name__)


class SardisClient:
    """Client for the Sardis API."""

    API_URL = "https://api.sardis.sh/api/v2"

    def __init__(self, credentials: SardisCredentials):
        self.credentials = credentials
        self.requests = Requests()

    def _get_headers(self) -> dict[str, str]:
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
        return response.json()

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
        return response.json()

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
        return response.json()
