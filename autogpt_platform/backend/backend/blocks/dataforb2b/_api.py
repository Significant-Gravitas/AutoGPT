import logging
from typing import Any

from backend.blocks.dataforb2b._auth import DataForB2BCredentials
from backend.util.request import Requests

logger = logging.getLogger(name=__name__)


class DataForB2BClient:
    """Client for the DataForB2B API (https://api.dataforb2b.ai)."""

    API_URL = "https://api.dataforb2b.ai"

    def __init__(self, credentials: DataForB2BCredentials):
        self.credentials = credentials
        self.requests = Requests()

    def _get_headers(self) -> dict[str, str]:
        return {
            "api_key": self.credentials.api_key.get_secret_value(),
            "Content-Type": "application/json",
        }

    async def _post(self, path: str, payload: dict[str, Any]) -> dict:
        response = await self.requests.post(
            f"{self.API_URL}{path}",
            headers=self._get_headers(),
            json=payload,
        )
        return response.json()

    async def _get(self, path: str, params: dict[str, Any]) -> dict:
        response = await self.requests.get(
            f"{self.API_URL}{path}",
            headers=self._get_headers(),
            params=params,
        )
        return response.json()

    # --- Search -----------------------------------------------------------

    async def search_people(self, payload: dict[str, Any]) -> dict:
        """POST /search/people — find professional profiles by filters."""
        return await self._post("/search/people", payload)

    async def search_companies(self, payload: dict[str, Any]) -> dict:
        """POST /search/companies — find companies by filters."""
        return await self._post("/search/companies", payload)

    async def reasoning_search(self, payload: dict[str, Any]) -> dict:
        """POST /search/reasoning — natural-language search.

        May return ``status == "needs_input"`` with clarifying questions; in
        that case re-call with ``session_id`` + ``answers``.
        """
        return await self._post("/search/reasoning", payload)

    async def typeahead(self, type_: str, q: str, limit: int = 20) -> dict:
        """GET /typeahead — resolve the exact stored value for a free-text filter."""
        return await self._get("/typeahead", {"type": type_, "q": q, "limit": limit})

    # --- Enrich -----------------------------------------------------------

    async def enrich_profile(self, payload: dict[str, Any]) -> dict:
        """POST /enrich/profile — enrich a profile; ≥1 enrich_* flag required."""
        return await self._post("/enrich/profile", payload)

    async def enrich_company(self, company_identifier: str) -> dict:
        """POST /enrich/company — enrich a company by domain/name/identifier."""
        return await self._post(
            "/enrich/company", {"company_identifier": company_identifier}
        )
