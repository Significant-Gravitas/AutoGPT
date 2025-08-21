"""
DataForSEO API client with async support using the SDK patterns.
"""

import base64
from typing import Any, Dict, List, Optional

from backend.sdk import Requests, UserPasswordCredentials


class DataForSeoClient:
    """Client for the DataForSEO API using async requests."""

    API_URL = "https://api.dataforseo.com"

    def __init__(self, credentials: UserPasswordCredentials):
        self.credentials = credentials
        self.requests = Requests(
            trusted_origins=["https://api.dataforseo.com"],
            raise_for_status=False,
        )

    def _get_headers(self) -> Dict[str, str]:
        """Generate the authorization header using Basic Auth."""
        username = self.credentials.username.get_secret_value()
        password = self.credentials.password.get_secret_value()
        credentials_str = f"{username}:{password}"
        encoded = base64.b64encode(credentials_str.encode("ascii")).decode("ascii")
        return {
            "Authorization": f"Basic {encoded}",
            "Content-Type": "application/json",
        }

    async def keyword_suggestions(
        self,
        keywords: List[str],
        location_code: Optional[int] = None,
        language_code: Optional[str] = None,
        include_seed_keyword: bool = True,
        include_serp_info: bool = False,
        include_clickstream_data: bool = False,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get keyword suggestions from DataForSEO Labs.

        Args:
            keywords: Seed keywords (up to 200)
            location_code: Location code for targeting
            language_code: Language code (e.g., "en")
            include_seed_keyword: Include seed keyword in results
            include_serp_info: Include SERP data
            include_clickstream_data: Include clickstream metrics
            limit: Maximum number of results (up to 3000)

        Returns:
            API response with keyword suggestions
        """
        endpoint = f"{self.API_URL}/v3/dataforseo_labs/google/keyword_suggestions/live"

        payload = [
            {
                "keywords": keywords,
                "location_code": location_code,
                "language_code": language_code,
                "include_seed_keyword": include_seed_keyword,
                "include_serp_info": include_serp_info,
                "include_clickstream_data": include_clickstream_data,
                "limit": limit,
            }
        ]

        response = await self.requests.post(
            endpoint,
            headers=self._get_headers(),
            json=payload,
        )

        data = response.json()

        # Check for API errors
        if response.status != 200:
            error_message = data.get("status_message", "Unknown error")
            raise Exception(
                f"DataForSEO API error ({response.status}): {error_message}"
            )

        # Extract the results from the response
        if data.get("tasks") and len(data["tasks"]) > 0:
            task = data["tasks"][0]
            if task.get("status_code") == 20000:  # Success code
                return task.get("result", [])
            else:
                error_msg = task.get("status_message", "Task failed")
                raise Exception(f"DataForSEO task error: {error_msg}")

        return []

    async def related_keywords(
        self,
        keywords: List[str],
        location_code: Optional[int] = None,
        language_code: Optional[str] = None,
        include_seed_keyword: bool = True,
        include_serp_info: bool = False,
        include_clickstream_data: bool = False,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get related keywords from DataForSEO Labs.

        Args:
            keywords: Seed keywords (up to 200)
            location_code: Location code for targeting
            language_code: Language code (e.g., "en")
            include_seed_keyword: Include seed keyword in results
            include_serp_info: Include SERP data
            include_clickstream_data: Include clickstream metrics
            limit: Maximum number of results (up to 3000)

        Returns:
            API response with related keywords
        """
        endpoint = f"{self.API_URL}/v3/dataforseo_labs/google/related_keywords/live"

        payload = [
            {
                "keywords": keywords,
                "location_code": location_code,
                "language_code": language_code,
                "include_seed_keyword": include_seed_keyword,
                "include_serp_info": include_serp_info,
                "include_clickstream_data": include_clickstream_data,
                "limit": limit,
            }
        ]

        response = await self.requests.post(
            endpoint,
            headers=self._get_headers(),
            json=payload,
        )

        data = response.json()

        # Check for API errors
        if response.status != 200:
            error_message = data.get("status_message", "Unknown error")
            raise Exception(
                f"DataForSEO API error ({response.status}): {error_message}"
            )

        # Extract the results from the response
        if data.get("tasks") and len(data["tasks"]) > 0:
            task = data["tasks"][0]
            if task.get("status_code") == 20000:  # Success code
                return task.get("result", [])
            else:
                error_msg = task.get("status_message", "Task failed")
                raise Exception(f"DataForSEO task error: {error_msg}")

        return []
