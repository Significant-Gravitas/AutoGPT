from typing import List

from backend.blocks.apollo._auth import ApolloCredentials
from backend.blocks.apollo.models import (
    Contact,
    Organization,
    SearchOrganizationsRequest,
    SearchOrganizationsResponse,
    SearchPeopleRequest,
    SearchPeopleResponse,
)
from backend.util.request import Requests


class ApolloClient:
    """Client for the Apollo API"""

    API_URL = "https://api.apollo.io/api/v1"

    def __init__(self, credentials: ApolloCredentials):
        self.credentials = credentials
        self.requests = Requests()

    def _get_headers(self) -> dict[str, str]:
        return {"x-api-key": self.credentials.api_key.get_secret_value()}

    def search_people(self, query: SearchPeopleRequest) -> List[Contact]:
        """Search for people in Apollo"""
        response = self.requests.get(
            f"{self.API_URL}/mixed_people/search",
            headers=self._get_headers(),
            params=query.model_dump(exclude={"credentials"}),
        )
        response_json = response.json()
        parsed_response = SearchPeopleResponse(**response_json)
        if parsed_response.pagination.total_entries == 0:
            return []
        return parsed_response.people

    def search_organizations(
        self, query: SearchOrganizationsRequest
    ) -> List[Organization]:
        """Search for organizations in Apollo"""
        response = self.requests.get(
            f"{self.API_URL}/mixed_companies/search",
            headers=self._get_headers(),
            params=query.model_dump(exclude={"credentials"}),
        )
        parsed_response = SearchOrganizationsResponse(**response.json())
        if parsed_response.pagination.total_entries == 0:
            return []
        return parsed_response.organizations
