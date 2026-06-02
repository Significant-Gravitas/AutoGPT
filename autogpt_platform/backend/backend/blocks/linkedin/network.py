"""
LinkedIn network blocks using LinkedIn OAuth2.

Note: LinkedIn's public API does not expose a user's full connections list to
third-party apps; that endpoint (r_1st_connections) requires LinkedIn Partner
status. These blocks expose what IS available to regular OAuth apps:
  - Authenticated user profile (OpenID Connect)
  - First-degree connection count (r_1st_connections_size scope)
"""

import logging
from typing import Optional

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import OAuth2Credentials, SchemaField
from backend.util.request import Requests

from ._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    LinkedInCredentials,
    LinkedInCredentialsField,
)

logger = logging.getLogger(__name__)

LINKEDIN_API_BASE = "https://api.linkedin.com/v2"


async def _linkedin_get(path: str, access_token: str, params: dict | None = None):
    headers = {
        "Authorization": f"Bearer {access_token}",
        "X-Restli-Protocol-Version": "2.0.0",
    }
    return await Requests().get(
        f"{LINKEDIN_API_BASE}{path}", headers=headers, params=params or {}
    )


class GetLinkedInProfileBlock(Block):
    """Fetch the authenticated user's LinkedIn profile via OpenID Connect."""

    class Input(BlockSchemaInput):
        credentials: LinkedInCredentials = LinkedInCredentialsField(
            scopes=["openid", "profile", "email"]
        )

    class Output(BlockSchemaOutput):
        sub: str = SchemaField(description="LinkedIn member ID")
        name: str = SchemaField(description="Full name")
        given_name: Optional[str] = SchemaField(description="First name", default=None)
        family_name: Optional[str] = SchemaField(description="Last name", default=None)
        email: Optional[str] = SchemaField(description="Email address", default=None)
        picture: Optional[str] = SchemaField(
            description="Profile picture URL", default=None
        )
        error: str = SchemaField(description="Error message if request failed")

    def __init__(self):
        super().__init__(
            id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            description="Fetch your LinkedIn profile (name, email, picture) using OAuth2",
            categories={BlockCategory.SOCIAL},
            input_schema=GetLinkedInProfileBlock.Input,
            output_schema=GetLinkedInProfileBlock.Output,
            test_input={"credentials": TEST_CREDENTIALS_INPUT},
            test_output=[
                ("sub", "abc123"),
                ("name", "Jane Doe"),
                ("email", "jane@example.com"),
            ],
            test_credentials=TEST_CREDENTIALS,
            test_mock={
                "_fetch_profile": lambda *a, **kw: {
                    "sub": "abc123",
                    "name": "Jane Doe",
                    "email": "jane@example.com",
                }
            },
        )

    @staticmethod
    async def _fetch_profile(credentials: OAuth2Credentials) -> dict:
        response = await _linkedin_get(
            "/userinfo", credentials.access_token.get_secret_value()
        )
        response.raise_for_status()
        return response.json()

    async def run(
        self, input_data: Input, *, credentials: OAuth2Credentials, **kwargs
    ) -> BlockOutput:
        try:
            data = await self._fetch_profile(credentials)
            yield "sub", data.get("sub", "")
            yield "name", data.get("name", "")
            yield "given_name", data.get("given_name")
            yield "family_name", data.get("family_name")
            yield "email", data.get("email")
            yield "picture", data.get("picture")
        except Exception as e:
            logger.error(f"Error fetching LinkedIn profile: {e}")
            yield "error", str(e)


class GetLinkedInNetworkSizeBlock(Block):
    """
    Return the number of first-degree LinkedIn connections for the authenticated user.

    Uses the r_1st_connections_size scope — available to all LinkedIn apps.
    The full connections list (names, profiles) requires LinkedIn Partner API
    access (r_1st_connections scope) which is not available to regular apps.
    """

    class Input(BlockSchemaInput):
        credentials: LinkedInCredentials = LinkedInCredentialsField(
            scopes=["r_1st_connections_size"]
        )

    class Output(BlockSchemaOutput):
        connection_count: int = SchemaField(
            description="Number of first-degree connections"
        )
        error: str = SchemaField(description="Error message if request failed")

    def __init__(self):
        super().__init__(
            id="b2c3d4e5-f6a7-8901-bcde-f12345678901",
            description="Get your LinkedIn first-degree connection count (requires r_1st_connections_size scope)",
            categories={BlockCategory.SOCIAL},
            input_schema=GetLinkedInNetworkSizeBlock.Input,
            output_schema=GetLinkedInNetworkSizeBlock.Output,
            test_input={"credentials": TEST_CREDENTIALS_INPUT},
            test_output=[("connection_count", 342)],
            test_credentials=TEST_CREDENTIALS,
            test_mock={"_fetch_connection_count": lambda *a, **kw: 342},
        )

    @staticmethod
    async def _fetch_connection_count(credentials: OAuth2Credentials) -> int:
        response = await _linkedin_get(
            "/connections",
            credentials.access_token.get_secret_value(),
            params={"q": "viewer", "count": "0"},
        )
        response.raise_for_status()
        return response.json().get("paging", {}).get("total", 0)

    async def run(
        self, input_data: Input, *, credentials: OAuth2Credentials, **kwargs
    ) -> BlockOutput:
        try:
            count = await self._fetch_connection_count(credentials)
            yield "connection_count", count
        except Exception as e:
            logger.error(f"Error fetching LinkedIn connection count: {e}")
            yield "error", str(e)
