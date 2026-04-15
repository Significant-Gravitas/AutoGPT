"""
Shared configuration for all Joy Trust Network blocks.

Joy provides trust verification for AI agent delegation. Before delegating
a task to another agent, verify their trust score meets your threshold.
"""

import httpx

from backend.sdk import APIKeyCredentials, ProviderBuilder, SecretStr

JOY_BASE_URL = "https://choosejoy.com.au"

joy_trust = (
    ProviderBuilder("joy_trust")
    .with_api_key("JOY_API_KEY", "Joy Trust API Key (optional, increases rate limits)")
    .build()
)

# Test credentials for unit tests
TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="joy_trust",
    title="Mock Joy Trust API Key",
    api_key=SecretStr("mock-joy-api-key"),
    expires_at=None,
)

TEST_CREDENTIALS_INPUT = {
    "id": TEST_CREDENTIALS.id,
    "provider": TEST_CREDENTIALS.provider,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


def _client(credentials: APIKeyCredentials | None = None) -> httpx.AsyncClient:
    """Create an httpx client for Joy API requests."""
    headers = {"Content-Type": "application/json"}
    if credentials and credentials.api_key:
        headers["x-api-key"] = credentials.api_key.get_secret_value()
    return httpx.AsyncClient(base_url=JOY_BASE_URL, headers=headers, timeout=10.0)


async def get_agent(agent_id: str, credentials: APIKeyCredentials | None = None) -> dict:
    """Fetch agent details from Joy API."""
    async with _client(credentials) as client:
        response = await client.get(f"/agents/{agent_id}")
        response.raise_for_status()
        return response.json()


async def discover_agents(
    query: str | None = None,
    capability: str | None = None,
    limit: int = 10,
    credentials: APIKeyCredentials | None = None,
) -> dict:
    """Discover agents from Joy API."""
    params = {"limit": limit}
    if query:
        params["query"] = query
    if capability:
        params["capability"] = capability

    async with _client(credentials) as client:
        response = await client.get("/agents/discover", params=params)
        response.raise_for_status()
        return response.json()
