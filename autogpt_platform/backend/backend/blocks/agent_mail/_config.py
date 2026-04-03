"""
Shared configuration for all AgentMail blocks.
"""

from agentmail import AsyncAgentMail

from backend.sdk import APIKeyCredentials, ProviderBuilder, SecretStr

agent_mail = (
    ProviderBuilder("agent_mail")
    .with_api_key("AGENTMAIL_API_KEY", "AgentMail API Key")
    .build()
)

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="agent_mail",
    title="Mock AgentMail API Key",
    api_key=SecretStr("mock-agentmail-api-key"),
    expires_at=None,
)

TEST_CREDENTIALS_INPUT = {
    "id": TEST_CREDENTIALS.id,
    "provider": TEST_CREDENTIALS.provider,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


def _client(credentials: APIKeyCredentials) -> AsyncAgentMail:
    """Create an AsyncAgentMail client from credentials."""
    return AsyncAgentMail(api_key=credentials.api_key.get_secret_value())
