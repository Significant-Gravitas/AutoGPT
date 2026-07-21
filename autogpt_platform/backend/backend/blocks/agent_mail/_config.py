"""
Shared configuration for all AgentMail blocks.
"""

from agentmail import AsyncAgentMail

from backend.sdk import APIKeyCredentials, BlockCostType, ProviderBuilder, SecretStr

# AgentMail is in beta with no published paid tier yet, but ~37 blocks
# without any BLOCK_COSTS entry means they currently execute wallet-free.
# 1 cr/call is a conservative interim floor so no AgentMail work leaks
# past billing. Revisit once AgentMail publishes usage-based pricing.
agent_mail = (
    ProviderBuilder("agent_mail")
    .with_description("Managed email accounts for agents")
    .with_api_key("AGENTMAIL_API_KEY", "AgentMail API Key")
    .with_base_cost(1, BlockCostType.RUN)
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
