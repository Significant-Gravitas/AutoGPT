"""
Shared configuration for all ElevenLabs blocks using the SDK pattern.
"""

from backend.sdk import BlockCostType, ProviderBuilder

from ._webhook import ElevenLabsWebhookManager

# Configure the ElevenLabs provider with API key authentication
elevenlabs = (
    ProviderBuilder("elevenlabs")
    .with_api_key("ELEVENLABS_API_KEY", "ElevenLabs API Key")
    .with_webhook_manager(ElevenLabsWebhookManager)
    .with_base_cost(2, BlockCostType.RUN)  # Base cost for API calls
    .build()
)
