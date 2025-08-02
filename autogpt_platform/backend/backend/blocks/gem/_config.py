"""
Shared configuration for all GEM blocks using the new SDK pattern.
"""

from backend.sdk import BlockCostType, ProviderBuilder

# Configure the GEM provider once for all blocks
gem = (
    ProviderBuilder("gem")
    .with_api_key("GEM_API_KEY", "GEM API Key")
    .with_base_cost(1, BlockCostType.RUN)
    .build()
)
