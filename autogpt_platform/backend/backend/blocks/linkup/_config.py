from backend.sdk import BlockCostType, ProviderBuilder

# Linkup bills a flat USD price per call, set by the search depth and output
# type (search) or by JavaScript rendering (fetch). Each block's run() reports
# that documented price via merge_stats, which populates
# NodeExecutionStats.provider_cost before billing reconciliation. 150 platform
# credits per USD matches the 1.5x margin baseline used by the other COST_USD
# blocks (see backend/data/block_cost_config.py).
linkup = (
    ProviderBuilder("linkup")
    .with_description("Real-time web search and page fetching for AI")
    .with_api_key("LINKUP_API_KEY", "Linkup API Key")
    .with_base_cost(150, BlockCostType.COST_USD)
    .build()
)
