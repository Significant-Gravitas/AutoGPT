"""
Shared configuration for all Exa blocks using the new SDK pattern.
"""

from backend.sdk import BlockCostType, ProviderBuilder

from ._webhook import ExaWebhookManager

# Configure the Exa provider once for all blocks
exa = (
    ProviderBuilder("exa")
    .with_description("Neural web search")
    .with_api_key("EXA_API_KEY", "Exa API Key")
    .with_webhook_manager(ExaWebhookManager)
    # Exa returns `cost_dollars.total` on every response and ExaSearchBlock
    # (plus ~45 sibling blocks that share this provider config) already
    # populates NodeExecutionStats.provider_cost with it. Bill 100 credits
    # per USD (~$0.01/credit): cheap searches stay at 1–2 credits, a Deep
    # Research run at $0.20 lands at 20 credits, matching provider spend.
    .with_base_cost(100, BlockCostType.COST_USD)
    .build()
)
