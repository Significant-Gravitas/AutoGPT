from backend.sdk import BlockCostType, ProviderBuilder

stagehand = (
    ProviderBuilder("stagehand")
    .with_api_key("STAGEHAND_API_KEY", "Stagehand API Key")
    .with_api_key(
        "STAGEHAND_MODEL_API_KEY", "Stagehand An API Key for a supported provider "
    )
    .with_base_cost(1, BlockCostType.RUN)
    .build()
)
