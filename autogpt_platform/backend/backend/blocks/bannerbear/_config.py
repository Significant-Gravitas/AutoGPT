from backend.sdk import BlockCostType, ProviderBuilder

bannerbear = (
    ProviderBuilder("bannerbear")
    .with_description("Auto-generate images and videos")
    .with_api_key("BANNERBEAR_API_KEY", "Bannerbear API Key")
    .with_base_cost(3, BlockCostType.RUN)
    .build()
)
