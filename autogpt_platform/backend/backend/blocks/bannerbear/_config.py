from backend.sdk import BlockCostType, ProviderBuilder

bannerbear = (
    ProviderBuilder("bannerbear")
    .with_api_key("BANNERBEAR_API_KEY", "Bannerbear API Key")
    .with_base_cost(1, BlockCostType.RUN)
    .build()
)
