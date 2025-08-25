"""
Configuration for all DataForSEO blocks using the new SDK pattern.
"""

from backend.sdk import BlockCostType, ProviderBuilder

# Build the DataForSEO provider with username/password authentication
dataforseo = (
    ProviderBuilder("dataforseo")
    .with_user_password(
        username_env_var="DATAFORSEO_USERNAME",
        password_env_var="DATAFORSEO_PASSWORD",
        title="DataForSEO Credentials",
    )
    .with_base_cost(1, BlockCostType.RUN)
    .build()
)
