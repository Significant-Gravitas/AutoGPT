"""
Configuration for all DataForSEO blocks using the new SDK pattern.
"""

from backend.sdk import BlockCostType, ProviderBuilder

# Build the DataForSEO provider with username/password authentication
dataforseo = (
    ProviderBuilder("dataforseo")
    .with_description("SEO and SERP data")
    .with_user_password(
        username_env_var="DATAFORSEO_USERNAME",
        password_env_var="DATAFORSEO_PASSWORD",
        title="DataForSEO Credentials",
    )
    # DataForSEO reports USD cost per task (e.g. $0.001/keyword returned).
    # DataForSeoClient stashes it on last_cost_usd; each block emits it via
    # merge_stats so the COST_USD resolver bills against real spend.
    # 1000 platform credits per USD → 1 credit per $0.001 (≈ 1 credit/
    # returned keyword on the standard tier).
    .with_base_cost(1000, BlockCostType.COST_USD)
    .build()
)
