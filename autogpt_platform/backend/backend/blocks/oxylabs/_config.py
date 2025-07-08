"""
Shared configuration for all Oxylabs blocks using the SDK pattern.
"""

from backend.sdk import BlockCostType, ProviderBuilder

# Configure the Oxylabs provider with username/password authentication
oxylabs = (
    ProviderBuilder("oxylabs")
    .with_user_password(
        "OXYLABS_USERNAME", "OXYLABS_PASSWORD", "Oxylabs API Credentials"
    )
    .with_base_cost(10, BlockCostType.RUN)  # Higher cost for web scraping service
    .build()
)
