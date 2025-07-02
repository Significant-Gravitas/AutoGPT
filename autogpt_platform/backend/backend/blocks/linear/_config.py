"""
Shared configuration for all Linear blocks using the new SDK pattern.
"""

import os
from backend.sdk import BlockCostType, ProviderBuilder
from ._oauth import LinearOAuthHandler

# Check if Linear OAuth is configured
LINEAR_OAUTH_IS_CONFIGURED = bool(
    os.getenv("LINEAR_CLIENT_ID") and os.getenv("LINEAR_CLIENT_SECRET")
)

# Build the Linear provider
builder = ProviderBuilder("linear").with_base_cost(1, BlockCostType.RUN)

# Add OAuth support if configured
if LINEAR_OAUTH_IS_CONFIGURED:
    builder = builder.with_oauth(
        LinearOAuthHandler,
        scopes=["read", "write", "issues:create", "comments:create"]
    )

# Add API key support as a fallback
builder = builder.with_api_key("LINEAR_API_KEY", "Linear API Key")

# Build the provider
linear = builder.build()