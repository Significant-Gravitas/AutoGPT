"""
Shared configuration for all Linear blocks using the new SDK pattern.
"""

import os

from backend.sdk import BlockCostType, ProviderBuilder

from ._oauth import LinearOAuthHandler

# Check if Linear OAuth is configured
client_id = os.getenv("LINEAR_CLIENT_ID")
client_secret = os.getenv("LINEAR_CLIENT_SECRET")
LINEAR_OAUTH_IS_CONFIGURED = bool(client_id and client_secret)

# Build the Linear provider
builder = ProviderBuilder("linear").with_base_cost(1, BlockCostType.RUN)

# Linear only supports OAuth authentication
if LINEAR_OAUTH_IS_CONFIGURED:
    builder = builder.with_oauth(
        LinearOAuthHandler, scopes=["read", "write", "issues:create", "comments:create"]
    )

# Build the provider
linear = builder.build()
