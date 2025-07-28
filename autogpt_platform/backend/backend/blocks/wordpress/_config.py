import os

from backend.sdk import BlockCostType, ProviderBuilder

from ._oauth import WordPressOAuthHandler, WordPressScope

builder = ProviderBuilder("wordpress").with_base_cost(1, BlockCostType.RUN)


client_id = os.getenv("WORDPRESS_CLIENT_ID")
client_secret = os.getenv("WORDPRESS_CLIENT_SECRET")
WORDPRESS_OAUTH_IS_CONFIGURED = bool(client_id and client_secret)

if WORDPRESS_OAUTH_IS_CONFIGURED:
    builder = builder.with_oauth(
        WordPressOAuthHandler,
        scopes=[
            v.value
            for v in [
                WordPressScope.POSTS,
            ]
        ],
        client_id_env_var="WORDPRESS_CLIENT_ID",
        client_secret_env_var="WORDPRESS_CLIENT_SECRET",
    )

wordpress = builder.build()
