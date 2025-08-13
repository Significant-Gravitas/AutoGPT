from backend.sdk import BlockCostType, ProviderBuilder

from ._oauth import WordPressOAuthHandler, WordPressScope

wordpress = (
    ProviderBuilder("wordpress")
    .with_base_cost(1, BlockCostType.RUN)
    .with_oauth(
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
    .build()
)
