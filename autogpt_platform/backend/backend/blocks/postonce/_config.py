from backend.sdk import ProviderBuilder

# PostOnce bills the user's own PostOnce plan, so the provider registers no
# platform key and no base_cost, and its blocks report no provider_cost.
postonce = (
    ProviderBuilder("postonce")
    .with_description(
        "Publish and schedule posts to LinkedIn, X, Threads, Bluesky, Facebook, "
        "Instagram, TikTok, YouTube and Pinterest"
    )
    .with_supported_auth_types("api_key")
    .build()
)

CREDENTIALS_DESCRIPTION = "PostOnce API key from Developers in the PostOnce dashboard"
