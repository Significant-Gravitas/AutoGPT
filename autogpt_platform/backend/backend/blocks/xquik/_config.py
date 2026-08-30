from backend.sdk import ProviderBuilder

xquik = (
    ProviderBuilder("xquik")
    .with_description("Public X post search without an X developer account")
    .with_api_key("X_TWITTER_SCRAPER_API_KEY", "Xquik API Key")
    .build()
)
