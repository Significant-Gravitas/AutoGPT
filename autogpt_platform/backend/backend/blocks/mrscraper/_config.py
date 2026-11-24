from backend.sdk import ProviderBuilder

# MrScraper bills in its own tokens and publishes no per-call USD rate, so this
# provider registers no base_cost and its blocks report no provider_cost, the
# same convention as other providers without a public rate. The blocks run on
# the user's own MrScraper API key.
mrscraper = (
    ProviderBuilder("mrscraper")
    .with_description("Web scraping, screenshots, brand data and web search")
    .with_api_key("MRSCRAPER_API_KEY", "MrScraper API Key")
    .build()
)


def mrscraper_credentials_field():
    return mrscraper.credentials_field(description="MrScraper API key")
