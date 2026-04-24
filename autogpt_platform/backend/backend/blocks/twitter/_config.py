"""Provider registration for X (Twitter) — metadata only (auth lives in ``_auth.py``)."""

from backend.sdk import ProviderBuilder

twitter = ProviderBuilder("twitter").with_description(
    "Tweets, timelines, and DMs"
).build()
