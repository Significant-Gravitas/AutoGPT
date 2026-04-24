"""Provider registration for fal — metadata only (auth lives in ``_auth.py``)."""

from backend.sdk import ProviderBuilder

fal = ProviderBuilder("fal").with_description("Hosted model inference").build()
