"""Provider registration for Slant 3D — metadata only (auth lives in ``_api.py``)."""

from backend.sdk import ProviderBuilder

slant3d = (
    ProviderBuilder("slant3d")
    .with_description("On-demand 3D printing")
    .with_supported_auth_types("api_key")
    .build()
)
