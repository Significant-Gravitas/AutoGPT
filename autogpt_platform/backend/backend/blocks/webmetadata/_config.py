"""
Shared configuration for all Web Metadata Extractor blocks.
"""

from backend.sdk import BlockCostType, ProviderBuilder

# Web Metadata Extractor (https://webmetadataextractor.com) is a single REST
# API for SEO/OpenGraph metadata, HTTP security-header grading, tech-stack
# fingerprinting, public contact discovery, and clean Markdown extraction.
# Auth is a single RapidAPI key, free tier: 1,000 requests/month, no card.
webmetadata_extractor = (
    ProviderBuilder("web_metadata_extractor")
    .with_description("Security headers, tech-stack, and SEO intelligence for any URL")
    .with_api_key("WEB_METADATA_EXTRACTOR_API_KEY", "Web Metadata Extractor API Key")
    .with_base_cost(1, BlockCostType.RUN)
    .build()
)
