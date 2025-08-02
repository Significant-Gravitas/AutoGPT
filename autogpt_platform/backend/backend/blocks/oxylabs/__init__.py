"""
Oxylabs Web Scraper API integration blocks.
"""

from .blocks import (
    OxylabsCallbackerIPListBlock,
    OxylabsCheckJobStatusBlock,
    OxylabsGetJobResultsBlock,
    OxylabsProcessWebhookBlock,
    OxylabsProxyFetchBlock,
    OxylabsSubmitBatchBlock,
    OxylabsSubmitJobAsyncBlock,
    OxylabsSubmitJobRealtimeBlock,
)

__all__ = [
    "OxylabsSubmitJobAsyncBlock",
    "OxylabsSubmitJobRealtimeBlock",
    "OxylabsSubmitBatchBlock",
    "OxylabsCheckJobStatusBlock",
    "OxylabsGetJobResultsBlock",
    "OxylabsProxyFetchBlock",
    "OxylabsProcessWebhookBlock",
    "OxylabsCallbackerIPListBlock",
]
