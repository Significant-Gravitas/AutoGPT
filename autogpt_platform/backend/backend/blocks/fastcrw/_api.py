import os
from enum import Enum

# Default to the fastCRW managed cloud; self-hosters can override the base URL
# (e.g. http://localhost:3000) via the CRW_API_URL environment variable.
DEFAULT_FASTCRW_API_URL = "https://fastcrw.com/api"


def get_fastcrw_api_url() -> str:
    """Resolve the fastCRW base URL, honoring the self-host override."""
    return os.getenv("CRW_API_URL") or DEFAULT_FASTCRW_API_URL


class ScrapeFormat(Enum):
    MARKDOWN = "markdown"
    HTML = "html"
    RAW_HTML = "rawHtml"
    LINKS = "links"
    SCREENSHOT = "screenshot"
    SCREENSHOT_FULL_PAGE = "screenshot@fullPage"
    JSON = "json"
    CHANGE_TRACKING = "changeTracking"
