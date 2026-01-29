from .playwright_browser import BrowsingError, WebPlaywrightComponent
from .search import WebSearchComponent

# WebPlaywrightComponent is the default browser component
WebBrowserComponent = WebPlaywrightComponent

__all__ = [
    "WebSearchComponent",
    "BrowsingError",
    "WebPlaywrightComponent",
    "WebBrowserComponent",
]
