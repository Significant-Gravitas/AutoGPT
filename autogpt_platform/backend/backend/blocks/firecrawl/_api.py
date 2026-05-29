from enum import Enum


class ScrapeFormat(Enum):
    MARKDOWN = "markdown"
    HTML = "html"
    RAW_HTML = "rawHtml"
    LINKS = "links"
    SCREENSHOT = "screenshot"
    SCREENSHOT_FULL_PAGE = "screenshot@fullPage"
    JSON = "json"
    CHANGE_TRACKING = "changeTracking"
