"""Utility functions for converting between our ScrapeFormat enum and firecrawl FormatOption types."""

from typing import List

from firecrawl.v2.types import FormatOption, ScreenshotFormat

from backend.blocks.firecrawl._api import ScrapeFormat


def convert_to_format_options(
    formats: List[ScrapeFormat],
) -> List[FormatOption]:
    """Convert our ScrapeFormat enum values to firecrawl FormatOption types.

    Handles special cases like screenshot@fullPage which needs to be converted
    to a ScreenshotFormat object.
    """
    result: List[FormatOption] = []

    for format_enum in formats:
        if format_enum.value == "screenshot@fullPage":
            # Special case: convert to ScreenshotFormat with full_page=True
            result.append(ScreenshotFormat(type="screenshot", full_page=True))
        else:
            # Regular string literals
            result.append(format_enum.value)

    return result
