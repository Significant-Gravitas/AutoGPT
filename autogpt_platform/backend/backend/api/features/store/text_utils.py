"""Small text helpers shared across store search modules."""

import logging

logger = logging.getLogger(__name__)

_MAX_CAMELCASE_INPUT_LEN = 500


def split_camelcase(text: str) -> str:
    """Split CamelCase into separate words.

    Uses a single-pass character-by-character algorithm to avoid any
    regex backtracking concerns (guaranteed O(n) time).

    .. note::

       Only CamelCase boundaries are detected.  Underscores, hyphens, and
       other non-alpha separators are left as-is (e.g. ``"HTTP_Request"``
       is **not** converted to ``"HTTP Request"``).

       Single-letter uppercase prefixes are not split from the following
       word (e.g. ``"ABlock"`` stays ``"ABlock"``).  This matches the
       original regex behaviour ``([A-Z]{2,})([A-Z][a-z])``.

    Examples::

        >>> split_camelcase("AITextGeneratorBlock")
        'AI Text Generator Block'
        >>> split_camelcase("OAuth2Block")
        'OAuth2 Block'
    """
    if len(text) > _MAX_CAMELCASE_INPUT_LEN:
        logger.debug(
            "split_camelcase: truncating input from %d to %d chars",
            len(text),
            _MAX_CAMELCASE_INPUT_LEN,
        )
        text = text[:_MAX_CAMELCASE_INPUT_LEN]
    if len(text) <= 1:
        return text

    parts: list[str] = []
    prev = 0
    for i in range(1, len(text)):
        # Insert split between lowercase/digit and uppercase: "camelCase" -> "camel|Case"
        if (text[i - 1].islower() or text[i - 1].isdigit()) and text[i].isupper():
            parts.append(text[prev:i])
            prev = i
        # Insert split between uppercase run (2+ chars) and uppercase+lowercase:
        # "AIText" -> "AI|Text".  Requires at least 3 consecutive uppercase chars
        # before the lowercase so that the left part keeps 2+ uppercase chars
        # (mirrors the original regex r"([A-Z]{2,})([A-Z][a-z])").
        elif (
            i >= 2
            and text[i - 2].isupper()
            and text[i - 1].isupper()
            and text[i].islower()
            and (i - 1 - prev) >= 2  # left part must retain at least 2 upper chars
        ):
            parts.append(text[prev : i - 1])
            prev = i - 1

    parts.append(text[prev:])
    return " ".join(parts)
