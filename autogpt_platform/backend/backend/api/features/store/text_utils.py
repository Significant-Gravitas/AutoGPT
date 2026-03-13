"""Small text helpers shared across store search modules."""

import re


def split_camelcase(text: str) -> str:
    """Split CamelCase into separate words.

    Examples::

        >>> split_camelcase("AITextGeneratorBlock")
        'AI Text Generator Block'
        >>> split_camelcase("HTTPRequestBlock")
        'HTTP Request Block'
    """
    text = text[:500]  # Bound input length to prevent regex DoS
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    return text
