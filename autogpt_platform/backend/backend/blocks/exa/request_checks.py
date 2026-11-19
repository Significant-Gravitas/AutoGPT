"""Input checks shared by the Exa Search, Contents and Answer blocks.

They run before the request so a bad value fails with a message naming the
block field, rather than a generic 400 from Exa.
"""

import re

USER_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
COUNTRY_CODE_PATTERN = re.compile(r"^[A-Za-z]{2}$")


def checked_user_identifier(value: str | None) -> str | None:
    """Return the identifier if Exa will accept it, None if unset.

    Exa requires 1-64 letters, digits, underscores or hyphens and rejects
    anything else with a 400.
    """
    if not value:
        return None
    if not USER_IDENTIFIER_PATTERN.match(value):
        raise ValueError(
            "User Identifier must be 1-64 letters, digits, underscores or hyphens "
            "(an opaque ID such as a UUID or a hash; never an email or name)."
        )
    return value


def checked_country_code(value: str | None) -> str | None:
    """Return a two-letter ISO 3166-1 country code in upper case, None if unset."""
    if value is None or not value.strip():
        return None
    code = value.strip()
    if not COUNTRY_CODE_PATTERN.match(code):
        raise ValueError(
            f"User Location must be a two-letter ISO 3166-1 country code such as "
            f"'US' or 'GB', got {value!r}."
        )
    return code.upper()
