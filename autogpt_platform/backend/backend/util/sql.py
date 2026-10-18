"""Helpers for building SQL ``LIKE``/``ILIKE`` patterns from user input."""


def escape_like(value: str) -> str:
    """Return ``value`` as literal ``LIKE``/``ILIKE`` pattern text for Postgres.

    prisma-client-py does not escape ``contains``/``startswith``/``endswith``
    or case-insensitive ``equals`` (all compiled to ``LIKE``/``ILIKE``), so a
    raw ``_`` or ``%`` acts as a wildcard. Postgres' default ``LIKE`` escape
    character is the backslash, so it is escaped first. Wrap every
    user-supplied value passed to those filters, or to a raw ``LIKE`` pattern,
    exactly once.

    Raises:
        ValueError: if ``value`` contains a NUL character. Postgres ``text``
            can't hold one, so the query would fail; the API maps this to 400.
    """
    if "\x00" in value:
        raise ValueError("Search text must not contain NUL characters")
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
