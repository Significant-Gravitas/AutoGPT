"""Shared security constants for field-level filtering.

Other modules (e.g. orchestrator, future blocks) import from here so the
sensitive-field list stays in one place.
"""

# Field names to exclude from hardcoded-defaults descriptions (case-insensitive).
SENSITIVE_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "credentials",
        "api_key",
        "password",
        "secret",
        "token",
        "auth",
        "authorization",
        "access_token",
        "refresh_token",
    }
)
