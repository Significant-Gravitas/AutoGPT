"""Shared security constants for field-level filtering.

Other modules (e.g. orchestrator, future blocks) import from here so the
sensitive-field list stays in one place.
"""

from typing import Any

# Substrings used for case-insensitive matching against field names.
# A field is considered sensitive if any of these appear anywhere in the
# lowercased field name (substring match, not exact match).
SENSITIVE_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "credentials",
        "api_key",
        "password",
        "secret",
        "secret_key",
        "private_key",
        "client_secret",
        "token",
        "auth",
        "authorization",
        "access_token",
        "refresh_token",
        "bearer_token",
        "passphrase",
        "webhook_secret",
    }
)


def is_sensitive_field(field_name: str) -> bool:
    """Check if a field name is sensitive using substring matching.

    Returns True if the lowercased field_name contains any of the
    SENSITIVE_FIELD_NAMES as a substring.
    """
    lower = field_name.lower()
    return any(s in lower for s in SENSITIVE_FIELD_NAMES)


def filter_sensitive_fields(
    data: dict[str, Any],
    *,
    extra_excludes: frozenset[str] | None = None,
    linked_fields: set[str] | None = None,
) -> dict[str, Any]:
    """Return a copy of *data* with sensitive and private fields removed.

    This also recursively scans one level of nested dicts to remove keys
    that match sensitive field names, preventing secrets from leaking
    through benign top-level key names (e.g. ``{"config": {"api_key": "..."}}``)

    Args:
        data: The dict to filter.
        extra_excludes: Additional exact field names to exclude (e.g.
            ``{"graph_id", "graph_version", "input_schema"}``).
        linked_fields: Field names to exclude because they are linked.
    """
    result: dict[str, Any] = {}
    excludes = extra_excludes or frozenset()
    linked = linked_fields or set()

    for k, v in data.items():
        if k in linked:
            continue
        if k.startswith("_"):
            continue
        if k in excludes:
            continue
        if is_sensitive_field(k):
            continue
        # Recursively filter nested dicts one level deep
        if isinstance(v, dict):
            v = {nk: nv for nk, nv in v.items() if not is_sensitive_field(nk)}
            if not v:
                continue
        result[k] = v
    return result
