"""Shared attribute-resolution helpers for Graphiti edge/episode objects.

graphiti-core edge and episode objects have varying attribute names across
versions. These helpers centralise the fallback chains so there's one place
to update when upstream changes an attribute name.
"""


def extract_fact(edge) -> str:
    """Extract the human-readable fact from an edge object."""
    return getattr(edge, "fact", None) or getattr(edge, "name", "") or ""


def extract_temporal_validity(edge) -> tuple[str, str]:
    """Return ``(valid_from, valid_to)`` for an edge."""
    valid_from = getattr(edge, "valid_at", None) or "unknown"
    valid_to = getattr(edge, "invalid_at", None) or "present"
    return str(valid_from), str(valid_to)


def extract_episode_body(episode, max_len: int = 500) -> str:
    """Extract the body text from an episode object, truncated to *max_len*."""
    body = str(
        getattr(episode, "content", None)
        or getattr(episode, "body", None)
        or getattr(episode, "episode_body", None)
        or ""
    )
    return body[:max_len]


def extract_episode_timestamp(episode) -> str:
    """Extract the created_at timestamp from an episode object."""
    return str(getattr(episode, "created_at", None) or "")
