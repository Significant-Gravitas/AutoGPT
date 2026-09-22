"""Shared share-token primitives used by execution + chat sharing.

A share token is a v4 UUID rendered with hyphens.  The format is enforced
by ``SHARE_TOKEN_PATTERN`` on every public route that takes a token,
which (a) makes the token un-guessable for callers, and (b) keeps the
404 fast path purely string-shape-based instead of a DB round-trip.
"""

import uuid

# Public route path validators must use this exact pattern so that an
# attacker probing for tokens cannot distinguish "well-formed but not
# shared" from "malformed input" — both 404 immediately.
SHARE_TOKEN_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"


def generate_share_token() -> str:
    return str(uuid.uuid4())
