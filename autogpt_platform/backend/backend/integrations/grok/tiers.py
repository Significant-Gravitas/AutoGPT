"""Which xAI plan an account is on.

Two vocabularies describe the same thing and they do not agree. ``/user``
returns product names (``GrokPro``), and the access token's ``tier`` claim
carries an internal name (``supergrok``) or its ordinal. This maps between
them, and exists mainly because of one trap:

    GrokPro       -> supergrok        # the $30 SuperGrok plan
    SuperGrokPro  -> supergrok_heavy  # the Heavy plan

The product name that starts with "SuperGrok" is *not* the SuperGrok plan,
and the one that does not, is. Any string test -- ``startswith``,
``"SuperGrok" in name``, sorting by name -- mis-buckets every plan, and it
mis-buckets them into each other's price bands rather than into an error.
So the mapping is exhaustive and explicit, and a name that is not in it is
returned as unknown rather than guessed at.

The other thing worth stating: **a free account is a working connection.**
On xAI the plan raises a usage limit rather than unlocking access, so
``is_paid`` answers "will this account have room to work with", not "may
this account be used at all". Treating free as unusable would turn a
supported state into an onboarding failure.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

UNKNOWN_TIER = "unknown"
FREE_TIER = "free"

# Product name as ``/user`` reports it -> the token's internal tier name.
# Exhaustive on purpose: see the module docstring for why no string rule
# can stand in for it.
_PRODUCT_TO_TIER: dict[str, str] = {
    "GrokPro": "supergrok",
    "SuperGrokPro": "supergrok_heavy",
    "SuperGrokPlus": "supergrok_plus",
    "SuperGrokLite": "supergrok_lite",
    "XBasic": "x_basic",
    "XPremium": "x_premium",
    "XPremiumPlus": "x_premium_plus",
}

# The ordinal form of the same claim, for tokens that carry a number.
_ORDINAL_TO_TIER: dict[int, str] = {
    0: FREE_TIER,
    1: "supergrok",
    2: "x_basic",
    3: "x_premium",
    4: "x_premium_plus",
    5: "supergrok_heavy",
    6: "supergrok_lite",
    7: "supergrok_plus",
}

_PAID_TIERS = frozenset(_PRODUCT_TO_TIER.values())


def tier_for_product(product_name: str | None) -> str:
    """The internal tier for a product name from ``/user``."""
    if not product_name:
        return UNKNOWN_TIER
    name = product_name.strip()
    if name.lower() == FREE_TIER:
        return FREE_TIER
    return _PRODUCT_TO_TIER.get(name, UNKNOWN_TIER)


def is_paid(tier: str | None) -> bool:
    """Whether this tier gets more than the free allowance.

    Not "may this account be used". A free xAI account can run chats; it
    just has a small quota and a specific refusal when it runs out.
    """
    return tier in _PAID_TIERS


def tier_from_access_token(access_token: str) -> str:
    """The ``tier`` claim, read from the token rather than asked for.

    Saves a ``/user`` round trip on every turn, and is the same claim the
    server will enforce with -- so a token whose claim disagrees with a
    cached ``/user`` answer is the one that decides.

    Never raises. A token this cannot read is reported unknown: a malformed
    claim is not a reason to fail a chat that the server may well accept.
    """
    try:
        segments = access_token.split(".")
        if len(segments) < 2:
            return UNKNOWN_TIER
        payload = _decode_segment(segments[1])
    except Exception:
        logger.debug("[grok] could not read the tier claim from the token")
        return UNKNOWN_TIER

    return _tier_from_claim(payload.get("tier"))


def _tier_from_claim(claim: Any) -> str:
    if isinstance(claim, bool):
        # bool is an int in Python, and True would silently read as tier 1.
        return UNKNOWN_TIER
    if isinstance(claim, int):
        return _ORDINAL_TO_TIER.get(claim, str(claim))
    if isinstance(claim, str) and claim.strip():
        text = claim.strip()
        if text.isdigit():
            return _ORDINAL_TO_TIER.get(int(text), text)
        return text.lower()
    return UNKNOWN_TIER


def _decode_segment(segment: str) -> dict[str, Any]:
    # JWT uses base64url without padding; b64decode insists on padding.
    padded = segment + "=" * (-len(segment) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded)
    except (binascii.Error, ValueError):
        return {}
    decoded = json.loads(raw)
    return decoded if isinstance(decoded, dict) else {}
