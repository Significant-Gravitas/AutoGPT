"""Posting an identical setup card to a conversation only once per window.

The per-turn guard in ``turn_stream`` stops a repeat inside one turn; these
helpers stop a follow-up turn that runs the same blocked block again from
posting the same card.
"""

import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# An identical setup card is not posted to the same conversation again within
# this window: the earlier one is still on screen and its button opens the
# same session. Long enough to cover a burst of follow-up turns, short enough
# that coming back to the thread later re-surfaces it.
SETUP_CARD_REPOST_SECONDS = 15 * 60

# Delete a setup-card claim only if it still holds this turn's token.
_RELEASE_SETUP_CARD_SCRIPT = (
    'if redis.call("get", KEYS[1]) == ARGV[1] then '
    'return redis.call("del", KEYS[1]) else return 0 end'
)


async def claim_setup_card(redis: Any, key: str, token: str, target_id: str) -> bool:
    """Whether this setup card should be posted: True the first time it is
    seen in this conversation within ``SETUP_CARD_REPOST_SECONDS``.

    The per-turn guard stops a repeat inside one turn, but a follow-up turn
    that runs the same blocked block again produces the same card, and the
    thread filled up with identical sign-in prompts. The claim holds this
    turn's ``token`` so only this turn can release it if the post fails.
    Fails open: a Redis blip costs a duplicate card, never a missing one.
    """
    try:
        claimed = await redis.set(key, token, nx=True, ex=SETUP_CARD_REPOST_SECONDS)
        # A retried SET whose first attempt landed but lost its reply finds
        # this turn's own token: the claim is still ours.
        if not claimed and await redis.get(key) == token:
            claimed = True
    except Exception:
        logger.warning("Setup-card dedupe unavailable; posting the card anyway")
        return True
    if not claimed:
        logger.info("Setup card already shown in target %s; not reposting", target_id)
    return bool(claimed)


async def release_setup_card(redis: Any, key: str, token: str) -> None:
    """Drop this turn's claim after a failed post, leaving alone a claim
    another turn took once ours expired."""
    try:
        await redis.eval(_RELEASE_SETUP_CARD_SCRIPT, 1, key, token)
    except Exception:
        logger.warning("Failed to release setup-card claim %s", key, exc_info=True)


def setup_card_key(
    platform: str,
    target_id: str,
    session_id: str,
    setup_output: dict[str, Any],
) -> str:
    """The card's identity is what it asks for (the agent/block and the
    missing credentials) plus the session its button opens."""
    info = setup_output.get("setup_info")
    info = info if isinstance(info, dict) else {}
    readiness = info.get("user_readiness")
    readiness = readiness if isinstance(readiness, dict) else {}
    missing = readiness.get("missing_credentials")
    identity = json.dumps(
        [
            session_id,
            str(info.get("agent_id") or ""),
            sorted(missing) if isinstance(missing, dict) else [],
            str(setup_output.get("message") or ""),
        ]
    )
    digest = hashlib.sha256(identity.encode()).hexdigest()
    return f"copilot-bot:setup-card:{platform}:{target_id}:{digest}"
