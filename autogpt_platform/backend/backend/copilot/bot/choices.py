"""Redis-backed storage for in-flight ask_question choice buttons.

A native choice button/select (Discord component, Slack block action,
Telegram inline keyboard, Teams Adaptive Card action) round-trips a short
opaque token rather than the option text itself -- Telegram's
``callback_data`` caps at 64 bytes, too tight for arbitrary option text, and
carrying every platform's payload the same way keeps the click handlers
uniform. The click handler looks the real text up here by (platform, token,
index).

A token is bound to the person the question was asked of. In a shared
channel the buttons are visible to everyone, and a click is a great deal
easier than typing a reply -- without the binding, any passer-by could
consume the token and leave the person who was actually asked with nothing
but "this question has expired".
"""

import json
import uuid
from typing import Optional

from pydantic import BaseModel

from backend.data.redis_client import get_redis_async

CHOICE_TTL = 3600  # 1 hour -- long enough to answer, short enough that a
# stale button reliably reports "expired" instead of silently misfiring.


class ResolvedChoice(BaseModel):
    """Outcome of a click on a choice button.

    ``text`` is the chosen option, or ``None`` when the token is expired,
    forged, already used, or the index is out of range. ``refused`` marks
    the one case that is none of those: somebody other than the person
    asked clicked it. The token is deliberately left intact then, so the
    right person can still answer.
    """

    text: Optional[str]
    refused: bool = False


def _key(platform: str, token: str) -> str:
    return f"copilot-bot:choice:{platform}:{token}"


async def store_choice(platform: str, options: list[str], owner_id: str) -> str:
    """Persist a question's options against the user it was asked of.

    Returns an opaque token that native button/select payloads can carry
    within their size limits.
    """
    token = uuid.uuid4().hex[:12]
    redis = await get_redis_async()
    await redis.set(
        _key(platform, token),
        json.dumps({"options": options, "owner": owner_id}),
        ex=CHOICE_TTL,
    )
    return token


async def resolve_choice(
    platform: str, token: str, index: int, clicker_id: str
) -> ResolvedChoice:
    """Fetch-and-clear the option text for ``token``/``index``.

    Two steps, and the order is what makes both properties hold:

    - A read first, so a click by anyone other than the owner can be
      refused *without mutating anything*. That is what stops a bystander
      in a shared channel destroying the question.
    - Then GETDEL, so the owner's own double-click (or a platform delivery
      retry racing itself) still consumes exactly once -- each extra
      continuation is a billable AutoPilot turn. The loser gets ``None``,
      the same as an expired or forged payload.

    There is no window between them worth closing: only the owner ever
    reaches the GETDEL, and their concurrent clicks serialise on it.
    """
    redis = await get_redis_async()
    key = _key(platform, token)
    raw = await redis.get(key)
    if not raw:
        return ResolvedChoice(text=None)
    if _owner_of(raw) != clicker_id:
        return ResolvedChoice(text=None, refused=True)

    claimed = await redis.getdel(key)
    if not claimed:
        return ResolvedChoice(text=None)
    options = _options_of(claimed)
    if not (0 <= index < len(options)):
        return ResolvedChoice(text=None)
    return ResolvedChoice(text=options[index])


async def clear_choice(platform: str, token: str) -> None:
    """Invalidate a token that was never resolved (e.g. the native send
    itself failed) -- ``resolve_choice`` already clears on a successful
    resolve, so this is not needed on that path."""
    redis = await get_redis_async()
    await redis.delete(_key(platform, token))


def _owner_of(raw: bytes | str) -> Optional[str]:
    try:
        return str(json.loads(raw)["owner"])
    except (ValueError, TypeError, KeyError):
        return None


def _options_of(raw: bytes | str) -> list[str]:
    try:
        return [str(option) for option in json.loads(raw)["options"]]
    except (ValueError, TypeError, KeyError):
        return []
