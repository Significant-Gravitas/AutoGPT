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

An approval card's buttons are a second kind on the same widgets; their
token lives with the linking manager, which checks who may answer.
"""

import json
import logging
import uuid
from typing import TYPE_CHECKING, Literal, Optional

from pydantic import BaseModel

from backend.data.redis_client import get_redis_async
from backend.platform_linking.models import CardTurn

if TYPE_CHECKING:
    from .bot_backend import BotBackend

logger = logging.getLogger(__name__)

ButtonKind = Literal["qans", "appr"]
QUESTION_KIND: ButtonKind = "qans"
CARD_KIND: ButtonKind = "appr"
# Parsers map the prefix a click carries back onto its kind.
BUTTON_KINDS: dict[str, ButtonKind] = {
    QUESTION_KIND: QUESTION_KIND,
    CARD_KIND: CARD_KIND,
}

_EXPIRED_NOTICE = "This question has expired — type your answer instead."
_NOT_YOUR_QUESTION = (
    "This question was for someone else — they still need to answer it."
)
_NOT_SENT = "Couldn't send your answer. Nothing ran. Try again."

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


class ButtonAnswer(BaseModel):
    """A click on either kind of button.

    ``reply`` is what a question's click says as the clicker's next message;
    ``follow`` is the turn a card's click woke, which the bot carries here.
    ``text`` replaces the buttons when the click answered, and is otherwise
    shown to the clicker alone.
    """

    reply: Optional[str]
    text: str
    follow: Optional[CardTurn] = None

    @property
    def answered(self) -> bool:
        return self.reply is not None or self.follow is not None


async def answer_button(
    api: "BotBackend",
    platform: str,
    kind: ButtonKind,
    token: str,
    index: int,
    clicker_id: str,
    server_id: Optional[str],
) -> ButtonAnswer:
    """Never raises: a click runs detached from anything that would report it."""
    try:
        if kind == CARD_KIND:
            card = await api.answer_card(platform, server_id, clicker_id, token, index)
            return ButtonAnswer(reply=None, follow=card.follow, text=card.text)
        resolved = await resolve_choice(platform, token, index, clicker_id)
    except Exception:
        logger.exception(f"A {kind} click on {platform} could not be answered")
        return ButtonAnswer(reply=None, text=_NOT_SENT)
    if resolved.text is None:
        return ButtonAnswer(
            reply=None,
            text=_NOT_YOUR_QUESTION if resolved.refused else _EXPIRED_NOTICE,
        )
    return ButtonAnswer(reply=resolved.text, text=f"✅ You answered: {resolved.text}")


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
