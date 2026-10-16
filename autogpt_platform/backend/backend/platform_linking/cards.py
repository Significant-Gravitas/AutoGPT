"""Approval cards in a linked chat channel: the token their buttons carry,
and who may answer them.

Telegram caps a button's payload at 64 bytes, which a review id alone can
exceed, so a card's buttons carry a short token instead. The token names the
row, the conversation it was posted in and the options the card showed, so a
button can answer nothing the card did not show.
"""

import uuid
from typing import Literal

from prisma.enums import ReviewStatus
from pydantic import BaseModel

from backend.copilot.gate import channel
from backend.copilot.gate.review import GATE_NODE_PREFIX
from backend.data.db_accessors import platform_linking_db, review_db
from backend.data.redis_client import get_redis_async

from .chat import resolve_owner
from .models import CardAnswer, CardTurn, ChannelCard, Platform

AnswerPolicy = Literal["linking_owner", "any_member"]
# Toran, 2026-09-30: anyone who can message the bot can already make it do
# anything; the gate follows the user's will, it does not moderate access.
ANSWER_POLICY: AnswerPolicy = "any_member"

# As long as the held call it answers is kept.
CARD_TTL_SECONDS = 30 * 24 * 60 * 60
_KEY = "copilot-bot:card:"

_NOT_YOURS = "Only the person who linked this server to AutoGPT can answer this."
_EXPIRED = "This request has expired, so nothing ran."
_ANSWERED = "This was already answered."
_FAILED = "Couldn't send your answer. Nothing ran. Try again."


class PostedCard(BaseModel):
    """What a card's token names."""

    platform: Platform
    server_id: str | None
    # The linking owner's identity on the platform; in a DM, the person in it.
    answerer: str
    user_id: str
    session_id: str
    review_id: str
    options: list[channel.CardOption]


async def open_card(
    platform: Platform,
    platform_server_id: str | None,
    platform_user_id: str,
    session_id: str,
    review_id: str,
) -> ChannelCard | None:
    """The card for a row this conversation raised, or None when its owner is
    not waiting on that row."""
    if not review_id.startswith(GATE_NODE_PREFIX):
        return None
    user_id = await resolve_owner(platform.value, platform_server_id, platform_user_id)
    rows = await review_db().get_reviews_by_node_exec_ids([review_id], user_id)
    row = rows.get(review_id)
    if row is None or row.session_id != session_id:
        return None
    if row.status != ReviewStatus.WAITING:
        return None
    answerer = await _answerer(platform, platform_server_id, platform_user_id, user_id)
    if answerer is None:
        return None
    view = channel.card_for(row)
    card = PostedCard(
        platform=platform,
        server_id=platform_server_id,
        answerer=answerer,
        user_id=user_id,
        session_id=session_id,
        review_id=review_id,
        options=view.options,
    )
    token = uuid.uuid4().hex[:12]
    redis = await get_redis_async()
    await redis.set(_key(token), card.model_dump_json(), ex=CARD_TTL_SECONDS)
    return ChannelCard(
        token=token, text=view.text, options=[o.label for o in view.options]
    )


async def answer_card(
    platform: Platform,
    platform_server_id: str | None,
    clicker_id: str,
    token: str,
    index: int,
) -> CardAnswer:
    """Answer the row a button names, if the clicker may.

    A read first, so a click the policy refuses changes nothing; then GETDEL,
    so the answerer's own double-click answers once. The answer wakes the
    chat's next turn, which the bot carries into the channel.
    """
    redis = await get_redis_async()
    raw = await redis.get(_key(token))
    if not raw:
        return CardAnswer(text=_EXPIRED)
    card = PostedCard.model_validate_json(raw)
    if card.platform != platform or not may_answer(
        card, clicker_id, platform_server_id
    ):
        return CardAnswer(text=_NOT_YOURS)
    if not 0 <= index < len(card.options):
        return CardAnswer(text=_EXPIRED)
    if not await redis.getdel(_key(token)):
        return CardAnswer(text=_ANSWERED)
    option = card.options[index]
    outcome = await channel.answer(
        card.user_id, card.session_id, card.review_id, option.choice
    )
    if outcome == "failed":
        await redis.set(_key(token), raw, ex=CARD_TTL_SECONDS)
        return CardAnswer(text=_FAILED)
    if outcome == "expired":
        return CardAnswer(text=_EXPIRED)
    if outcome == "answered_elsewhere":
        return CardAnswer(text=_ANSWERED)
    return CardAnswer(
        text=option.receipt,
        follow=CardTurn(
            session_id=card.session_id,
            user_id=card.user_id,
            review_id=card.review_id,
        ),
    )


def may_answer(card: PostedCard, clicker_id: str, server_id: str | None) -> bool:
    """Who may answer a card: someone in the conversation it was posted to,
    and only its linking owner unless ``ANSWER_POLICY`` says anyone there."""
    if server_id != card.server_id:
        return False
    return ANSWER_POLICY == "any_member" or clicker_id == card.answerer


async def _answerer(
    platform: Platform, server_id: str | None, sender_id: str, user_id: str
) -> str | None:
    if server_id is None:
        return sender_id
    links = await platform_linking_db().list_server_links(user_id)
    return next(
        (
            link.owner_platform_user_id
            for link in links
            if link.platform == platform.value and link.platform_server_id == server_id
        ),
        None,
    )


def _key(token: str) -> str:
    return f"{_KEY}{token}"
