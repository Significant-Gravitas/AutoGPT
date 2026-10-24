"""A proposal card is answered once: a second decision on it is refused,
whether it comes from a stale tab, a double click or two tabs at once."""

import asyncio
import uuid
from unittest.mock import AsyncMock

import pytest
import pytest_mock

from backend.copilot.gate import card_decided
from backend.copilot.gate.card_decided import decided_ids, is_already_answered
from backend.copilot.model import ChatMessage
from backend.data import redis_client
from backend.util.testing import is_tcp_port_reachable

needs_redis = pytest.mark.skipif(
    not is_tcp_port_reachable(redis_client.HOST, redis_client.PORT),
    reason="needs Redis",
)


def _tagged(row_id: str, *confirmation_ids: str) -> ChatMessage:
    return ChatMessage(
        id=row_id,
        role="user",
        content="\n".join(
            f"Approved: hire Maya (confirmation_id: {cid})." for cid in confirmation_ids
        ),
        metadata={
            "kind": "card_decision",
            "decisions": [
                {
                    "verdict": "approved",
                    "confirmation_id": cid,
                    "verb": "hire",
                    "name": "Maya",
                }
                for cid in confirmation_ids
            ],
        },
    )


def test_the_chat_says_which_cards_were_answered() -> None:
    messages = [
        ChatMessage(role="user", content="Get someone for marketing."),
        _tagged("row-1", "c-1"),
        # Saved before rows were tagged: the line itself is the record.
        ChatMessage(
            id="row-2",
            role="user",
            content="Not approved: do not hire Ada, discard that proposal "
            "(confirmation_id: c-2).",
        ),
        # Otto quoting a line is not the user deciding.
        ChatMessage(
            role="assistant", content="Approved: hire Bo (confirmation_id: c-3)."
        ),
    ]

    assert decided_ids(messages, skip_message_id=None) == {"c-1", "c-2"}
    # A retransmit of the request that wrote row-1 is not a second answer.
    assert decided_ids(messages, skip_message_id="row-1") == {"c-2"}


@needs_redis
async def test_only_one_of_two_racing_answers_claims_a_card() -> None:
    session = f"s-{uuid.uuid4()}"

    results = await asyncio.gather(
        *(card_decided._claim(session, ["c-1"], f"tab-{i}") for i in range(10))
    )

    assert sum(results) == 1


@needs_redis
async def test_a_claim_is_all_or_nothing_and_its_own_retry_passes() -> None:
    session = f"s-{uuid.uuid4()}"

    assert await card_decided._claim(session, ["c-1", "c-2"], "tab-a")
    assert await card_decided._claim(session, ["c-1", "c-2"], "tab-a")
    assert not await card_decided._claim(session, ["c-2", "c-3"], "tab-b")
    # The refused claim took nothing, so c-3 is still free.
    assert await card_decided._claim(session, ["c-3"], "tab-b")


async def test_a_card_answered_in_the_chat_is_refused_without_a_claim(
    mocker: pytest_mock.MockerFixture,
) -> None:
    session = mocker.MagicMock(messages=[_tagged("row-1", "c-1")])
    mocker.patch.object(
        card_decided, "get_chat_session", new_callable=AsyncMock, return_value=session
    )
    claim = mocker.patch.object(card_decided, "_claim", new_callable=AsyncMock)

    assert await is_already_answered("u-1", "s-1", ["c-1"], owner="row-9")
    claim.assert_not_awaited()


@pytest.mark.parametrize("claimed, answered", [(True, False), (False, True)])
async def test_an_unanswered_card_is_answered_by_whoever_claims_it(
    mocker: pytest_mock.MockerFixture, claimed: bool, answered: bool
) -> None:
    session = mocker.MagicMock(messages=[_tagged("row-1", "c-1")])
    mocker.patch.object(
        card_decided, "get_chat_session", new_callable=AsyncMock, return_value=session
    )
    claim = mocker.patch.object(
        card_decided, "_claim", new_callable=AsyncMock, return_value=claimed
    )

    assert await is_already_answered("u-1", "s-1", ["c-2"], owner="row-9") is answered
    claim.assert_awaited_once_with("s-1", ["c-2"], "row-9")


async def test_a_store_outage_never_blocks_an_answer(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch.object(
        card_decided,
        "get_chat_session",
        new_callable=AsyncMock,
        side_effect=RuntimeError("db down"),
    )
    mocker.patch.object(
        card_decided, "get_redis_async", new_callable=AsyncMock, side_effect=OSError
    )

    assert not await is_already_answered("u-1", "s-1", ["c-1"], owner="row-9")
