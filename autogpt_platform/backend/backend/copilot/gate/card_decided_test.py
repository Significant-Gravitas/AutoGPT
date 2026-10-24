"""A proposal card is answered once: a second decision on it is refused,
whether it comes from a stale tab, a double click or two tabs at once."""

import asyncio
import uuid
from unittest.mock import AsyncMock

import pytest
import pytest_mock

from backend.copilot.gate import card_approval, card_decided
from backend.copilot.gate.card_decided import (
    claim_answer,
    decided_ids,
    forget_unsaved_answers,
)
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


def _row(content: str, metadata: dict | None = None, role: str = "user"):
    return ChatMessage(role=role, content=content, metadata=metadata)


def test_the_chat_says_which_cards_the_user_answered() -> None:
    messages = [
        _row("Get someone for marketing."),
        _tagged("row-1", "c-1"),
        # Saved before rows were tagged: the line itself is the record.
        ChatMessage(
            id="row-2",
            role="user",
            content="Not approved: do not hire Ada, discard that proposal "
            "(confirmation_id: c-2).",
        ),
        # Queued behind a running turn: the row carries the queue's own
        # metadata, and is still the user's answer.
        _row(
            "Approved: hire Bo (confirmation_id: c-5).\nAnd start on Monday.",
            {"cost_context": {}, "llm_auth_provider": "platform"},
        ),
        # Otto quoting a line is not the user deciding.
        _row("Approved: hire Bo (confirmation_id: c-3).", role="assistant"),
        # Rows the server wrote for something else are not either.
        _row(
            "Approved: hire Cy (confirmation_id: c-4).",
            {"from_session_id": "s-parent"},
        ),
        _row("Approved: hire Di (confirmation_id: c-6).", {"held_calls_answered": 1}),
        _row(
            "Approved: hire Ed (confirmation_id: c-7).",
            {"hidden": True, "kind": "expert_kickoff"},
        ),
    ]

    assert decided_ids(messages, skip_message_id=None) == {"c-1", "c-2", "c-5"}
    # A retransmit of the request that wrote row-1 is not a second answer.
    assert decided_ids(messages, skip_message_id="row-1") == {"c-2", "c-5"}


@needs_redis
async def test_only_one_of_two_racing_answers_claims_a_card() -> None:
    session = f"s-{uuid.uuid4()}"

    results = await asyncio.gather(
        *(card_decided._claim(session, ["c-1"], f"tab-{i}") for i in range(10))
    )

    assert results.count("claimed") == 1
    assert results.count("taken") == 9


@needs_redis
async def test_a_claim_is_all_or_nothing_and_its_own_retry_holds_it() -> None:
    session = f"s-{uuid.uuid4()}"

    assert await card_decided._claim(session, ["c-1", "c-2"], "tab-a") == "claimed"
    assert await card_decided._claim(session, ["c-1", "c-2"], "tab-a") == "held"
    assert await card_decided._claim(session, ["c-2", "c-3"], "tab-b") == "taken"
    # The refused claim took nothing, so c-3 is still free.
    assert await card_decided._claim(session, ["c-3"], "tab-b") == "claimed"


@needs_redis
async def test_a_failed_request_gives_back_only_its_own_claim() -> None:
    session = f"s-{uuid.uuid4()}"
    assert await card_decided._claim(session, ["c-1"], "tab-a") == "claimed"

    await card_decided.release_claim(session, ["c-1"], "tab-b")
    assert await card_decided._claim(session, ["c-1"], "tab-b") == "taken"

    await card_decided.release_claim(session, ["c-1"], "tab-a")
    assert await card_decided._claim(session, ["c-1"], "tab-b") == "claimed"


@needs_redis
async def test_answers_dropped_before_they_were_saved_are_forgotten(
    mocker: pytest_mock.MockerFixture,
) -> None:
    session_id = f"s-{uuid.uuid4()}"
    # c-2 was answered by a row the chat holds; a dropped copy must not undo it.
    session = mocker.MagicMock(messages=[_tagged("row-1", "c-2")])
    mocker.patch.object(
        card_decided, "get_chat_session", new_callable=AsyncMock, return_value=session
    )
    for cid in ("c-1", "c-2"):
        await card_approval.record_card_decisions(
            "u-1", session_id, f"Approved: hire Maya (confirmation_id: {cid})."
        )
        assert (
            await card_decided._claim(session_id, [cid], f"queued-{cid}") == "claimed"
        )

    await forget_unsaved_answers(
        "u-1",
        session_id,
        [
            "Approved: hire Maya (confirmation_id: c-1).",
            "Approved: hire Maya (confirmation_id: c-2).",
        ],
    )

    assert await card_decided._claim(session_id, ["c-1"], "tab-b") == "claimed"
    assert not await card_approval.approved_on_card(
        "confirm_expert_change", {"confirmation_id": "c-1"}, "u-1", session_id
    )
    assert await card_decided._claim(session_id, ["c-2"], "tab-b") == "taken"
    assert await card_approval.approved_on_card(
        "confirm_expert_change", {"confirmation_id": "c-2"}, "u-1", session_id
    )


async def test_a_card_answered_in_the_chat_is_refused_without_a_claim(
    mocker: pytest_mock.MockerFixture,
) -> None:
    session = mocker.MagicMock(messages=[_tagged("row-1", "c-1")])
    mocker.patch.object(
        card_decided, "get_chat_session", new_callable=AsyncMock, return_value=session
    )
    claim = mocker.patch.object(card_decided, "_claim", new_callable=AsyncMock)

    assert await claim_answer("u-1", "s-1", ["c-1"], owner="row-9") == "answered"
    claim.assert_not_awaited()


@pytest.mark.parametrize(
    "claimed, result",
    [("claimed", "claimed"), ("held", "held"), ("taken", "answered")],
)
async def test_an_unanswered_card_is_answered_by_whoever_claims_it(
    mocker: pytest_mock.MockerFixture, claimed: str, result: str
) -> None:
    session = mocker.MagicMock(messages=[_tagged("row-1", "c-1")])
    mocker.patch.object(
        card_decided, "get_chat_session", new_callable=AsyncMock, return_value=session
    )
    claim = mocker.patch.object(
        card_decided, "_claim", new_callable=AsyncMock, return_value=claimed
    )

    assert await claim_answer("u-1", "s-1", ["c-2"], owner="row-9") == result
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

    # Nothing to give back either: the claim was never taken.
    assert await claim_answer("u-1", "s-1", ["c-1"], owner="row-9") == "held"
