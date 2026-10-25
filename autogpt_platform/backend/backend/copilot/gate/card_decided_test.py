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


async def _claim(
    session_id: str, ids: list[str], owner: str, attempt: str | None = None
) -> str:
    """One attempt per owner unless a test says otherwise."""
    claim = await card_decided._claim(session_id, ids, owner, attempt or owner)
    return claim.result


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
        *(_claim(session, ["c-1"], f"tab-{i}") for i in range(10))
    )

    assert results.count("claimed") == 1
    # The winner has not saved yet: the others are told to wait, not that
    # the card was answered.
    assert results.count("pending") == 9


@needs_redis
async def test_a_claim_is_all_or_nothing_and_an_attempt_asking_again_keeps_it() -> None:
    session = f"s-{uuid.uuid4()}"

    assert await _claim(session, ["c-1", "c-2"], "tab-a") == "claimed"
    assert await _claim(session, ["c-1", "c-2"], "tab-a") == "claimed"
    assert await _claim(session, ["c-2", "c-3"], "tab-b") == "pending"
    # The refused claim took nothing, so c-3 is still free.
    assert await _claim(session, ["c-3"], "tab-b") == "claimed"


@needs_redis
async def test_a_failed_request_gives_back_only_its_own_claim() -> None:
    session = f"s-{uuid.uuid4()}"
    assert await _claim(session, ["c-1"], "tab-a") == "claimed"

    await card_decided.release_claim(session, ["c-1"], "tab-b", "tab-b")
    assert await _claim(session, ["c-1"], "tab-b") == "pending"

    await card_decided.release_claim(session, ["c-1"], "tab-a", "tab-a")
    assert await _claim(session, ["c-1"], "tab-b") == "claimed"


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
        assert await _claim(session_id, [cid], f"queued-{cid}") == "claimed"
        await card_decided.accept_claim(
            session_id, [cid], f"queued-{cid}", f"queued-{cid}"
        )

    await forget_unsaved_answers(
        "u-1",
        session_id,
        [
            "Approved: hire Maya (confirmation_id: c-1).",
            "Approved: hire Maya (confirmation_id: c-2).",
        ],
    )

    assert await _claim(session_id, ["c-1"], "tab-b") == "claimed"
    assert not await card_approval.approved_on_card(
        "confirm_expert_change", {"confirmation_id": "c-1"}, "u-1", session_id
    )
    assert await _claim(session_id, ["c-2"], "tab-b") == "answered"
    assert await card_approval.approved_on_card(
        "confirm_expert_change", {"confirmation_id": "c-2"}, "u-1", session_id
    )


@needs_redis
async def test_answers_dropped_from_a_chat_that_cannot_be_read_are_forgotten(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Stop drained a queued answer that was never saved. If the chat cannot
    be read to check, its approval and claim go anyway: a confirm must not
    pass on an answer the chat may not hold."""
    session_id = f"s-{uuid.uuid4()}"
    mocker.patch.object(
        card_decided,
        "get_chat_session",
        new_callable=AsyncMock,
        side_effect=RuntimeError("db down"),
    )
    line = "Approved: hire Maya (confirmation_id: c-1)."
    await card_approval.record_card_decisions("u-1", session_id, line)
    assert await _claim(session_id, ["c-1"], "queued") == "claimed"
    await card_decided.accept_claim(session_id, ["c-1"], "queued", "queued")

    await forget_unsaved_answers("u-1", session_id, [line])

    assert not await card_approval.approved_on_card(
        "confirm_expert_change", {"confirmation_id": "c-1"}, "u-1", session_id
    )
    assert await _claim(session_id, ["c-1"], "tab-b") == "claimed"


@needs_redis
async def test_an_answer_still_being_sent_is_waited_on_not_taken_as_given() -> None:
    """Tab A claims, tab B asks while A is still sending. If A then fails,
    B's retry must land; if A saves, B's must be refused."""
    session = f"s-{uuid.uuid4()}"
    assert await _claim(session, ["c-1"], "tab-a") == "claimed"

    assert await _claim(session, ["c-1"], "tab-b") == "pending"
    await card_decided.release_claim(session, ["c-1"], "tab-a", "tab-a")  # A failed
    assert await _claim(session, ["c-1"], "tab-b") == "claimed"

    await card_decided.accept_claim(session, ["c-1"], "tab-b", "tab-b")  # B saved
    assert await _claim(session, ["c-1"], "tab-a") == "answered"
    # An accepted answer is not given back by a late failure path either.
    await card_decided.release_claim(session, ["c-1"], "tab-b", "tab-b")
    assert await _claim(session, ["c-1"], "tab-a") == "answered"
    # A replay of the saved send is a replay, not a new answer.
    assert await _claim(session, ["c-1"], "tab-b") == "replayed"


@needs_redis
async def test_a_claim_whose_request_died_lapses(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Tab A claims, then its worker is killed before it saves or gives the
    claim back. The card must not stay locked: once the lease is past, the
    next answer lands. An accepted answer never lapses."""
    mocker.patch.object(card_decided, "_PENDING_LEASE_MS", 200)
    session = f"s-{uuid.uuid4()}"
    assert await _claim(session, ["c-1", "c-2"], "tab-a") == "claimed"
    await card_decided.accept_claim(session, ["c-2"], "tab-a", "tab-a")
    assert await _claim(session, ["c-1"], "tab-b") == "pending"

    await asyncio.sleep(0.3)  # A is gone

    assert await _claim(session, ["c-1"], "tab-b") == "claimed"
    assert await _claim(session, ["c-2"], "tab-b") == "answered"
    # A's late paths no longer touch B's claim.
    await card_decided.release_claim(session, ["c-1"], "tab-a", "tab-a")
    await card_decided.accept_claim(session, ["c-1"], "tab-a", "tab-a")
    assert await _claim(session, ["c-1"], "tab-c") == "pending"
    await card_decided.accept_claim(session, ["c-1"], "tab-b", "tab-b")
    assert await _claim(session, ["c-1"], "tab-c") == "answered"


@needs_redis
async def test_a_retransmit_does_not_ride_on_a_claim_still_being_sent() -> None:
    """Two copies of one send share its client id. While the first is still
    sending, the copy neither holds its claim nor records anything: if the
    first then fails and takes its approvals back, the copy's retry claims
    afresh. Once the first saved, the copy holds it as before."""
    session = f"s-{uuid.uuid4()}"
    first = await card_decided._claim(session, ["c-1"], "click-1", "try-1")
    assert first.result == "claimed"

    copy = await card_decided._claim(session, ["c-1"], "click-1", "try-2")
    assert copy.result == "in_progress"

    await card_decided.release_claim(session, ["c-1"], "click-1", "try-1")
    retry = await card_decided._claim(session, ["c-1"], "click-1", "try-3")
    assert retry.result == "claimed"
    await card_decided.accept_claim(session, ["c-1"], "click-1", "try-3")
    late = await card_decided._claim(session, ["c-1"], "click-1", "try-4")
    assert late.result == "replayed"


@needs_redis
async def test_a_claim_names_the_cards_someone_else_answered() -> None:
    """Another tab answered one card of a group: the claim says which, takes
    nothing, and the open card can still be claimed on its own."""
    session = f"s-{uuid.uuid4()}"
    assert (await card_decided._claim(session, ["c-1"], "tab-a", "a")).result == (
        "claimed"
    )
    await card_decided.accept_claim(session, ["c-1"], "tab-a", "a")

    both = await card_decided._claim(session, ["c-1", "c-2"], "tab-b", "b")
    assert both.result == "answered"
    assert both.answered == {"c-1"}
    rest = await card_decided._claim(session, ["c-2"], "tab-b", "b")
    assert rest.result == "claimed"


async def test_a_replay_of_a_saved_send_is_not_a_new_answer(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """The chat holds this very send's row (row-9 is its client id): the
    request is a replay, to be deduplicated, and records nothing."""
    session = mocker.MagicMock(messages=[_tagged("row-9", "c-1")])
    mocker.patch.object(
        card_decided, "get_chat_session", new_callable=AsyncMock, return_value=session
    )
    claim = mocker.patch.object(card_decided, "_claim", new_callable=AsyncMock)

    replay = await claim_answer("u-1", "s-1", ["c-1"], owner="row-9", attempt="a")

    assert replay.result == "replayed"
    claim.assert_not_awaited()


async def test_a_card_answered_in_the_chat_is_refused_without_a_claim(
    mocker: pytest_mock.MockerFixture,
) -> None:
    session = mocker.MagicMock(messages=[_tagged("row-1", "c-1")])
    mocker.patch.object(
        card_decided, "get_chat_session", new_callable=AsyncMock, return_value=session
    )
    claim = mocker.patch.object(card_decided, "_claim", new_callable=AsyncMock)

    claimed = await claim_answer("u-1", "s-1", ["c-1"], owner="row-9", attempt="a")
    assert claimed.result == "answered"
    assert claimed.answered == {"c-1"}
    claim.assert_not_awaited()


async def test_an_unanswered_card_is_answered_by_whoever_claims_it(
    mocker: pytest_mock.MockerFixture,
) -> None:
    session = mocker.MagicMock(messages=[_tagged("row-1", "c-1")])
    mocker.patch.object(
        card_decided, "get_chat_session", new_callable=AsyncMock, return_value=session
    )
    taken = card_decided.CardClaim(result="claimed")
    claim = mocker.patch.object(
        card_decided, "_claim", new_callable=AsyncMock, return_value=taken
    )

    assert await claim_answer("u-1", "s-1", ["c-2"], owner="row-9", attempt="a") is (
        taken
    )
    claim.assert_awaited_once_with("s-1", ["c-2"], "row-9", "a")


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
    claimed = await claim_answer("u-1", "s-1", ["c-1"], owner="row-9", attempt="a")
    assert claimed.result == "held"
