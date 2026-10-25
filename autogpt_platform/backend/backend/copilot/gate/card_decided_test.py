"""A proposal card is answered once: a second decision on it is refused,
whether it comes from a stale tab, a double click or two tabs at once."""

import asyncio
import uuid
from unittest.mock import AsyncMock

import pytest
import pytest_mock

from backend.copilot.gate import answers_in_flight, card_approval, card_decided
from backend.copilot.gate.card_decided import (
    claim_answer,
    decided_ids,
    forget_unsaved_answers,
)
from backend.copilot.model import ChatMessage
from backend.copilot.pending_messages import PendingMessage
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
async def test_a_dropped_message_takes_back_only_what_its_sender_holds() -> None:
    """Stop drops three queued decisions. c-1's sender holds its accepted
    claim, c-3's is still queueing it: both go, approvals and all. c-2 was
    answered meanwhile by another tab, which holds it now: that stays."""
    session_id = f"s-{uuid.uuid4()}"
    for cid in ("c-1", "c-3"):
        await card_approval.record_card_decisions(
            "u-1", session_id, _line(cid), by=f"queued-{cid}|a"
        )
        assert await _claim(session_id, [cid], f"queued-{cid}", "a") == "claimed"
    await card_decided.accept_claim(session_id, ["c-1"], "queued-c-1", "a")
    await card_approval.record_card_decisions(
        "u-1", session_id, _line("c-2"), by="tab-x|x"
    )
    assert await _claim(session_id, ["c-2"], "tab-x", "x") == "claimed"
    await card_decided.accept_claim(session_id, ["c-2"], "tab-x", "x")

    await forget_unsaved_answers(
        "u-1",
        session_id,
        [
            PendingMessage(content=_line(cid), card_owner=f"queued-{cid}")
            for cid in ("c-1", "c-2", "c-3")
        ],
    )

    for cid in ("c-1", "c-3"):
        assert await _claim(session_id, [cid], "tab-b") == "claimed"
        assert not await _approved(session_id, cid)
    assert await _claim(session_id, ["c-2"], "tab-b") == "answered"
    assert await _approved(session_id, "c-2")


@needs_redis
async def test_a_message_queued_without_its_sender_is_left_to_the_recovery() -> None:
    """An entry queued before messages named their sender: Stop cannot tell
    whose claim the card holds, so it takes nothing. A lost answer there is
    found and cleared by the next tab's claim instead."""
    session_id = f"s-{uuid.uuid4()}"
    await card_approval.record_card_decisions(
        "u-1", session_id, _line("c-1"), by="queued|a"
    )
    assert await _claim(session_id, ["c-1"], "queued", "a") == "claimed"
    await card_decided.accept_claim(session_id, ["c-1"], "queued", "a")

    await forget_unsaved_answers(
        "u-1", session_id, [PendingMessage(content=_line("c-1"))]
    )

    assert await _claim(session_id, ["c-1"], "tab-b") == "answered"
    assert await _approved(session_id, "c-1")


def _line(cid: str) -> str:
    return f"Approved: hire Maya (confirmation_id: {cid})."


async def _approved(session_id: str, cid: str) -> bool:
    return await card_approval.approved_on_card(
        "confirm_expert_change", {"confirmation_id": cid}, "u-1", session_id
    )


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
    # Long enough that the live checks below never outlast it on a slow
    # runner; the sleep then outlasts it.
    mocker.patch.object(card_decided, "_PENDING_LEASE_MS", 2000)
    session = f"s-{uuid.uuid4()}"
    assert await _claim(session, ["c-1", "c-2"], "tab-a") == "claimed"
    await card_decided.accept_claim(session, ["c-2"], "tab-a", "tab-a")
    assert await _claim(session, ["c-1"], "tab-b") == "pending"

    await asyncio.sleep(2.2)  # A is gone

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
        card_decided,
        "_saved_rows",
        new_callable=AsyncMock,
        return_value=session.messages,
    )
    claim = mocker.patch.object(card_decided, "_claim", new_callable=AsyncMock)

    replay = await claim_answer("u-1", "s-1", ["c-1"], owner="row-9", attempt="a")

    assert replay.result == "replayed"
    claim.assert_not_awaited()


@needs_redis
@pytest.mark.parametrize("on_its_way", [None, "queued", "drained", "turn"])
async def test_an_accepted_answer_that_was_lost_stops_holding_the_card(
    mocker: pytest_mock.MockerFixture, on_its_way: str | None
) -> None:
    """Tab A's decision was queued, so its claim was accepted, then lost
    before it was saved (its buffer expired, or a turn dropped it). The card
    must be answerable again, and the lost answer's approval gone. While the
    answer is still queued, drained for saving, or inside a running turn, it
    stands."""
    session_id = f"s-{uuid.uuid4()}"
    line = "Approved: hire Maya (confirmation_id: c-1)."
    mocker.patch.object(
        card_decided,
        "_saved_rows",
        new_callable=AsyncMock,
        return_value=[],
    )
    holding = [PendingMessage(content=line)]
    mocker.patch.object(
        answers_in_flight,
        "peek_pending_messages",
        new_callable=AsyncMock,
        return_value=holding if on_its_way == "queued" else [],
    )
    mocker.patch.object(
        answers_in_flight,
        "peek_pending_for_persist",
        new_callable=AsyncMock,
        return_value=holding if on_its_way == "drained" else [],
    )
    mocker.patch.object(
        answers_in_flight,
        "is_turn_in_flight",
        new_callable=AsyncMock,
        return_value=on_its_way == "turn",
    )
    await card_approval.record_card_decisions("u-1", session_id, line, by="tab-a")
    assert await _claim(session_id, ["c-1"], "tab-a") == "claimed"
    await card_decided.accept_claim(session_id, ["c-1"], "tab-a", "tab-a")

    other_tab = await claim_answer(
        "u-1", session_id, ["c-1"], owner="tab-b", attempt="b"
    )

    lost = on_its_way is None
    assert other_tab.result == ("claimed" if lost else "answered")
    assert await card_approval.approved_on_card(
        "confirm_expert_change", {"confirmation_id": "c-1"}, "u-1", session_id
    ) is (not lost)


async def test_a_card_answered_in_the_chat_is_refused_without_a_claim(
    mocker: pytest_mock.MockerFixture,
) -> None:
    session = mocker.MagicMock(messages=[_tagged("row-1", "c-1")])
    mocker.patch.object(
        card_decided,
        "_saved_rows",
        new_callable=AsyncMock,
        return_value=session.messages,
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
        card_decided,
        "_saved_rows",
        new_callable=AsyncMock,
        return_value=session.messages,
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
        "_saved_rows",
        new_callable=AsyncMock,
        return_value=None,  # the chat cannot be read
    )
    mocker.patch.object(
        card_decided, "get_redis_async", new_callable=AsyncMock, side_effect=OSError
    )

    # Nothing to give back either: the claim was never taken.
    claimed = await claim_answer("u-1", "s-1", ["c-1"], owner="row-9", attempt="a")
    assert claimed.result == "held"
