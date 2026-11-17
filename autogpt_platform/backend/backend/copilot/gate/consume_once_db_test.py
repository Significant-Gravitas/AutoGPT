"""One approval runs one call, however the readers of its card interleave.

Every case runs on the real review store against Postgres, through the
harness in ``consume_once_harness_test``. The seeded case checks that no card
still waiting for the user went missing between steps, and that no more calls
ran than the user approved.
"""

import asyncio
import random
from unittest.mock import AsyncMock, patch

import pytest
from prisma.enums import ReviewStatus

from backend.copilot.gate import check_action, held
from backend.copilot.gate import review as review_store
from backend.copilot.gate import spent_approval
from backend.copilot.gate.consume_once_harness_test import (
    ARGS,
    TOOL,
    AnHourLater,
    PostStub,
    deletes_fail,
    new_scene,
    resolved,
    scripted,
)
from backend.copilot.pending_messages import PendingMessage

_SEEDS = range(40)


@pytest.fixture
def gate_on():
    with patch("backend.copilot.gate.is_feature_enabled", AsyncMock(return_value=True)):
        yield


@pytest.fixture
def post_tool():
    tool = PostStub()
    with patch("backend.copilot.tools.get_tool", return_value=tool):
        yield tool


@pytest.mark.asyncio(loop_scope="session")
async def test_a_stale_approval_never_takes_the_card_an_identical_call_opened_after(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """A and B read one approval; B spends it and runs; C, the same call
    again, opens a fresh card under the same id. A's spend must miss C's card:
    A refuses, and C's card waits for the user."""
    scene = await new_scene(test_user_id, post_tool)
    await scene.approve()
    first = await scene.interleaving.card()
    assert first is not None

    with scene.stepping():
        outputs = await scene.interleaving.run(
            {"A": scene.caller(1), "B": scene.caller(2), "C": scene.caller(3)},
            scripted([("A", "consume"), ("B", None), ("C", None), ("A", None)]),
        )

    assert post_tool.runs == [ARGS]
    assert "already used" in outputs["A"]
    assert scene.review_id in outputs["C"]
    card = await scene.interleaving.card()
    assert card is not None and card.status == ReviewStatus.WAITING
    assert card.createdAt > first.createdAt
    assert scene.interleaving.lost == []


@pytest.mark.asyncio(loop_scope="session")
async def test_one_approval_runs_at_most_once_and_loses_no_card_in_any_order(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """Three identical calls, two folds and three answers, in seeded orders."""
    ran = 0
    for seed in _SEEDS:
        rng = random.Random(seed)
        post_tool.runs.clear()
        scene = await new_scene(test_user_id, post_tool)
        verdicts = [
            rng.choice([ReviewStatus.APPROVED] * 3 + [ReviewStatus.REJECTED])
            for _ in range(3)
        ]
        actors = {f"call-{n}": scene.caller(n) for n in (1, 2, 3)}
        actors |= {"fold-1": scene.fold(), "fold-2": scene.fold()}
        actors["user"] = scene.user(verdicts)

        with scene.stepping():
            await scene.interleaving.run(
                actors, lambda parked: rng.choice(sorted(parked))
            )

        assert scene.interleaving.lost == [], f"seed {seed}"
        assert len(post_tool.runs) <= scene.approvals, f"seed {seed}"
        ran += len(post_tool.runs)
    assert ran, "no seed ran the call at all"


@pytest.mark.asyncio(loop_scope="session")
async def test_a_fold_with_the_gate_off_spends_the_approval_before_it_runs(
    setup_test_user, test_user_id, gate_on, post_tool
):
    scene = await new_scene(test_user_id, post_tool)
    await scene.approve()

    with patch(
        "backend.copilot.gate.is_feature_enabled", AsyncMock(return_value=False)
    ):
        [delivered] = await resolved(test_user_id, scene.session)

    assert post_tool.runs == [ARGS]
    assert post_tool.card_at_run == [False]
    assert delivered.metadata["held_call"]["outcome"] == "approved"


@pytest.mark.parametrize("gate", [True, False])
@pytest.mark.asyncio(loop_scope="session")
async def test_a_fold_that_cannot_spend_the_approval_runs_nothing(
    setup_test_user, test_user_id, gate_on, post_tool, gate
):
    """A spend that fails runs nothing. Its card may be spent or may still
    stand, so the chat is asked to propose the call again rather than the
    call being held. Here the card stands, and the call proposed again runs
    on it, once."""
    scene = await new_scene(test_user_id, post_tool)
    await scene.approve()
    gate_state = patch(
        "backend.copilot.gate.is_feature_enabled", AsyncMock(return_value=gate)
    )

    with deletes_fail(), gate_state:
        [delivered] = await resolved(test_user_id, scene.session)

    assert post_tool.runs == []
    assert delivered.metadata["held_call"]["outcome"] == "closed"
    assert "Nothing ran" in delivered.content
    card = await scene.interleaving.card()
    assert card is not None and card.status == ReviewStatus.APPROVED
    assert await held.answered(test_user_id, scene.session.session_id) == []

    with gate_state:
        await scene.caller(1)()
    assert post_tool.runs == [ARGS]


@pytest.mark.asyncio(loop_scope="session")
async def test_a_call_whose_approval_cannot_be_spent_runs_nothing(
    setup_test_user, test_user_id, gate_on, post_tool
):
    scene = await new_scene(test_user_id, post_tool)
    await scene.approve()

    with deletes_fail():
        output = await scene.caller(1)()

    assert post_tool.runs == []
    # The gate's own failure refusal, not a miss reported as "already used".
    assert "could not be checked" in output and "already used" not in output
    card = await scene.interleaving.card()
    assert card is not None and card.status == ReviewStatus.APPROVED


@pytest.mark.parametrize("then_reads_fail", [False, True])
@pytest.mark.asyncio(loop_scope="session")
async def test_a_spend_whose_answer_was_lost_says_nothing_ran(
    setup_test_user, test_user_id, gate_on, post_tool, then_reads_fail
):
    """The delete took the card, but its answer never came back. The call did
    not run, so the chat hears that and is asked to propose it again, not
    told that it may have run. Nor is the call held again when the card can't
    be read back: no later turn would pick up a held call whose card is gone,
    so it would never be answered."""
    scene = await new_scene(test_user_id, post_tool)
    await scene.approve()

    with deletes_fail(committed=True, then_reads_fail=then_reads_fail):
        [delivered] = await resolved(test_user_id, scene.session)

    assert post_tool.runs == []
    assert delivered.metadata["held_call"]["outcome"] == "closed"
    assert "Nothing ran" in delivered.content
    assert "may have run" not in delivered.content
    assert await scene.interleaving.card() is None
    assert await held.answered(test_user_id, scene.session.session_id) == []


@pytest.mark.asyncio(loop_scope="session")
async def test_a_card_opened_after_the_fold_spent_its_own_is_not_taken_for_it(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """The fold spends A's card and runs A. While it runs, C, the same call,
    opens a fresh card under the same id, and then A's result fails to
    deliver. Recovery must not read C's card as A's unspent approval: A may
    have run, and the held call under that id stays C's."""
    scene = await new_scene(test_user_id, post_tool)
    await scene.approve()
    first = await scene.interleaving.card()
    assert first is not None

    async def c_parks() -> None:
        await scene.park(tool_call_id="call-C")

    post_tool.during = c_parks
    with patch.object(held, "_card_copy", side_effect=RuntimeError("lost")):
        [delivered] = await resolved(test_user_id, scene.session)

    assert post_tool.runs == [ARGS]
    assert delivered.metadata["held_call"]["outcome"] == "unknown"
    card = await scene.interleaving.card()
    assert card is not None and card.status == ReviewStatus.WAITING
    assert card.createdAt > first.createdAt
    kept = await held._held(scene.session.session_id)
    assert kept[scene.review_id].tool_call_id == "call-C"


@pytest.mark.asyncio(loop_scope="session")
async def test_an_approval_that_lapses_while_its_fold_waits_runs_nothing(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """The fold reads the card while it is live, then waits at its status
    line past the hour. The gate that would have caught the lapse lets the
    fold's spent call through, so the fold checks again before it spends."""
    scene = await new_scene(test_user_id, post_tool)
    await scene.approve()
    delivered: list[PendingMessage] = []

    turn = held.resolve_answered(test_user_id, scene.session, delivered.append)
    await anext(turn)
    with patch.object(spent_approval, "datetime", AnHourLater):
        async for _status in turn:
            pass

    assert post_tool.runs == []
    [result] = delivered
    assert result.metadata["held_call"]["outcome"] == "expired"
    assert await scene.interleaving.card() is None


@pytest.mark.asyncio(loop_scope="session")
async def test_the_folds_pass_lets_its_one_call_through_once(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """A second identical call in the same run is gated as any other: with
    the approval spent, it asks again."""
    scene = await new_scene(test_user_id, post_tool)
    await scene.approve()
    sid = scene.session.session_id
    card = await review_store.find_review(scene.review_id, test_user_id, sid)
    assert card is not None and await review_store.consume(card, test_user_id)

    async def twice():
        scene.enter_turn()
        with spent_approval.running(scene.review_id, card):
            return [
                await check_action(TOOL, ARGS, test_user_id, scene.session)
                for _ in range(2)
            ]

    first, second = await asyncio.create_task(twice())

    assert first.allowed and first.approved
    assert not second.allowed and second.review_id == scene.review_id
