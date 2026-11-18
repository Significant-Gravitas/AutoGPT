"""One approval runs one call, however the readers of its card interleave.

Every case runs on the real review store against Postgres, through the
harness in ``consume_once_harness_test``. The seeded case checks that no card
still waiting for the user went missing between steps, and that no more calls
ran than the user approved.
"""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest
from prisma.enums import ReviewStatus

from backend.copilot.gate import chat_rules, check_action, held
from backend.copilot.gate import review as review_store
from backend.copilot.gate import spent_approval
from backend.copilot.gate.consume_once_harness_test import (
    ARGS,
    TOOL,
    AnHourLater,
    PostStub,
    SameMillisecond,
    faulty_reviews,
    new_scene,
    resolved,
    scripted,
    seeded_round,
)
from backend.copilot.pending_messages import PendingMessage

_SEEDS = range(80)


async def _held_under(scene) -> list[str]:
    """The tool calls held under the scene's review id, whatever their card."""
    calls = (await held._held(scene.session.session_id)).values()
    return [c.tool_call_id for c in calls if c.review_id == scene.review_id]


@pytest.fixture
def gate_on():
    with patch("backend.copilot.gate.is_feature_enabled", AsyncMock(return_value=True)):
        yield


@pytest.fixture
def post_tool():
    tool = PostStub()
    with patch("backend.copilot.tools.get_tool", return_value=tool):
        yield tool


@pytest.mark.parametrize(
    "c_card", [ReviewStatus.WAITING, ReviewStatus.APPROVED], ids=["waiting", "approved"]
)
@pytest.mark.asyncio(loop_scope="session")
async def test_a_stale_approval_never_takes_the_card_an_identical_call_opened_after(
    setup_test_user, test_user_id, gate_on, post_tool, c_card
):
    """A and B read one approval; B spends it and runs; C, the same call
    again, opens a fresh card under the same id. A's spend must miss C's card:
    A refuses, and C's card stays. Approved before A resumes, C's card
    matches A's read in everything but its creation time."""
    scene = await new_scene(test_user_id, post_tool)
    await scene.approve()
    first = await scene.interleaving.card()
    assert first is not None
    answers = [c_card] if c_card == ReviewStatus.APPROVED else []

    with scene.stepping():
        outputs = await scene.interleaving.run(
            {
                "A": scene.caller(1),
                "B": scene.caller(2),
                "C": scene.caller(3),
                "user": scene.user(answers),
            },
            scripted(
                [
                    ("A", "consume"),
                    ("B", None),
                    ("C", None),
                    ("user", None),
                    ("A", None),
                ]
            ),
        )

    assert post_tool.runs == [ARGS]
    assert "already used" in outputs["A"]
    assert scene.review_id in outputs["C"]
    card = await scene.interleaving.card()
    assert card is not None and card.status == c_card
    assert card.createdAt > first.createdAt
    assert scene.interleaving.lost == []


@pytest.mark.asyncio(loop_scope="session")
async def test_one_approval_runs_at_most_once_and_loses_no_card_in_any_order(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """Identical calls (each a replacement under the same review id once the
    card before it is spent), folds, a fold that fails before it spends, a
    restore from the card and the user's answers, in seeded orders; every
    third seed from a call held before held calls knew their card, every
    third with all parks in one instant. Each step is checked on its own: no
    waiting card goes missing, no approved card goes without its call
    running, an approved result goes only to the call whose card it spent,
    and no held call is removed except by the fold that listed it. Then later
    turns settle what is left: no approved call stays unrun, none ran twice."""
    ran = 0
    for seed in _SEEDS:
        found = await seeded_round(test_user_id, post_tool, seed)
        assert found == [], f"seed {seed}: {found}"
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

    with faulty_reviews(deletes="fail"), gate_state:
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

    with faulty_reviews(deletes="fail"):
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

    with faulty_reviews("commit_then_fail", reads_fail_after_delete=then_reads_fail):
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
    assert "call-C" in {c.tool_call_id for c in kept.values()}


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
    """The first call to reach the gate spends the fold's card and goes
    through; a second identical call in the same run is gated as any other:
    with the approval spent, it asks again."""
    scene = await new_scene(test_user_id, post_tool)
    await scene.approve()
    sid = scene.session.session_id
    card = await review_store.find_review(scene.review_id, test_user_id, sid)
    assert card is not None
    fold_pass = spent_approval.FoldPass(card)

    async def twice():
        scene.enter_turn()
        with spent_approval.running(scene.review_id, fold_pass):
            return [
                await check_action(TOOL, ARGS, test_user_id, scene.session)
                for _ in range(2)
            ]

    first, second = await asyncio.create_task(twice())

    assert first.allowed and first.approved
    assert fold_pass.use == "spent"
    assert not second.allowed and second.review_id == scene.review_id


@pytest.mark.asyncio(loop_scope="session")
async def test_a_rejection_whose_card_cannot_be_removed_still_lands(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """Removing a rejected card is housekeeping: when its delete fails, the
    call still refuses, the chat asks from then on, and the card is left for
    the next look."""
    scene = await new_scene(test_user_id, post_tool)
    await scene.user([ReviewStatus.REJECTED])()

    with faulty_reviews(deletes="fail"):
        output = await scene.caller(1)()

    assert post_tool.runs == []
    assert "declined" in output
    card = await scene.interleaving.card()
    assert card is not None and card.status == ReviewStatus.REJECTED
    rule = await chat_rules.rule_for(scene.session.session_id, TOOL, test_user_id, None)
    assert rule is not None and rule.rule == "ask"


# What recovery tells the chat when a fold fails, by the fold's spend state
# and what a lookup of the card then finds. The PR description has the table.


@pytest.mark.asyncio(loop_scope="session")
async def test_a_spent_call_that_fails_while_lookups_fail_says_it_may_have_run(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """Spent, lookup fails. The fold spent the card and the call ran, then
    failed; the card can't be read back. Held again, the call would never be
    answered, since its card is gone; the chat hears it may have run."""
    scene = await new_scene(test_user_id, post_tool)
    await scene.approve()

    with (
        faulty_reviews(reads_fail_after_delete=True),
        patch.object(held, "_tool_outcome", side_effect=RuntimeError("lost")),
    ):
        [delivered] = await resolved(test_user_id, scene.session)

    assert post_tool.runs == [ARGS]
    assert delivered.metadata["held_call"]["outcome"] == "unknown"
    assert "may have run" in delivered.content
    assert await scene.interleaving.card() is None
    assert await held.answered(test_user_id, scene.session.session_id) == []
    assert await _held_under(scene) == []


@pytest.mark.asyncio(loop_scope="session")
async def test_a_fold_that_fails_before_spending_while_lookups_fail_says_nothing_ran(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """Unspent, lookup fails. Nothing ran, but whether the card still stands
    is unknown, so the chat is asked to propose the call again rather than
    the call being held where no turn may ever pick it up."""
    scene = await new_scene(test_user_id, post_tool)
    await scene.approve()

    with faulty_reviews() as database:

        async def fails_before_spending(*_args):
            database.reads_fail = True
            raise RuntimeError("failed before the spend")

        with patch.object(held, "_run", fails_before_spending):
            [delivered] = await resolved(test_user_id, scene.session)

    assert post_tool.runs == []
    assert delivered.metadata["held_call"]["outcome"] == "closed"
    assert "Nothing ran" in delivered.content
    assert await _held_under(scene) == []
    card = await scene.interleaving.card()
    assert card is not None and card.status == ReviewStatus.APPROVED


@pytest.mark.asyncio(loop_scope="session")
async def test_a_fold_that_fails_before_spending_leaves_a_later_card_to_its_call(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """Unspent, lookup finds a later card. Before the fold spends A's card,
    B, the same call, spends it and runs, and C opens a fresh card; then the
    fold fails. Nothing ran for A, and C's card and held call stay C's."""
    scene = await new_scene(test_user_id, post_tool)
    await scene.approve()
    first = await scene.interleaving.card()
    assert first is not None

    async def b_runs_c_parks_then_fails(*_args):
        await scene.caller(2)()
        await scene.park(tool_call_id="call-C")
        raise RuntimeError("failed before the spend")

    with patch.object(held, "_run", b_runs_c_parks_then_fails):
        [delivered] = await resolved(test_user_id, scene.session)

    assert post_tool.runs == [ARGS]
    assert delivered.metadata["held_call"]["outcome"] == "closed"
    assert "Nothing ran" in delivered.content
    card = await scene.interleaving.card()
    assert card is not None and card.status == ReviewStatus.WAITING
    assert card.createdAt > first.createdAt
    kept = await held._held(scene.session.session_id)
    assert "call-C" in {c.tool_call_id for c in kept.values()}


@pytest.mark.asyncio(loop_scope="session")
async def test_a_call_refused_before_its_gate_spends_nothing_and_is_not_approved(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """The tool refuses the held call before its gate is reached (here the
    turn denies the tool by name). The spend sits at the gate, so the
    approval still stands, and the chat isn't told the call was approved."""
    scene = await new_scene(test_user_id, post_tool)
    await scene.approve()

    with patch("backend.copilot.tools.base.gate_denied", return_value=True):
        [delivered] = await resolved(test_user_id, scene.session)

    assert post_tool.runs == []
    assert delivered.metadata["held_call"]["outcome"] == "closed"
    card = await scene.interleaving.card()
    assert card is not None and card.status == ReviewStatus.APPROVED


@pytest.mark.asyncio(loop_scope="session")
async def test_an_approval_runs_the_call_held_for_its_own_card(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """P's card is spent by an identical call B, which runs, leaving P's held
    call behind with no card. C, the same call, then parks on a fresh card,
    which the user approves. The fold runs C's call on C's approval: P's held
    call belongs to a card that is gone, so it never runs on C's, and C's
    result reaches C."""
    scene = await new_scene(test_user_id, post_tool)
    await scene.approve()
    await scene.caller(2)()
    await scene.park(tool_call_id="call-C")
    await scene.approve()

    delivered = await resolved(test_user_id, scene.session)

    assert post_tool.runs == [ARGS, ARGS]
    ran = [m for m in delivered if m.metadata["held_call"]["outcome"] == "approved"]
    assert [m.metadata["held_call"]["tool_call_id"] for m in ran] == ["call-C"]
    assert await scene.interleaving.card() is None


@pytest.mark.asyncio(loop_scope="session")
async def test_a_stale_fold_never_claims_a_call_parked_after_it_listed_its_own(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """The fold lists P's held call on its approved card, then stops before
    claiming it. B, the same call, spends that card and runs; C parks on a
    fresh card. The fold's claim takes P's held call only, so C's stays, and
    C runs once its card is approved."""
    scene = await new_scene(test_user_id, post_tool)
    await scene.approve()

    with scene.stepping_the_held_store():
        await scene.interleaving.run(
            {"fold": scene.fold(), "B": scene.caller(2), "C": scene.caller(3)},
            scripted([("fold", "claim"), ("B", None), ("C", None), ("fold", None)]),
            scene.seen,
        )
    assert scene.interleaving.lost == []
    assert await _held_under(scene) == ["call-3"]

    await scene.approve()
    await scene.settle()
    assert post_tool.runs == [ARGS, ARGS]
    assert await scene.interleaving.card() is None


@pytest.mark.asyncio(loop_scope="session")
async def test_a_recovery_never_overwrites_a_call_parked_after_its_lookup(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """A fold fails before it spends and recovers: its lookup finds the card
    it read, so it holds the call again. Before that write, B spends the card
    and runs, and C parks on a fresh one. The re-hold goes under the old card
    and leaves C's held call as it was."""
    scene = await new_scene(test_user_id, post_tool)
    await scene.approve()

    with scene.stepping_the_held_store():
        await scene.interleaving.run(
            {
                "failing-fold": scene.fold(),
                "B": scene.caller(2),
                "C": scene.caller(3),
            },
            scripted(
                [
                    ("failing-fold", "remember(recover)"),
                    ("B", None),
                    ("C", None),
                    ("failing-fold", None),
                ]
            ),
            scene.seen,
        )

    assert scene.interleaving.lost == []
    assert sorted(await _held_under(scene)) == ["call-0", "call-3"]
    assert post_tool.runs == [ARGS]


@pytest.mark.asyncio(loop_scope="session")
async def test_a_fold_that_listed_a_call_before_its_card_was_replaced_runs_nothing(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """The fold lists P's held call, then stops. B, the same call, spends P's
    card and runs; C parks on a fresh card; the user approves it. When the
    fold reads the card again it finds C's, which isn't P's: P's card is gone,
    and C's approval waits for C's own held call."""
    scene = await new_scene(test_user_id, post_tool)
    await scene.approve()

    with scene.stepping_the_held_store(pause_after_listing=True):
        await scene.interleaving.run(
            {
                "fold": scene.fold(),
                "B": scene.caller(2),
                "C": scene.caller(3),
                "user": scene.user([ReviewStatus.APPROVED]),
            },
            scripted(
                [
                    ("fold", "listed"),
                    ("B", None),
                    ("C", None),
                    ("user", None),
                    ("fold", None),
                ]
            ),
            scene.seen,
        )

    assert scene.interleaving.lost == []
    assert post_tool.runs == [ARGS]
    [p] = scene.delivered
    assert p.metadata["held_call"]["tool_call_id"] == "call-0"
    assert p.metadata["held_call"]["outcome"] == "closed"
    await scene.settle()
    assert post_tool.runs == [ARGS, ARGS]


@pytest.mark.parametrize(
    "first", ["failing-fold", "restorer"], ids=["recovery-first", "restore-first"]
)
@pytest.mark.asyncio(loop_scope="session")
async def test_a_restore_never_replaces_a_held_call_with_its_cards_copy(
    setup_test_user, test_user_id, gate_on, post_tool, first
):
    """A long call's card keeps a clipped copy of its arguments, which can't
    run. A fold fails before it spends and holds the exact call again, while
    a restore that found nothing held (the fold had claimed it) writes the
    card's copy, before or after. Either way the exact call ends up held, and
    it runs."""
    long = {"text": "x" * 30_000}
    scene = await new_scene(test_user_id, post_tool, long)
    await scene.approve()
    second = "restorer" if first == "failing-fold" else "failing-fold"

    with scene.stepping_the_held_store():
        await scene.interleaving.run(
            {"failing-fold": scene.fold(), "restorer": scene.restorer()},
            scripted(
                [
                    ("failing-fold", "remember(recover)"),
                    ("restorer", "remember(restore)"),
                    (first, None),
                    (second, None),
                ]
            ),
            scene.seen,
        )
    await scene.settle()

    assert scene.interleaving.lost == []
    assert post_tool.runs == [long]


@pytest.mark.asyncio(loop_scope="session")
async def test_identical_calls_parked_at_once_leave_one_held_call_for_the_one_card(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """X and Y, the same call in one turn, both find no card and both store a
    held call before either opens the card; X's opens it. Y's held call would
    never be answered, so Y is told the call is already waiting, as it would
    be had it come a moment later, and the one approval runs X once."""
    scene = await new_scene(test_user_id, post_tool, parked=False)

    with scene.stepping_the_held_store():
        outputs = await scene.interleaving.run(
            {"X": scene.caller(1), "Y": scene.caller(2)},
            scripted(
                [
                    ("X", "open_review_row"),
                    ("Y", "open_review_row"),
                    ("X", None),
                    ("Y", None),
                ]
            ),
            scene.seen,
        )

    assert scene.interleaving.lost == []
    assert "already waiting" in outputs["Y"]
    assert await _held_under(scene) == ["call-1"]
    await scene.approve()
    await scene.settle()
    assert post_tool.runs == [ARGS]
    assert await scene.interleaving.card() is None


@pytest.mark.parametrize("gate", [True, False], ids=["gate-on", "gate-off"])
@pytest.mark.parametrize(
    "when",
    [datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC), pd.Timestamp("2026-01-02 03:04:05")],
    ids=["datetime", "pandas-timestamp"],
)
@pytest.mark.asyncio(loop_scope="session")
async def test_a_held_call_with_a_non_json_argument_spends_its_approval_once(
    setup_test_user, test_user_id, gate_on, post_tool, gate, when
):
    """A held call's arguments come back from Redis as JSON (a timestamp as
    ISO text), and its gate must still name the approval the fold spends.
    Missed with the gate off, the call would run without spending it, and a
    later identical call could run on it again; with the gate on, the user
    would be asked again."""
    scene = await new_scene(test_user_id, post_tool, {"text": "once", "when": when})
    await scene.approve()

    with patch("backend.copilot.gate.is_feature_enabled", AsyncMock(return_value=gate)):
        [delivered] = await resolved(test_user_id, scene.session)

    assert delivered.metadata["held_call"]["outcome"] == "approved"
    assert len(post_tool.runs) == 1
    assert await scene.interleaving.card() is None
    # The gate on again, the same call asks rather than riding the approval.
    output = await scene.caller(1)()
    assert len(post_tool.runs) == 1
    assert scene.review_id in output


@pytest.mark.asyncio(loop_scope="session")
async def test_a_gate_that_cannot_tell_whether_it_is_on_spends_nothing(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """Reading whether the gate is on fails at the fold's call: the gate
    refuses, and the approval it never got to use still stands."""
    scene = await new_scene(test_user_id, post_tool)
    await scene.approve()

    with patch(
        "backend.copilot.gate.gate_active",
        AsyncMock(side_effect=RuntimeError("flag unreadable")),
    ):
        [delivered] = await resolved(test_user_id, scene.session)

    assert post_tool.runs == []
    assert delivered.metadata["held_call"]["outcome"] == "closed"
    card = await scene.interleaving.card()
    assert card is not None and card.status == ReviewStatus.APPROVED


@pytest.mark.asyncio(loop_scope="session")
async def test_parks_in_one_instant_never_share_a_held_calls_identity(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """Every park in the same millisecond. P's card is spent by B, the same
    call, which runs, leaving P's held call behind; C then parks on a fresh
    card. C's held call is its own, so the approval of C's card runs C."""
    with patch.object(held, "datetime", SameMillisecond):
        scene = await new_scene(test_user_id, post_tool)
        await scene.approve()
        await scene.caller(2)()
        await scene.park(tool_call_id="call-C")
        await scene.approve()

        delivered = await resolved(test_user_id, scene.session)

    ran = [m for m in delivered if m.metadata["held_call"]["outcome"] == "approved"]
    assert [m.metadata["held_call"]["tool_call_id"] for m in ran] == ["call-C"]
    assert post_tool.runs == [ARGS, ARGS]


@pytest.mark.asyncio(loop_scope="session")
async def test_a_call_held_before_held_calls_knew_their_card_never_runs_on_a_later_one(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """P was held, and its card opened, before held calls knew their card. B,
    the same call, spends P's card and runs; C parks on a fresh card, which
    the user approves. P's held call pairs only with a card from before, so
    the approval of C's card runs C, not P."""
    scene = await new_scene(test_user_id, post_tool)
    await scene.make_legacy()
    await scene.approve()
    await scene.caller(2)()
    await scene.park(tool_call_id="call-C")
    await scene.approve()

    delivered = await resolved(test_user_id, scene.session)

    ran = [m for m in delivered if m.metadata["held_call"]["outcome"] == "approved"]
    assert [m.metadata["held_call"]["tool_call_id"] for m in ran] == ["call-C"]
    assert post_tool.runs == [ARGS, ARGS]


@pytest.mark.asyncio(loop_scope="session")
async def test_a_call_held_before_held_calls_knew_their_card_still_runs_on_its_own(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """A held call from before, with its own card from before still open,
    runs once that card is approved."""
    scene = await new_scene(test_user_id, post_tool)
    await scene.make_legacy()
    await scene.approve()

    [delivered] = await resolved(test_user_id, scene.session)

    assert delivered.metadata["held_call"]["outcome"] == "approved"
    assert post_tool.runs == [ARGS]
    assert await scene.interleaving.card() is None


@pytest.mark.parametrize("gate", [True, False], ids=["gate-on", "gate-off"])
@pytest.mark.asyncio(loop_scope="session")
async def test_a_held_call_whose_arguments_change_before_its_gate_leaves_no_approval(
    setup_test_user, test_user_id, gate_on, post_tool, gate
):
    """The replayed call reaches its gate under another review id (its tool
    changed the arguments on the way), so the fold's pass goes unused. With
    the gate off it runs; with it on it asks under its own card. Either way
    the card the fold read is dropped, so it never buys an identical call a
    second run."""
    scene = await new_scene(test_user_id, post_tool)
    await scene.approve()
    named = review_store.review_id_for

    def renamed(*args, **kwargs) -> str:
        return named(*args, **kwargs) + "-changed"

    with (
        patch.object(review_store, "review_id_for", renamed),
        patch("backend.copilot.gate.is_feature_enabled", AsyncMock(return_value=gate)),
    ):
        [delivered] = await resolved(test_user_id, scene.session)

    assert delivered.metadata["held_call"]["outcome"] == "closed"
    assert post_tool.runs == ([] if gate else [ARGS])
    assert await scene.interleaving.card() is None
    output = await scene.caller(2)()
    assert scene.review_id in output
    assert post_tool.runs == ([] if gate else [ARGS])


class _Unreadable:
    async def get_reviews_by_node_exec_ids(self, node_exec_ids, user_id):
        raise ConnectionError("database unreachable")


@pytest.mark.asyncio(loop_scope="session")
async def test_a_park_that_cannot_read_its_card_back_keeps_its_call(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """The park stored its call and opened its card, then couldn't read the
    card back. The call is kept, and told its card is open: it pairs only
    with a card of its own generation, so keeping it is safe, and dropping it
    (or telling it it's already waiting) would leave the card with nothing to
    run, or the chat told the wrong thing."""
    scene = await new_scene(test_user_id, post_tool, parked=False)

    with patch.object(held, "review_db", lambda: _Unreadable()):
        decision = await scene.park()

    assert decision.headline is not None
    assert await _held_under(scene) == ["call-0"]
    await scene.approve()
    [delivered] = await resolved(test_user_id, scene.session)
    assert delivered.metadata["held_call"]["outcome"] == "approved"
    assert post_tool.runs == [ARGS]


@pytest.mark.asyncio(loop_scope="session")
async def test_held_calls_no_card_can_answer_are_dropped_once_stale(
    setup_test_user, test_user_id, gate_on, post_tool
):
    """B, the same call, spends P's card, so P's held call can never be
    answered; C then waits on a fresh card. A later fold keeps P's while it is
    younger than an approval lasts (a park may still be opening its card),
    then drops it. C's, whose card still waits, stays however old."""
    scene = await new_scene(test_user_id, post_tool)
    await scene.approve()
    await scene.caller(2)()
    await scene.park(tool_call_id="call-C")
    assert sorted(await _held_under(scene)) == ["call-0", "call-C"]

    await resolved(test_user_id, scene.session)
    assert sorted(await _held_under(scene)) == ["call-0", "call-C"]

    with patch.object(held, "datetime", AnHourLater):
        await resolved(test_user_id, scene.session)
    assert await _held_under(scene) == ["call-C"]
    assert post_tool.runs == [ARGS]


def test_a_cards_rule_comes_from_its_call_not_a_restored_copy():
    """A restore's copy keeps no rule key; beside the exact call under the
    same review id, the exact call's rule is the one a rejection sets."""
    exact = held.HeldCall(
        review_id="r", tool_name=TOOL, tool_call_id="c", args={}, rule_key="subject"
    )
    copy = exact.model_copy(update={"rule_key": None, "generation": "other"})
    for order in ([exact, copy], [copy, exact]):
        by_id = held._by_review_id({str(i): call for i, call in enumerate(order)})
        assert by_id["r"].rule_key == "subject"
