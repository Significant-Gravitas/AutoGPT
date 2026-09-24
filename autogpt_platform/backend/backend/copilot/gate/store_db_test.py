"""The gate's state against Postgres: the approval mutex and the chat's mode."""

import asyncio

import pytest
from prisma.enums import ReviewStatus
from prisma.models import PendingHumanReview

from backend.copilot.gate import resolve_mode
from backend.copilot.gate import review as review_store
from backend.copilot.model import (
    ChatSession,
    get_chat_session,
    update_session_autopilot_mode,
    upsert_chat_session,
)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_parked_call_is_the_chats_review_and_its_approval_is_found(
    setup_test_user, test_user_id
):
    """The card is read from the chat's queue, and the click reaches the retry."""
    session = await upsert_chat_session(
        ChatSession.new(user_id=test_user_id, dry_run=False)
    )
    args = {"command": "rm report.md"}
    review_id = review_store.review_id_for(
        session.session_id, test_user_id, "bash_exec", args
    )
    assert await review_store.open_review(
        review_id, test_user_id, session, "bash_exec", args, "needs you"
    )

    assert await review_store.has_open_review(test_user_id, session.session_id)
    await PendingHumanReview.prisma().update(
        where={"nodeExecId": review_id}, data={"status": ReviewStatus.APPROVED}
    )
    assert (
        await review_store.find_decision(review_id, test_user_id, session.session_id)
        == ReviewStatus.APPROVED
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_four_concurrent_consumes_of_one_approval_run_once(
    setup_test_user, test_user_id
):
    """Parallel tool dispatch is deliberate, so the delete is the only mutex."""
    session = await upsert_chat_session(
        ChatSession.new(user_id=test_user_id, dry_run=False)
    )
    args = {"command": "ls"}
    review_id = review_store.review_id_for(
        session.session_id, test_user_id, "bash_exec", args
    )
    assert await review_store.open_review(
        review_id, test_user_id, session, "bash_exec", args, "needs you"
    )
    await PendingHumanReview.prisma().update(
        where={"nodeExecId": review_id}, data={"status": ReviewStatus.APPROVED}
    )

    results = await asyncio.gather(
        *(review_store.consume(review_id, test_user_id) for _ in range(4))
    )

    assert sorted(results) == [False, False, False, True]


@pytest.mark.asyncio(loop_scope="session")
async def test_a_set_mode_persists_and_leaves_the_rest_of_the_metadata(
    setup_test_user, test_user_id
):
    session = await upsert_chat_session(
        ChatSession.new(user_id=test_user_id, dry_run=False)
    )
    assert resolve_mode(session) == "auto"

    assert await update_session_autopilot_mode(
        session.session_id, test_user_id, "ask_first"
    )

    reloaded = await get_chat_session(session.session_id, test_user_id)
    assert reloaded is not None
    assert resolve_mode(reloaded) == "ask_first"
    assert reloaded.metadata.origin == "interactive"


@pytest.mark.asyncio(loop_scope="session")
async def test_another_users_session_cannot_be_moved(setup_test_user, test_user_id):
    session = await upsert_chat_session(
        ChatSession.new(user_id=test_user_id, dry_run=False)
    )
    assert not await update_session_autopilot_mode(
        session.session_id, "someone-else", "unsupervised"
    )
