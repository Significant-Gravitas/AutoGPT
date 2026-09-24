"""A held call names what its ids point at, read from Postgres as its owner."""

import uuid

import pytest
from prisma.models import PendingHumanReview

from backend.api.features.library.db import create_folder
from backend.copilot.gate import review as review_store
from backend.copilot.gate.references import Reference, resolve_references
from backend.copilot.model import ChatSession, upsert_chat_session


@pytest.mark.asyncio(loop_scope="session")
async def test_a_held_delete_names_the_folder_in_the_stored_card(
    setup_test_user, test_user_id
):
    name = f"Q3 reports {uuid.uuid4().hex[:6]}"
    folder = await create_folder(test_user_id, name)
    session = await upsert_chat_session(
        ChatSession.new(user_id=test_user_id, dry_run=False)
    )
    args = {"folder_id": folder.id}
    review_id = review_store.review_id_for(
        session.session_id, test_user_id, "delete_folder", args
    )

    headline = await review_store.open_review(
        review_id, test_user_id, session, "delete_folder", args, "needs you"
    )

    assert headline is not None and headline.text == f"Delete a folder “{name}”"
    row = await PendingHumanReview.prisma().find_unique(where={"nodeExecId": review_id})
    assert row is not None
    stored = review_store.GateReviewPayload.model_validate(row.payload)
    assert stored.references == [
        Reference(
            key="folder_id",
            entity="library_folder",
            id=folder.id,
            name=name,
            href=f"/library?folder={folder.id}",
        )
    ]


@pytest.mark.asyncio(loop_scope="session")
async def test_another_users_folder_is_never_named(
    setup_test_user, test_user_id, setup_admin_user, admin_user_id
):
    theirs = await create_folder(admin_user_id, f"Private {uuid.uuid4().hex[:6]}")
    session = ChatSession.new(user_id=test_user_id, dry_run=False)

    [ref] = await resolve_references(
        "delete_folder", {"folder_id": theirs.id}, test_user_id, session
    )

    assert (ref.id, ref.name, ref.href) == (theirs.id, None, None)
