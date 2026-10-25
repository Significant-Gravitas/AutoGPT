"""A card answered long ago stays answered. A session loads only its newest
rows, and in a long chat the answer to an old card is older than that."""

import uuid
from unittest.mock import patch

import pytest
import pytest_asyncio
from prisma import Json
from prisma.models import ChatMessage as PrismaChatMessage
from prisma.models import User

from backend.copilot import db as copilot_db
from backend.copilot.gate import card_decided
from backend.copilot.gate.card_approval import card_decision_metadata
from backend.copilot.model import create_chat_session, get_chat_session
from backend.data.db import prisma as db_client
from backend.util.test import SpinTestServer

_LINE = "Approved: hire Maya (confirmation_id: c-1)."


@pytest.mark.parametrize(
    "claim_left", [False, True], ids=["claim-expired", "claim-still-accepted"]
)
@pytest.mark.asyncio(loop_scope="session")
async def test_an_answer_older_than_the_loaded_window_still_holds_the_card(
    claim_left: bool,
):
    """Whether its claim has expired or is still accepted, another tab's
    answer to the card is a second answer, and is refused."""
    user_id = str(uuid.uuid4())
    await User.prisma().create(
        data={"id": user_id, "email": f"card-decided-{user_id}@example.com"}
    )
    try:
        session_id = (await create_chat_session(user_id, dry_run=False)).session_id
        await _save_rows(session_id)
        if claim_left:
            assert (
                await card_decided._claim(session_id, ["c-1"], "tab-a", "a")
            ).result == "claimed"
            await card_decided.accept_claim(session_id, ["c-1"], "tab-a", "a")

        with patch.object(copilot_db, "MAX_LOADED_CHAT_MESSAGES", 3):
            window = await get_chat_session(session_id, user_id)
            assert window is not None
            assert _LINE not in [message.content for message in window.messages]

            other_tab = await card_decided.claim_answer(
                user_id, session_id, ["c-1"], owner="tab-b", attempt="b"
            )

        assert other_tab.result == "answered"
    finally:
        await User.prisma().delete(where={"id": user_id})


async def _save_rows(session_id: str) -> None:
    """The answer, then a page of newer rows that push it out of the window."""
    rows = [("user", _LINE, card_decision_metadata(_LINE))] + [
        ("assistant" if i % 2 == 0 else "user", f"Later message {i}", None)
        for i in range(6)
    ]
    for sequence, (role, content, metadata) in enumerate(rows):
        await PrismaChatMessage.prisma().create(
            data={
                "sessionId": session_id,
                "sequence": sequence,
                "role": role,
                "content": content,
                **({"metadata": Json(metadata)} if metadata else {}),
            }
        )


@pytest_asyncio.fixture(autouse=True)
async def absorb_a_stale_event_loop(server: SpinTestServer):
    """An earlier test can leave the shared Prisma client bound to a closed
    loop; only the first query on the new loop fails. Spend it here."""
    try:
        await db_client.execute_raw("SELECT 1")
    except RuntimeError as error:
        if "Event loop is closed" not in str(error):
            raise
