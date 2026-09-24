"""A held call's ids named for the card: the table's coverage, each shape of
lookup, and the budget that keeps a slow one from holding the card."""

import asyncio
import re
import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features.library.model import LibraryFolder
from backend.copilot.gate import references
from backend.copilot.gate.headline import _OBJECT_ID, gated_tools, headline_for
from backend.copilot.gate.references import (
    REFERENCES,
    Reference,
    resolve_references,
    wanted_references,
)
from backend.copilot.gate.review import open_review, review_payload
from backend.copilot.model import ChatSession
from backend.util.exceptions import NotFoundError

# The survey's definition of an id-shaped input: its name, or its description.
_ID_NAME = re.compile(r"(^|_)(ids?|uuids?)$|Ids?$")
_ID_WORD = re.compile(r"\b(id|ID|ids|IDs|uuid|UUID|identifier)\b")


def test_every_id_shaped_input_of_a_gated_tool_has_a_decided_row():
    """A new tool's id must not fall back to a raw uuid unnoticed."""
    from backend.copilot.tools import get_tool

    missing = []
    for tool_name in sorted(gated_tools()):
        tool = get_tool(tool_name)
        assert tool is not None, tool_name
        for key, spec in (tool.parameters.get("properties") or {}).items():
            shaped = _ID_NAME.search(key) or _ID_WORD.search(
                spec.get("description") or ""
            )
            if shaped and (tool_name, key) not in REFERENCES:
                missing.append(f"{tool_name}.{key}")
    assert missing == [], "add a row (or a None row) to gate/references.py"


def test_every_row_names_a_gated_tool_and_every_entity_a_resolver():
    assert {tool for tool, _ in REFERENCES} <= gated_tools()
    entities = {entity for entity in REFERENCES.values() if entity}
    assert entities == set(references._RESOLVERS)


def test_every_headline_id_is_one_the_table_resolves():
    for tool, key in _OBJECT_ID.items():
        assert REFERENCES.get((tool, key)), f"{tool}.{key}"


async def test_a_single_id_resolves_to_its_name_and_page():
    lib = _library(folders={"f-111": "Q3 reports"})
    with patch.object(references, "library_db", return_value=lib):
        [ref] = await resolve_references(
            "delete_folder", {"folder_id": "f-111"}, "user-1", _session()
        )

    assert ref == Reference(
        key="folder_id",
        entity="library_folder",
        id="f-111",
        name="Q3 reports",
        href="/library?folder=f-111",
    )
    lib.get_folder.assert_awaited_once_with("f-111", "user-1")


async def test_an_id_that_does_not_resolve_keeps_the_raw_id_and_no_link():
    """Not found and not owned look the same: the id, never a broken link."""
    lib = _library(folders={})
    with patch.object(references, "library_db", return_value=lib):
        [ref] = await resolve_references(
            "delete_folder", {"folder_id": "f-gone"}, "user-1", _session()
        )

    assert (ref.id, ref.name, ref.href) == ("f-gone", None, None)


async def test_a_list_resolves_each_element_up_to_the_cap():
    lib = _library(agents={f"a{i}": f"Agent {i}" for i in range(7) if i != 1})
    args = {"agent_ids": [f"a{i}" for i in range(7)], "folder_id": None}
    with patch.object(references, "library_db", return_value=lib):
        refs = await resolve_references(
            "move_agents_to_folder", args, "user-1", _session()
        )

    assert [(r.id, r.name) for r in refs] == [
        ("a0", "Agent 0"),
        ("a1", None),
        ("a2", "Agent 2"),
        ("a3", "Agent 3"),
        ("a4", "Agent 4"),
    ]
    assert refs[0].href == "/library/agents/a0"
    assert lib.get_library_agent.await_count == references.MAX_LISTED


async def test_one_slow_lookup_is_cut_at_its_own_budget_and_the_rest_still_resolve():
    """A hung query must neither hold the card nor cost the other names."""
    lib = _library(folders={"f-dest": "Archive"})
    lib.get_folder = AsyncMock(side_effect=_folder_or_hang({"f-dest": "Archive"}))
    args = {"folder_id": "f-slow", "target_parent_id": "f-dest"}
    with (
        patch.object(references, "library_db", return_value=lib),
        patch.object(references, "LOOKUP_SECONDS", 0.05),
        patch.object(references, "CARD_SECONDS", 1.0),
    ):
        refs = await resolve_references("move_folder", args, "user-1", _session())

    assert [(r.key, r.name) for r in refs] == [
        ("folder_id", None),
        ("target_parent_id", "Archive"),
    ]


async def test_the_card_as_a_whole_is_cut_at_its_budget():
    lib = _library(folders={})
    lib.get_folder = AsyncMock(side_effect=_folder_or_hang({}))
    with (
        patch.object(references, "library_db", return_value=lib),
        patch.object(references, "LOOKUP_SECONDS", 30.0),
        patch.object(references, "CARD_SECONDS", 0.05),
    ):
        started = time.monotonic()
        [ref] = await resolve_references(
            "delete_folder", {"folder_id": "f-slow"}, "user-1", _session()
        )

    assert time.monotonic() - started < 5
    assert (ref.id, ref.name) == ("f-slow", None)


def test_an_external_id_is_never_looked_up():
    assert (
        wanted_references("create_feature_request", {"existing_issue_id": "L-1"}) == []
    )
    assert wanted_references("memory_forget_confirm", {"uuids": ["u1", "u2"]}) == []


def test_the_headline_takes_the_resolved_name_as_its_object():
    folder = Reference(
        key="folder_id", entity="library_folder", id="f-1", name="Q3 reports"
    )
    headline = headline_for("delete_folder", {"folder_id": "f-1"}, [folder])

    assert headline.text == "Delete a folder “Q3 reports”"
    # The card does not list the folder a second time.
    assert headline.object_key == "folder_id"
    unresolved = folder.model_copy(update={"name": None})
    assert headline_for("delete_folder", {}, [unresolved]).text == "Delete a folder"


def test_a_name_the_call_carries_wins_over_the_resolved_one():
    """``update_folder.name`` is the NEW name, which is what the user approves."""
    old = Reference(key="folder_id", entity="library_folder", id="f-1", name="Old")
    headline = headline_for("update_folder", {"folder_id": "f-1", "name": "New"}, [old])
    assert headline.text == "Update folder “New”"


def test_the_payload_freezes_the_references_and_the_named_headline():
    ref = Reference(
        key="schedule_id",
        entity="schedule",
        id="sch-1",
        name="Daily digest",
        href="/library/followups",
    )
    payload = review_payload(
        "pause_schedule", {"schedule_id": "sch-1"}, references=[ref]
    )

    assert payload["references"] == [ref.model_dump()]
    assert payload["headline"] == {
        "ask": "Pause a schedule",
        "object": "Daily digest",
        "object_key": "schedule_id",
    }


async def test_holding_a_call_resolves_its_ids_into_the_stored_card():
    lib = _library(folders={"f-111": "Q3 reports"})
    reviews = MagicMock(get_or_create_human_review=AsyncMock())
    session = ChatSession.new(user_id="user-1", dry_run=False)
    with (
        patch.object(references, "library_db", return_value=lib),
        patch("backend.copilot.gate.review.review_db", return_value=reviews),
    ):
        headline = await open_review(
            "rid", "user-1", session, "delete_folder", {"folder_id": "f-111"}, ""
        )

    assert headline is not None and headline.text == "Delete a folder “Q3 reports”"
    stored = reviews.get_or_create_human_review.await_args.kwargs
    assert stored["message"] == "Delete a folder “Q3 reports”"
    assert stored["input_data"]["references"][0]["name"] == "Q3 reports"


@pytest.mark.parametrize(
    "owner, session_id, expected",
    [
        ("user-1", "s-mine", "Ada"),
        ("user-2", "s-mine", None),
        ("user-1", "s-other", None),
    ],
)
async def test_a_pending_team_change_is_named_only_in_its_own_chat(
    owner, session_id, expected
):
    from backend.copilot.tools.expert_proposal import ExpertChangeProposal
    from backend.copilot.tools.models import ExpertChangePreview

    proposal = ExpertChangeProposal(
        user_id=owner,
        session_id=session_id,
        preview=ExpertChangePreview(kind="hire", name="Ada"),
    )
    redis = MagicMock(get=AsyncMock(return_value=proposal.model_dump_json()))
    session = ChatSession.new(user_id="user-1", dry_run=False)
    session.session_id = "s-mine"
    with patch.object(references, "get_redis_async", AsyncMock(return_value=redis)):
        [ref] = await resolve_references(
            "confirm_expert_change", {"confirmation_id": "c-1"}, "user-1", session
        )

    assert ref.name == expected
    # Read only: a proposal it will not name is left for the tool to refuse.
    redis.delete.assert_not_called()


def _session() -> ChatSession:
    return ChatSession.new(user_id="user-1", dry_run=False)


def _library(
    folders: dict[str, str] | None = None, agents: dict[str, str] | None = None
):
    folders, agents = folders or {}, agents or {}

    async def get_folder(folder_id: str, user_id: str):
        if folder_id not in folders:
            raise NotFoundError(f"Folder #{folder_id} not found")
        return _folder(folder_id, user_id, folders[folder_id])

    async def get_library_agent(agent_id: str, user_id: str):
        if agent_id not in agents:
            raise NotFoundError(f"Library agent #{agent_id} not found")
        agent = MagicMock(id=agent_id)
        agent.name = agents[agent_id]
        return agent

    lib = MagicMock()
    lib.get_folder = AsyncMock(side_effect=get_folder)
    lib.get_library_agent = AsyncMock(side_effect=get_library_agent)
    return lib


def _folder_or_hang(folders: dict[str, str]):
    async def get_folder(folder_id: str, user_id: str):
        if folder_id not in folders:
            await asyncio.Event().wait()
        return _folder(folder_id, user_id, folders[folder_id])

    return get_folder


def _folder(folder_id: str, user_id: str, name: str) -> LibraryFolder:
    now = datetime.now(UTC)
    return LibraryFolder(
        id=folder_id, user_id=user_id, name=name, created_at=now, updated_at=now
    )
