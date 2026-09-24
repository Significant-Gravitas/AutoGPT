"""The id-carrying approval cards the frontend's stories and tests render are
built here, by the real resolvers and payload builder over stubbed reads, so
their references cannot drift from what the server stores.

Regenerate with ``UPDATE_CARD_FIXTURE=1``.
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import SecretStr

from backend.api.features.library.model import LibraryFolder
from backend.copilot.gate import references
from backend.copilot.gate.review import review_id_for, review_payload
from backend.copilot.model import ChatSession, ChatSessionInfo
from backend.data.model import APIKeyCredentials
from backend.data.workspace import WorkspaceFile
from backend.executor.scheduler import GraphExecutionJobInfo
from backend.util.exceptions import NotFoundError

FIXTURE = (
    Path(__file__).parents[4]
    / "frontend/src/app/(platform)/copilot/components/ApprovalQueue/__tests__"
    / "referenceCards.json"
)
_USER = "user-1"
_FOLDERS = {"f-q3": "Q3 reports", "f-archive": "Archive"}
_AGENTS = {
    "lib-digest": ("Morning digest", "Summarises overnight email and news at 7am."),
    "lib-triage": ("Inbox triage", "Labels and routes new support email."),
    "lib-notes": ("Meeting notes", "Turns call recordings into action items."),
    "lib-invoices": ("Invoice chaser", "Nudges clients about overdue invoices."),
    "lib-leads": ("Lead scorer", "Scores inbound leads against the ICP."),
    "lib-social": ("Social scheduler", "Queues the week's posts."),
}
_WHEN = datetime(2026, 9, 24, 14, 5, tzinfo=UTC)
# The call, as the model sends it, per story.
_CALLS: list[tuple[str, str, dict[str, Any]]] = [
    ("Delete folder", "delete_folder", {"folder_id": "f-q3"}),
    (
        "Move agents",
        "move_agents_to_folder",
        {
            "agent_ids": [
                "lib-digest",
                "lib-gone",
                "lib-triage",
                "lib-notes",
                "lib-invoices",
                "lib-leads",
                "lib-social",
            ],
            "folder_id": "f-archive",
        },
    ),
    ("Pause schedule", "pause_schedule", {"schedule_id": "sch-digest"}),
    ("Hire expert", "hire_expert", {"template_id": "tpl-ada", "name": "Ada"}),
    ("Message chat", "message_session", {"session_id": "s-q3", "message": "Done?"}),
    (
        "Grant credential",
        "grant_expert_credential",
        {"expert_id": "exp-ada", "credential_id": "cred-gh"},
    ),
    (
        "Delete file",
        "delete_workspace_file",
        {"file_id": "file-q3", "path": "/reports/q3-report.pdf"},
    ),
    ("Unresolved id", "delete_preset", {"preset_id": "3f0c9a2e-preset-gone"}),
]


async def build_cards() -> list[dict[str, Any]]:
    with (
        patch.object(references, "library_db", return_value=_library()),
        patch.object(references, "experts_db", return_value=_experts()),
        patch.object(references, "get_scheduler_client", return_value=_scheduler()),
        patch.object(references, "get_chat_session_metadata", _chat),
        patch.object(references, "IntegrationCredentialsManager", _credentials),
        patch.object(references, "get_workspace_manager", _workspace),
    ):
        cards = [await _card(*call) for call in _CALLS]
    return json.loads(json.dumps(cards, default=str))


async def test_the_frontend_reference_fixture_is_what_the_builder_makes():
    cards = await build_cards()
    if os.environ.get("UPDATE_CARD_FIXTURE"):
        FIXTURE.write_text(json.dumps(cards, indent=2, ensure_ascii=False) + "\n")
    assert (
        json.loads(FIXTURE.read_text()) == cards
    ), "the reference-card fixture is stale; rerun with UPDATE_CARD_FIXTURE=1"


async def _card(story: str, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    session = ChatSession.new(user_id=_USER, dry_run=False)
    refs = await references.resolve_references(tool, args, _USER, session)
    review_id = review_id_for("session-1", _USER, tool, args)
    return {
        "story": story,
        "review": {
            "node_exec_id": review_id,
            "node_id": review_id.split(":")[0],
            "user_id": _USER,
            "session_id": "s1",
            "graph_exec_id": None,
            "graph_id": None,
            "graph_version": None,
            "payload": review_payload(
                tool,
                args,
                reason="Ask First is on for this chat, so this action needs your approval.",
                mode="ask_first",
                tool_call_id=f"call-{tool}",
                turn=1,
                references=refs,
            ),
            "instructions": tool,
            "editable": False,
            "status": "WAITING",
        },
    }


def _library() -> MagicMock:
    async def get_folder(folder_id: str, user_id: str) -> LibraryFolder:
        if folder_id not in _FOLDERS:
            raise NotFoundError(f"Folder #{folder_id} not found")
        now = datetime(2026, 9, 24, tzinfo=UTC)
        return LibraryFolder(
            id=folder_id,
            user_id=user_id,
            name=_FOLDERS[folder_id],
            agent_count=4,
            subfolder_count=1,
            created_at=now,
            updated_at=now,
        )

    async def get_library_agent(agent_id: str, user_id: str) -> MagicMock:
        if agent_id not in _AGENTS:
            raise NotFoundError(f"Library agent #{agent_id} not found")
        return _named(agent_id, *_AGENTS[agent_id])

    async def by_graph(user_id: str, graph_id: str) -> MagicMock | None:
        return (
            _named("lib-digest", *_AGENTS["lib-digest"])
            if graph_id == "g-digest"
            else None
        )

    lib = MagicMock()
    lib.get_folder = AsyncMock(side_effect=get_folder)
    lib.get_library_agent = AsyncMock(side_effect=get_library_agent)
    lib.get_library_agent_by_graph_id = AsyncMock(side_effect=by_graph)
    lib.get_preset = AsyncMock(return_value=None)
    return lib


def _experts() -> MagicMock:
    experts = MagicMock()
    ada = _named("tpl-ada", "Ada")
    ada.tagline, ada.job_title, ada.role = "Keeps the books balanced.", None, "Finance"
    experts.list_templates = AsyncMock(return_value=[ada])
    hired = _named("exp-ada", "Ada")
    hired.tagline, hired.job_title, hired.role = None, "Bookkeeper", "Finance"
    experts.get_expert = AsyncMock(return_value=hired)
    return experts


def _scheduler() -> MagicMock:
    job = GraphExecutionJobInfo(
        id="sch-digest",
        name="Daily digest",
        next_run_time="2026-09-25T07:00:00+00:00",
        user_id=_USER,
        graph_id="g-digest",
        graph_version=1,
        cron="0 7 * * *",
        input_data={},
    )
    return MagicMock(get_execution_schedules=AsyncMock(return_value=[job]))


async def _chat(session_id: str, user_id: str) -> ChatSessionInfo:
    return ChatSessionInfo(
        session_id=session_id,
        user_id=user_id,
        title="Q3 planning",
        usage=[],
        started_at=_WHEN,
        updated_at=_WHEN,
    )


def _credentials() -> MagicMock:
    creds = APIKeyCredentials(
        id="cred-gh", provider="github", title="GitHub (work)", api_key=SecretStr("x")
    )
    return MagicMock(store=MagicMock(get_creds_by_id=AsyncMock(return_value=creds)))


async def _workspace(user_id: str, session_id: str) -> MagicMock:
    file = WorkspaceFile(
        id="file-q3",
        workspace_id="ws-1",
        created_at=_WHEN,
        updated_at=_WHEN,
        name="q3-report.pdf",
        path="/reports/q3-report.pdf",
        storage_path="ws-1/file-q3",
        mime_type="application/pdf",
        size_bytes=248_000,
        folder_id="wf-reports",
    )
    return MagicMock(get_file_info=AsyncMock(return_value=file))


def _named(id: str, name: str, description: str = "") -> MagicMock:
    thing = MagicMock(id=id, description=description)
    thing.name = name
    return thing
