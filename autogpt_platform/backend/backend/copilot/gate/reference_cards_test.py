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

from backend.api.features.experts.models import ExpertRoutine, ExpertWorkflowLabel
from backend.api.features.library.model import LibraryFolder
from backend.copilot.gate import references
from backend.copilot.gate.review import review_id_for, review_payload
from backend.copilot.model import ChatSession, ChatSessionInfo
from backend.copilot.tools.expert_proposal import ExpertChangeProposal
from backend.copilot.tools.models import ExpertChangePreview
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
_FOLDERS = {"f-q3": "Q3 reports", "f-archive": "Archive", "f-finance": "Finance"}
_PARENTS = {"f-q3": "f-finance"}
_AGENTS = {
    "lib-digest": ("Morning digest", "Summarises overnight email and news at 7am."),
    "lib-triage": ("Inbox triage", "Labels and routes new support email."),
    "lib-notes": ("Meeting notes", "Turns call recordings into action items."),
    "lib-invoices": ("Invoice chaser", "Nudges clients about overdue invoices."),
    "lib-leads": ("Lead scorer", "Scores inbound leads against the ICP."),
    "lib-social": ("Social scheduler", "Queues the week's posts."),
}
_WHEN = datetime(2026, 9, 24, 14, 5, tzinfo=UTC)
_SESSION = "session-1"
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
    ("Edit agent", "edit_agent", {"agent_id": "g-digest"}),
    ("Update template", "update_preset", {"preset_id": "pre-weekly"}),
    ("Delete trigger", "delete_preset", {"preset_id": "pre-inbox"}),
    ("Schedule routine", "schedule_routine", {"routine_id": "rt-close"}),
    (
        "Install workflow",
        "install_expert_workflow",
        {"expert_id": "exp-ada", "store_listing_version_id": "slv-receipts"},
    ),
    ("Remove workflow", "remove_expert_workflow", {"workflow_id": "wf-receipts"}),
    ("Confirm team change", "confirm_expert_change", {"confirmation_id": "cf-1"}),
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
        patch.object(references, "store_db", return_value=_store()),
        patch.object(references, "get_redis_async", _redis),
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
    session = ChatSession.new(user_id=_USER, dry_run=False).model_copy(
        update={"session_id": _SESSION, "expert_id": "exp-ada"}
    )
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
            parent_id=_PARENTS.get(folder_id),
            agent_count=4,
            subfolder_count=1,
            created_at=now,
            updated_at=now,
        )

    async def get_library_agent(agent_id: str, user_id: str) -> MagicMock:
        if agent_id not in _AGENTS:
            raise NotFoundError(f"Library agent #{agent_id} not found")
        return _agent(agent_id)

    async def by_graph(user_id: str, graph_id: str) -> MagicMock | None:
        return _agent("lib-digest") if graph_id == "g-digest" else None

    async def get_preset(user_id: str, preset_id: str) -> MagicMock | None:
        return _presets().get(preset_id)

    lib = MagicMock()
    lib.get_folder = AsyncMock(side_effect=get_folder)
    lib.get_library_agent = AsyncMock(side_effect=get_library_agent)
    lib.get_library_agent_by_graph_id = AsyncMock(side_effect=by_graph)
    lib.get_preset = AsyncMock(side_effect=get_preset)
    return lib


def _agent(agent_id: str) -> MagicMock:
    agent = _named(agent_id, *_AGENTS[agent_id])
    agent.graph_version, agent.folder_name = 3, "Mornings"
    agent.last_run_at = _WHEN if agent_id == "lib-digest" else None
    return agent


def _preset(id: str, name: str, description: str, webhook_id: str | None) -> MagicMock:
    preset = _named(id, name, description)
    preset.graph_id, preset.webhook_id, preset.is_active = "g-digest", webhook_id, True
    return preset


def _experts() -> MagicMock:
    experts = MagicMock()
    ada = _named("tpl-ada", "Ada")
    ada.tagline, ada.job_title, ada.role = "Keeps the books balanced.", None, "Finance"
    experts.list_templates = AsyncMock(return_value=[ada])
    hired = _named("exp-ada", "Ada")
    hired.tagline, hired.job_title, hired.role = None, "Bookkeeper", "Finance"
    experts.get_expert = AsyncMock(return_value=hired)
    hired.bio, hired.is_archived = None, False
    routine = ExpertRoutine(
        id="rt-close",
        expert_id="exp-ada",
        title="Month-end close",
        prompt="Reconcile every account against the bank feed, flag anything "
        "over $500 that has no receipt, and draft the close summary for review.",
        crons=["0 9 1 * *"],
        enabled=True,
    )
    experts.list_routines = AsyncMock(return_value=[routine])
    experts.get_workflow_label = AsyncMock(
        return_value=ExpertWorkflowLabel(expert_id="exp-ada", name="Receipt matcher")
    )
    return experts


def _store() -> MagicMock:
    listing = MagicMock(
        agent_name="Receipt matcher",
        creator="ledgerly",
        slug="receipt-matcher",
        sub_heading="Matches card spend to emailed receipts.",
        description="",
        runs=1_204,
        rating=4.6,
    )
    return MagicMock(get_store_agent_by_version_id=AsyncMock(return_value=listing))


async def _redis() -> MagicMock:
    proposal = ExpertChangeProposal(
        user_id=_USER,
        session_id=_SESSION,
        preview=ExpertChangePreview(
            kind="hire",
            name="Grace",
            role="Engineering",
            job_title="Release manager",
            tagline="Ships on Thursdays.",
            template_id="tpl-grace",
        ),
        user_turn_watermark=1,
    )
    return MagicMock(get=AsyncMock(return_value=proposal.model_dump_json()))


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


def _presets() -> dict[str, MagicMock]:
    return {
        "pre-weekly": _preset(
            "pre-weekly", "Weekly digest", "The digest, but Mondays only.", None
        ),
        "pre-inbox": _preset(
            "pre-inbox", "New invoice email", "Runs when an invoice lands.", "wh-1"
        ),
    }
