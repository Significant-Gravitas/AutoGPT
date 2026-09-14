"""Real-storage regression for the whole learning loop.

Uses the real Postgres tables and the real workspace registry (local
storage under ``WORKSPACE_STORAGE_DIR``); only the reviewer model, budget,
flag, virus scanner, cost accounting, and memory link are mocked. Pins:

* two consecutive automated updates both publish (the second is not
  mistaken for a human edit — bytes hashed are bytes written);
* ``read_skill`` pins the exact version it loaded, before and after a
  restore, and a restore yields a new ready version whose file is readable;
* a same-second overwrite of the same workspace path does not collide on
  the soft-delete tombstone;
* the ledger, cursor, and inherited source provenance are correct.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from prisma.models import User

from backend.copilot import db as chats
from backend.copilot.dream.llm import CompletionUsage, StructuredCompletion
from backend.copilot.learning import nightly, publish
from backend.copilot.learning.chat_source import record_chat_turn
from backend.copilot.learning.owner_actions import apply_owner_edit, restore_version
from backend.copilot.learning.prompts import LearningProposal
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.tools.models import BashExecResponse
from backend.copilot.tools.skills import (
    ParsedSkill,
    ReadSkillResponse,
    ReadSkillTool,
    read_user_skill_markdown,
    render_skill_markdown,
    store_user_skill,
)
from backend.data import db
from backend.data import skill_learning as learning
from backend.data import skill_publication as publication
from backend.data import skill_reviews as reviews
from backend.data import skill_versions as versions
from backend.util.json import SafeJson
from backend.util.test import SpinTestServer
from backend.util.workspace import WorkspaceManager

logger = logging.getLogger(__name__)

BODY = """## Why
Check CSV inputs before importing.
## Trigger
Use when importing a CSV file.
## Prerequisites
A sample CSV and expected columns.
## Steps
1. Parse the sample as UTF-8.
2. Validate every row has the expected fields.
3. Report the validated row count before importing.
## Verification
The sample output reports three valid rows.
## Limits
Only the sample fixture was checked.
"""


async def _persist_turn(
    session: ChatSession, offset: int, request: str
) -> list[ChatMessage]:
    tool_id = str(uuid4())
    response = BashExecResponse(
        stdout="3 rows validated", stderr="", exit_code=0, message="ok"
    )
    messages = [
        ChatMessage(role="user", content=request, sequence=offset),
        ChatMessage(
            role="assistant",
            content="Checking the sample CSV.",
            sequence=offset + 1,
            tool_calls=[
                {
                    "id": tool_id,
                    "type": "function",
                    "function": {"name": "bash_exec", "arguments": "{}"},
                }
            ],
        ),
        ChatMessage(
            role="tool",
            content=response.model_dump_json(),
            tool_call_id=tool_id,
            sequence=offset + 2,
        ),
        ChatMessage(
            role="assistant",
            content="The sample passes the checks.",
            sequence=offset + 3,
        ),
    ]
    for message in messages:
        data = {
            "sessionId": session.session_id,
            "role": message.role,
            "content": message.content,
            "sequence": message.sequence,
        }
        if message.tool_call_id:
            data["toolCallId"] = message.tool_call_id
        if message.tool_calls:
            data["toolCalls"] = SafeJson(message.tool_calls)
        await db.prisma.chatmessage.create(data=data)
    session.messages.extend(messages)
    return messages


def _proposal(
    decision: str, body: str, supported: str, summary: str
) -> StructuredCompletion:
    return StructuredCompletion(
        value=LearningProposal(
            decision=decision,
            skill_name="csv-import-checks",
            description="Validate sample CSV rows before importing",
            triggers=["CSV import"],
            body=body,
            summary=summary,
            supported_by=[supported],
            verification="Validator exited successfully and reported three checked rows",
            private_values_replaced=True,
        ),
        usage=CompletionUsage(
            model="fixture-model", input_tokens=10, output_tokens=10, cost_usd=0
        ),
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_learn_load_update_restore_against_real_storage(server: SpinTestServer):
    user, expert = str(uuid4()), str(uuid4())
    await User.prisma().create(data={"id": user, "email": f"{user}@example.invalid"})
    try:
        await db.prisma.expert.create(
            data={
                "id": expert,
                "ownerUserId": user,
                "name": "CSV Expert",
                "role": "Data",
                "identity": "Validate CSV inputs.",
            }
        )
        session = ChatSession.new(user, dry_run=False, expert_id=expert)
        await chats.create_chat_session(session.session_id, user, expert_id=expert)
        reviewer = AsyncMock(
            side_effect=[
                _proposal("create", BODY, "msg:2", "Added sample CSV checks"),
                _proposal(
                    "update",
                    BODY.replace(
                        "2. Validate every row",
                        "2. Reject duplicate column names, then validate every row",
                    ),
                    "msg:6",
                    "Added duplicate column checks",
                ),
            ]
        )
        with patch.object(
            nightly, "is_feature_enabled", AsyncMock(return_value=True)
        ), patch.object(
            nightly, "check_dream_budget", AsyncMock(return_value=(True, None))
        ), patch.object(
            nightly, "review_evidence", reviewer
        ), patch.object(
            nightly, "record_review_cost", AsyncMock(return_value=0)
        ), patch(
            "backend.copilot.learning.dispositions.link_version_to_memory",
            AsyncMock(return_value=True),
        ), patch(
            "backend.util.workspace.scan_content_safe", AsyncMock()
        ), patch(
            "backend.copilot.tools.skills.is_skills_feature_enabled",
            AsyncMock(return_value=True),
        ):
            turn1 = await _persist_turn(session, 0, "Validate sample import rows.")
            source = await record_chat_turn(
                user, session, turn1, "Validate sample import rows."
            )
            first = await nightly.run_skill_learning_pass(user)
            assert first.applied == 1, first.model_dump()
            head = await versions.get_head(user, expert, "csv-import-checks")
            assert head is not None and head.current_version == 1
            v1_id = head.current_version_id

            loaded = await ReadSkillTool()._execute(
                user, session, name="csv-import-checks"
            )
            assert isinstance(loaded, ReadSkillResponse)
            assert loaded.version == 1 and loaded.version_id == v1_id
            assert loaded.origin == "saved_overnight"
            ledger = await reviews.get_review_for_revision(
                user, source.id, "3", nightly.POLICY_VERSION
            )
            assert ledger is not None and ledger.disposition == "applied"
            assert (
                await learning.get_source(user, source.id)
            ).processed_revision == "000000000003"

            turn2 = await _persist_turn(
                session, 4, "Also reject duplicate column names."
            )
            await record_chat_turn(
                user, session, turn2, "Also reject duplicate column names."
            )
            second = await nightly.run_skill_learning_pass(user)
            assert second.applied == 1 and second.dispositions == {
                "applied": 1
            }, second.model_dump()
            head = await versions.get_head(user, expert, "csv-import-checks")
            assert head.current_version == 2
            loaded2 = await ReadSkillTool()._execute(
                user, session, name="csv-import-checks"
            )
            assert isinstance(loaded2, ReadSkillResponse) and loaded2.version == 2
            assert "Reject duplicate column names" in loaded2.body

            restored = await restore_version(
                user_id=user,
                expert_id=expert,
                skill_name="csv-import-checks",
                version_id=v1_id,
                actor_user_id=user,
            )
            assert restored.status == "applied", restored.reason
            assert restored.version is not None and restored.version.version == 3
            loaded3 = await ReadSkillTool()._execute(
                user, session, name="csv-import-checks"
            )
            assert isinstance(loaded3, ReadSkillResponse)
            assert loaded3.version == 3 and loaded3.origin == "restored"
            assert "Reject duplicate column names" not in loaded3.body
            all_versions = await versions.list_versions(
                user, expert, "csv-import-checks"
            )
            assert [v.version for v in all_versions] == [3, 2, 1]
            assert all(v.state == "ready" for v in all_versions)
            assert (
                all_versions[0].sources == all_versions[2].sources
            )  # inherited provenance
            assert all_versions[0].restored_from_version_id == v1_id
            assert (
                await versions.get_head(user, expert, "csv-import-checks")
            ).auto_improve is False
            assert reviewer.await_count == 2
            assert await publication.list_pending_publications(user) == []
    finally:
        await User.prisma().delete_many(where={"id": user})


@pytest.mark.asyncio(loop_scope="session")
async def test_same_second_overwrites_do_not_collide_on_the_tombstone(
    server: SpinTestServer,
):
    user = str(uuid4())
    await User.prisma().create(data={"id": user, "email": f"{user}@example.invalid"})
    try:
        from backend.data.workspace import get_or_create_workspace

        workspace = await get_or_create_workspace(user)
        manager = WorkspaceManager(user, workspace.id, session_id=None)
        with patch("backend.util.workspace.scan_content_safe", AsyncMock()):
            for i in range(3):
                await manager.write_file(
                    content=f"v{i}".encode(),
                    filename="SKILL.md",
                    path="/skills/rapid/SKILL.md",
                    mime_type="text/markdown",
                    overwrite=True,
                )
        assert (await manager.read_file("/skills/rapid/SKILL.md")) == b"v2"
        rows = await db.prisma.userworkspacefile.find_many(
            where={"workspaceId": workspace.id, "isDeleted": True}
        )
        assert len(rows) == 2 and len({r.path for r in rows}) == 2
        started = datetime.now(timezone.utc)
        assert all(r.deletedAt is not None and r.deletedAt <= started for r in rows)
    finally:
        await User.prisma().delete_many(where={"id": user})


@pytest.mark.asyncio(loop_scope="session")
async def test_delayed_writers_never_overwrite_a_newer_version(server: SpinTestServer):
    """The two persisted lost-update interleavings, against real storage.

    1. v1 ordinary write -> automated v2 committed (pending write) -> human
       registry write v3 -> the v2 workspace write resumes. The resumed
       writer must abandon, and the file must keep the human correction.
    2. An editor who captured v3 as their base submits after v4 landed. The
       edit must be a conflict, and v4 must remain the file and the head.
    """
    user = str(uuid4())
    name = "concurrent-sample-checks"
    await User.prisma().create(data={"id": user, "email": f"{user}@example.invalid"})
    try:
        with patch("backend.util.workspace.scan_content_safe", AsyncMock()):
            await store_user_skill(
                user,
                name=name,
                description="Check sample inputs",
                body="## Steps\n1. Check the sample input.\n",
            )
            head = await versions.get_head(user, "personal", name)
            assert head is not None and head.current_version == 1
            rendered = render_skill_markdown(
                ParsedSkill(
                    name=name,
                    description="Check sample inputs",
                    body="## Steps\n1. Check sample inputs using the automated draft.\n",
                )
            )
            pending = await publication.commit_version_safe(
                user,
                head=head,
                draft=publication.VersionDraft(
                    content=rendered,
                    description="Check sample inputs",
                    origin="saved_overnight",
                    summary="Automated update",
                    base_version_id=head.current_version_id,
                ),
                expected_current_version=head.current_version,
            )
            assert pending.committed and pending.version is not None
            await store_user_skill(
                user,
                name=name,
                description="Check sample inputs",
                body="## Steps\n1. Preserve the NEWER HUMAN CORRECTION.\n",
                version_origin="edited",
                actor_user_id=user,
            )
            human_head = await versions.get_head(user, "personal", name)
            assert human_head is not None and human_head.current_version == 3

            late = await publish.write_committed_version(user, pending.version, None)
            assert late.status == "conflict", late.reason
            after = await versions.get_head(user, "personal", name)
            assert after is not None
            assert after.current_version_id == human_head.current_version_id
            text = await read_user_skill_markdown(user, name)
            assert text is not None and "NEWER HUMAN CORRECTION" in text
            abandoned = await versions.get_version(user, pending.version.id)
            assert abandoned is not None and abandoned.state == "stale"
            assert await publication.list_pending_publications(user) == []

            # Editor captured v3 as the base, then v4 landed from elsewhere.
            stale_base = human_head.current_version_id
            await store_user_skill(
                user,
                name=name,
                description="Check sample inputs",
                body="## Steps\n1. Keep the FOURTH version from another tab.\n",
                version_origin="edited",
                actor_user_id=user,
            )
            fourth = await versions.get_head(user, "personal", name)
            assert fourth is not None and fourth.current_version == 4
            stale_edit = await apply_owner_edit(
                user_id=user,
                expert_id=None,
                skill_name=name,
                description="Check sample inputs",
                body="## Steps\n1. A late draft typed over the third version.\n",
                triggers=[],
                keep_auto_improve=False,
                allowed_pattern_classes=[],
                expected_version_id=stale_base,
            )
            assert stale_edit.status == "conflict", stale_edit.reason
            text = await read_user_skill_markdown(user, name)
            assert text is not None and "FOURTH version" in text
            unchanged = await versions.get_head(user, "personal", name)
            assert unchanged is not None
            assert unchanged.current_version_id == fourth.current_version_id

            fresh_edit = await apply_owner_edit(
                user_id=user,
                expert_id=None,
                skill_name=name,
                description="Check sample inputs",
                body="## Steps\n1. A draft based on the fourth version.\n",
                triggers=[],
                keep_auto_improve=False,
                allowed_pattern_classes=[],
                expected_version_id=fourth.current_version_id,
            )
            assert fresh_edit.status == "applied", fresh_edit.reason
            assert fresh_edit.version is not None and fresh_edit.version.version == 5
    finally:
        await User.prisma().delete_many(where={"id": user})
