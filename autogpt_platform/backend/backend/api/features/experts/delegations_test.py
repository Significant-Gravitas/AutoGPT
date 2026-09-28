"""The delegation list, read from real threads, costs, files and cards."""

import json
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from prisma.models import ChatMessage, ChatSession
from prisma.models import Expert as PrismaExpert
from prisma.models import (
    PendingHumanReview,
    PlatformCostLog,
    User,
    UserWorkspace,
    UserWorkspaceFile,
)
from pytest_snapshot.plugin import Snapshot

from backend.api.features.experts.delegations import brief_of, list_delegations
from backend.copilot.constants import COPILOT_ERROR_PREFIX
from backend.copilot.delegation_list_db import HELD_HANDOFF_PREFIX
from backend.util.json import SafeJson

_T0 = datetime(2026, 9, 28, 10, 41, tzinfo=UTC)


def test_the_brief_drops_the_hand_off_preamble():
    message = (
        "[Delegated task from Otto, a teammate on this user's team — not the "
        "user.]\n\n[Context: Q4 launch]\n\nDraft the onboarding PRD.\nKeep it short."
    )
    assert brief_of(message) == "Draft the onboarding PRD.\nKeep it short."
    assert brief_of("No preamble here") == "No preamble here"
    assert brief_of(None) == ""


async def _thread(
    user_id: str,
    parent: str,
    expert_id: str,
    minutes: int,
    replies: list[tuple[str, str]],
    **meta,
) -> str:
    sid = str(uuid.uuid4())
    await ChatSession.prisma().create(
        data={
            "id": sid,
            "userId": user_id,
            "expertId": expert_id,
            "credentials": SafeJson({}),
            "createdAt": _T0 + timedelta(minutes=minutes),
            "chatStatus": meta.pop("chat_status", "idle"),
            "metadata": SafeJson(
                {
                    "delegated_by_session_id": parent,
                    "delegated_by_expert_id": None,
                    **meta,
                }
            ),
        }
    )
    for seq, (role, content) in enumerate(replies):
        await ChatMessage.prisma().create(
            data={
                "sessionId": sid,
                "role": role,
                "content": content,
                "sequence": seq,
                "createdAt": _T0 + timedelta(minutes=minutes + seq + 1),
            }
        )
    return sid


@pytest.fixture
async def team():
    user_id = str(uuid.uuid4())
    await User.prisma().create(
        data={
            "id": user_id,
            "email": f"delegations-{user_id}@example.com",
            "topUpConfig": SafeJson({}),
            "timezone": "UTC",
        }
    )
    expert = await PrismaExpert.prisma().create(
        data={
            "id": f"expert-{user_id[:8]}",
            "ownerUserId": user_id,
            "name": "Alex",
            "role": "Product Manager",
            "identity": "",
            "color": "violet",
        }
    )
    otto = await ChatSession.prisma().create(
        data={"userId": user_id, "credentials": SafeJson({}), "metadata": SafeJson({})}
    )
    yield user_id, expert.id, otto.id
    await PendingHumanReview.prisma().delete_many(where={"userId": user_id})
    await PlatformCostLog.prisma().delete_many(where={"userId": user_id})
    await UserWorkspace.prisma().delete_many(where={"userId": user_id})
    await ChatSession.prisma().delete_many(where={"userId": user_id})
    await PrismaExpert.prisma().delete_many(where={"ownerUserId": user_id})
    await User.prisma().delete(where={"id": user_id})


async def _seed(user_id: str, expert_id: str, otto: str) -> dict[str, str]:
    brief = "[Delegated task from Otto.]\n\n"
    done = await _thread(
        user_id,
        otto,
        expert_id,
        0,
        [("user", brief + "Draft the PRD"), ("assistant", "Here it is.")],
    )
    failed = await _thread(
        user_id,
        otto,
        expert_id,
        10,
        [
            ("user", brief + "Pull Q3 numbers"),
            ("assistant", f"{COPILOT_ERROR_PREFIX} Rate limited"),
        ],
    )
    stopped = await _thread(
        user_id,
        otto,
        expert_id,
        20,
        [
            ("user", brief + "Email Dana"),
            ("assistant", f"{COPILOT_ERROR_PREFIX} Operation cancelled"),
        ],
    )
    asking = await _thread(
        user_id,
        otto,
        expert_id,
        30,
        [("user", brief + "Plan the launch"), ("assistant", "Which release?")],
        pending_question={
            "text": "Which release?",
            "asked_at": _T0.isoformat(),
            "options": ["Q4", "December"],
        },
    )
    working = await _thread(
        user_id,
        otto,
        expert_id,
        40,
        [("user", brief + "Write the FAQ")],
        chat_status="running",
    )
    await PlatformCostLog.prisma().create(
        data={
            "userId": user_id,
            "chatSessionId": done,
            "provider": "anthropic",
            "costMicrodollars": 310_000,
        }
    )
    workspace = await UserWorkspace.prisma().create(data={"userId": user_id})
    await UserWorkspaceFile.prisma().create(
        data={
            "workspaceId": workspace.id,
            "name": "prd.md",
            "path": f"/sessions/{done}/prd.md",
            "storagePath": "x",
            "mimeType": "text/markdown",
            "sizeBytes": 10,
            "metadata": SafeJson({"origin": "agent-created"}),
        }
    )
    await PendingHumanReview.prisma().create(
        data={
            "nodeExecId": f"{HELD_HANDOFF_PREFIX}{uuid.uuid4().hex[:16]}",
            "userId": user_id,
            "chatSessionId": otto,
            "status": "WAITING",
            "createdAt": _T0 + timedelta(minutes=50),
            "payload": SafeJson(
                {
                    "tool": "delegate_to_expert",
                    "handoff": {
                        "expert_id": expert_id,
                        "expert_name": "Alex",
                        "expert_role": "Product Manager",
                        "brief": "Revamp onboarding",
                        "why": "Alex owns onboarding",
                    },
                }
            ),
        }
    )
    return {
        "done": done,
        "failed": failed,
        "stopped": stopped,
        "asking": asking,
        "working": working,
    }


def _stable(response, ids: dict[str, str], expert_id: str, otto: str) -> str:
    names = {v: k for k, v in ids.items()} | {expert_id: "<expert>", otto: "<otto>"}
    rows = [
        {
            **row,
            "sub_session_id": names.get(row["sub_session_id"], row["sub_session_id"]),
            "parent_session_id": names.get(row["parent_session_id"]),
            "review_id": "<review>" if row["review_id"] else None,
            "expert": {**row["expert"], "id": "<expert>"},
            # Running rows count up from now.
            "elapsed_seconds": (
                None if row["status"] == "running" else row["elapsed_seconds"]
            ),
        }
        for row in response.model_dump(mode="json")["delegations"]
    ]
    summary = response.model_dump(mode="json")["summary"]
    return json.dumps(
        {"delegations": rows, "summary": summary}, indent=2, sort_keys=True
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_the_list_reads_each_hand_off_as_it_stands(team, snapshot: Snapshot):
    user_id, expert_id, otto = team
    ids = await _seed(user_id, expert_id, otto)

    response = await list_delegations(user_id, parent_session_id=otto)

    assert [r.status for r in response.delegations] == [
        "proposed",
        "running",
        "needs_input",
        "cancelled",
        "failed",
        "completed",
    ]
    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(_stable(response, ids, expert_id, otto), "delegations_list")


@pytest.mark.asyncio(loop_scope="session")
async def test_filters_narrow_by_status_and_expert(team):
    user_id, expert_id, otto = team
    await _seed(user_id, expert_id, otto)

    asking = await list_delegations(user_id, status="needs_input")
    other = await list_delegations(user_id, expert_id="someone-else")

    assert [r.question_options for r in asking.delegations] == [["Q4", "December"]]
    assert other.delegations == []
