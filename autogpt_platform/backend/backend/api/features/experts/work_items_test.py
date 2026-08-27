from datetime import datetime, timezone
from types import SimpleNamespace

import prisma.enums

from .work_items import _to_model


def test_work_item_link_targets_the_exact_team_history_row() -> None:
    now = datetime(2026, 8, 28, tzinfo=timezone.utc)
    row = SimpleNamespace(
        id="work item/1",
        expertId="expert/1",
        managerSessionId="manager-1",
        delegatedSessionId="delegated-1",
        projectPhase="Launch",
        taskTitle="Prepare launch plan",
        expectedDeliverable="A launch plan",
        deliverableMode="workspace_files",
        successCriteria=[],
        dependencies=[],
        sourceArtifacts=[],
        constraints=[],
        approvalBoundaries=[],
        estimateMinutes=30,
        progress=100,
        status=prisma.enums.ExpertWorkItemStatus.DELIVERED,
        result="Done",
        blocker=None,
        confidence=prisma.enums.ExpertWorkConfidence.VERIFIED,
        artifacts=[],
        createdAt=now,
        updatedAt=now,
        startedAt=now,
        completedAt=now,
    )

    item = _to_model(row)

    assert item.link == (
        "/team/expert%2F1?workItemId=work%20item%2F1#work-item-work%20item%2F1"
    )
