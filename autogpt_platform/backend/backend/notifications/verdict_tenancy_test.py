from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import NotificationType, SubmissionStatus
from pydantic import ValidationError

from backend.api.features.store.db import is_verdict_notification_source_live
from backend.data.notifications import (
    NotificationEventModel,
    NotificationScope,
    VerdictData,
)


def _verdict() -> VerdictData:
    return VerdictData(
        outcome="approved",
        agent_name="Lead Scout",
        version=4,
        reviewer_name="Morgan Reyes",
        reviewed_at_label="5 August",
        comments="Clean submission.",
        store_url="https://example.com/store/lead-scout",
        share_url="https://example.com/store/lead-scout",
    )


def test_verdict_event_requires_one_exact_source_and_scope() -> None:
    with pytest.raises(ValidationError, match="reviewed version"):
        NotificationEventModel[VerdictData](
            user_id="user-1",
            type=NotificationType.VERDICT,
            data=_verdict(),
            authorization_scopes=[NotificationScope()],
        )

    with pytest.raises(ValidationError, match="one exact source scope"):
        NotificationEventModel[VerdictData](
            user_id="user-1",
            type=NotificationType.VERDICT,
            data=_verdict(),
            authorization_scopes=[
                NotificationScope(),
                NotificationScope(organization_id="org-1"),
            ],
            source_store_listing_version_id="version-1",
            expected_store_listing_status=SubmissionStatus.APPROVED,
            expected_store_listing_reviewed_at=datetime.now(tz=timezone.utc),
        )


REVIEWED_AT = datetime(2026, 8, 26, 12, tzinfo=timezone.utc)


def _version(
    *,
    organization_id: str = "org-1",
    status: SubmissionStatus = SubmissionStatus.APPROVED,
):
    graph = SimpleNamespace(
        id="graph-1",
        version=2,
        userId="user-1",
        organizationId=organization_id,
        teamId="team-1",
    )
    listing = SimpleNamespace(
        owningUserId="user-1",
        owningOrgId=organization_id,
        isDeleted=False,
    )
    return SimpleNamespace(
        id="version-1",
        isDeleted=False,
        organizationId=organization_id,
        teamId="team-1",
        agentGraphId="graph-1",
        agentGraphVersion=2,
        submissionStatus=status,
        reviewedAt=REVIEWED_AT,
        AgentGraph=graph,
        StoreListing=listing,
    )


@pytest.mark.asyncio
async def test_verdict_source_must_still_match_owner_and_scope() -> None:
    client = MagicMock()
    client.find_unique = AsyncMock(return_value=_version())
    with patch(
        "backend.api.features.store.db.prisma.models.StoreListingVersion.prisma",
        return_value=client,
    ):
        assert await is_verdict_notification_source_live(
            "user-1",
            "version-1",
            "org-1",
            "team-1",
            SubmissionStatus.APPROVED,
            REVIEWED_AT,
        )
        assert not await is_verdict_notification_source_live(
            "user-1",
            "version-1",
            "org-2",
            "team-1",
            SubmissionStatus.APPROVED,
            REVIEWED_AT,
        )


@pytest.mark.asyncio
async def test_verdict_source_rejects_a_later_review_revision() -> None:
    client = MagicMock()
    client.find_unique = AsyncMock(
        return_value=_version(status=SubmissionStatus.REJECTED)
    )
    with patch(
        "backend.api.features.store.db.prisma.models.StoreListingVersion.prisma",
        return_value=client,
    ):
        assert not await is_verdict_notification_source_live(
            "user-1",
            "version-1",
            "org-1",
            "team-1",
            SubmissionStatus.APPROVED,
            REVIEWED_AT,
        )
