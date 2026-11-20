"""Prisma responses for store and external API submission deletion tests."""

from unittest.mock import AsyncMock, Mock

import prisma.models
from prisma.enums import SubmissionStatus
from pytest_mock import MockerFixture

from . import db


def mock_submission_deletion(
    mocker: MockerFixture,
    *,
    owning_user_id: str | None,
    owning_org_id: str | None = None,
    status: SubmissionStatus = SubmissionStatus.PENDING,
) -> tuple[AsyncMock, AsyncMock]:
    tx = AsyncMock()
    mocker.patch.object(db, "transaction").return_value.__aenter__.return_value = tx
    tx.query_raw.return_value = (
        [Mock(id="sl-1", owningUserId=owning_user_id, owningOrgId=owning_org_id)]
        if owning_user_id is not None
        else []
    )
    versions = AsyncMock()
    versions.delete_many.return_value = int(status != SubmissionStatus.APPROVED)
    versions.count.return_value = 1
    mocker.patch.object(
        prisma.models.StoreListingVersion, "prisma", return_value=versions
    )
    return tx, versions
