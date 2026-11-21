"""``seeded_expert_credential_ids``: an expert's grants read without the seed
that ``expert_allowed_credential_ids`` writes, so callers can read the grants
beside other reads and seed afterwards."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import prisma.models
import pytest
import pytest_mock

from backend.api.features.experts.credentials import seeded_expert_credential_ids
from backend.util.exceptions import ExpertNotFoundError


@pytest.fixture
def rows(mocker: pytest_mock.MockFixture) -> SimpleNamespace:
    """The expert row and its grant rows, with every write recorded."""
    expert = SimpleNamespace(find_first=AsyncMock(), update_many=AsyncMock())
    grants = SimpleNamespace(
        find_many=AsyncMock(
            return_value=[
                SimpleNamespace(credentialId="cred-a"),
                SimpleNamespace(credentialId="cred-b"),
            ]
        ),
        create_many=AsyncMock(),
    )
    mocker.patch.object(prisma.models.Expert, "prisma", return_value=expert)
    mocker.patch.object(prisma.models.ExpertCredential, "prisma", return_value=grants)
    return SimpleNamespace(expert=expert, grants=grants)


async def test_a_seeded_expert_reads_its_grants(rows: SimpleNamespace):
    rows.expert.find_first.return_value = SimpleNamespace(
        id="expert-1", credentialsSeededAt=datetime(2026, 1, 1, tzinfo=timezone.utc)
    )

    assert await seeded_expert_credential_ids("user-1", "expert-1") == [
        "cred-a",
        "cred-b",
    ]


async def test_an_unseeded_expert_reads_as_none_and_writes_nothing(
    rows: SimpleNamespace,
):
    rows.expert.find_first.return_value = SimpleNamespace(
        id="expert-1", credentialsSeededAt=None, Workflows=[]
    )

    assert await seeded_expert_credential_ids("user-1", "expert-1") is None
    rows.grants.find_many.assert_not_called()
    rows.grants.create_many.assert_not_called()
    rows.expert.update_many.assert_not_called()


async def test_an_expert_the_user_does_not_own_is_not_found(rows: SimpleNamespace):
    rows.expert.find_first.return_value = None

    with pytest.raises(ExpertNotFoundError):
        await seeded_expert_credential_ids("user-1", "expert-1")

    rows.grants.find_many.assert_not_called()
    assert rows.expert.find_first.await_args.kwargs["where"]["ownerUserId"] == "user-1"
