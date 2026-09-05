from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import prisma.models
import pytest

from backend.api.features.experts import credential_counts, experts_db
from backend.api.features.experts.models import Expert


@pytest.mark.asyncio
async def test_counts_live_credentials_in_one_owner_scoped_batch():
    experts = [
        prisma.models.Expert.model_construct(id="expert-1", ownerUserId="owner-1"),
        prisma.models.Expert.model_construct(id="expert-2", ownerUserId="owner-1"),
        prisma.models.Expert.model_construct(id="foreign", ownerUserId="owner-2"),
    ]
    grant_query = AsyncMock(
        return_value=[
            SimpleNamespace(expertId="expert-1", credentialId="live-1"),
            SimpleNamespace(expertId="expert-1", credentialId="deleted"),
            SimpleNamespace(expertId="expert-2", credentialId="live-1"),
            SimpleNamespace(expertId="expert-2", credentialId="live-2"),
        ]
    )
    with (
        patch.object(credential_counts, "_seed_if_needed", new=AsyncMock()) as seed,
        patch.object(
            credential_counts,
            "_user_credentials",
            new=AsyncMock(
                return_value=[
                    SimpleNamespace(id="live-1"),
                    SimpleNamespace(id="live-2"),
                ]
            ),
        ) as credentials,
        patch.object(
            prisma.models.ExpertCredential,
            "prisma",
            return_value=SimpleNamespace(find_many=grant_query),
        ),
    ):
        counts = await credential_counts.count_expert_credentials("owner-1", experts)

    assert counts == {"expert-1": 1, "expert-2": 2}
    assert seed.await_count == 2
    for call in seed.await_args_list:
        assert call.args[0] == "owner-1"
        assert call.args[1].ownerUserId == "owner-1"
    credentials.assert_awaited_once_with("owner-1")
    grant_query.assert_awaited_once_with(
        where={"expertId": {"in": ["expert-1", "expert-2"]}}
    )


@pytest.mark.asyncio
async def test_empty_roster_skips_credential_reads():
    with patch.object(
        credential_counts, "_user_credentials", new=AsyncMock()
    ) as credentials:
        assert await credential_counts.count_expert_credentials("owner-1", []) == {}
    credentials.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("with_metrics", [True, False])
async def test_roster_includes_counts_only_when_metrics_are_requested(with_metrics):
    row = prisma.models.Expert.model_construct(id="expert-1", ownerUserId="owner-1")
    client = SimpleNamespace(find_many=AsyncMock(return_value=[row]))
    with (
        patch.object(prisma.models.Expert, "prisma", return_value=client),
        patch.object(experts_db, "_latest_runs", new=AsyncMock(return_value={})),
        patch.object(experts_db, "_weekly_spends", new=AsyncMock(return_value={})),
        patch.object(
            experts_db, "_to_model", return_value=Expert.model_construct(id=row.id)
        ),
        patch.object(
            experts_db,
            "count_expert_credentials",
            new=AsyncMock(return_value={row.id: 2}),
        ) as counts,
    ):
        roster = await experts_db.list_experts("owner-1", with_metrics=with_metrics)

    assert client.find_many.await_args.kwargs["where"]["ownerUserId"] == "owner-1"
    assert roster[0].credential_count == (2 if with_metrics else 0)
    if with_metrics:
        counts.assert_awaited_once_with("owner-1", [row])
    else:
        counts.assert_not_awaited()
