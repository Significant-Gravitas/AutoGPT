from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import prisma.models
import pytest

from backend.api.features.experts import credential_counts, experts_db
from backend.api.features.experts.models import Expert


@pytest.mark.asyncio
async def test_lists_live_credential_providers_in_one_owner_scoped_batch():
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
                    SimpleNamespace(id="live-1", provider="github"),
                    SimpleNamespace(id="live-2", provider="notion"),
                ]
            ),
        ) as credentials,
        patch.object(
            prisma.models.ExpertCredential,
            "prisma",
            return_value=SimpleNamespace(find_many=grant_query),
        ),
    ):
        providers = await credential_counts.expert_credential_providers(
            "owner-1", experts
        )

    # One entry per live grant, so the count is the length and a provider
    # granted twice appears twice.
    assert providers == {"expert-1": ["github"], "expert-2": ["github", "notion"]}
    assert seed.await_count == 2
    for call in seed.await_args_list:
        assert call.args[0] == "owner-1"
        assert call.args[1].ownerUserId == "owner-1"
    credentials.assert_awaited_once_with("owner-1")
    # A fixed order is what makes "first-seen" mean the same thing on every
    # load, so the card's logos do not reshuffle.
    grant_query.assert_awaited_once_with(
        where={"expertId": {"in": ["expert-1", "expert-2"]}},
        order=[{"createdAt": "asc"}, {"id": "asc"}],
    )


@pytest.mark.asyncio
async def test_empty_roster_skips_credential_reads():
    with patch.object(
        credential_counts, "_user_credentials", new=AsyncMock()
    ) as credentials:
        assert await credential_counts.expert_credential_providers("owner-1", []) == {}
    credentials.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("with_metrics", [True, False])
@pytest.mark.parametrize("count_error", [False, True])
async def test_roster_includes_providers_only_when_metrics_are_requested(
    with_metrics, count_error
):
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
            "expert_credential_providers",
            new=AsyncMock(
                return_value={row.id: ["github", "github", "notion"]},
                side_effect=(
                    RuntimeError("credentials unavailable") if count_error else None
                ),
            ),
        ) as counts,
    ):
        roster = await experts_db.list_experts("owner-1", with_metrics=with_metrics)

    assert client.find_many.await_args.kwargs["where"]["ownerUserId"] == "owner-1"
    has_metrics = with_metrics and not count_error
    assert roster[0].credential_count == (3 if has_metrics else 0)
    # Distinct, first-seen order: the card shows one logo per provider.
    assert roster[0].credential_providers == (
        ["github", "notion"] if has_metrics else []
    )
    if with_metrics:
        counts.assert_awaited_once_with("owner-1", [row])
    else:
        counts.assert_not_awaited()
