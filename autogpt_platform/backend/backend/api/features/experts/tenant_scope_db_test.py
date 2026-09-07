from uuid import uuid4

import pytest
from prisma.enums import ResourceVisibility

from backend.api.features.experts import credentials, experts_db
from backend.data import db
from backend.util.exceptions import ExpertNotFoundError


@pytest.mark.asyncio
async def test_persisted_experts_are_private_to_their_owner_in_shared_workspace():
    users = [str(uuid4()), str(uuid4())]
    organization_id = str(uuid4())
    team_id = str(uuid4())
    try:
        for user_id in users:
            await db.prisma.user.create(
                data={"id": user_id, "email": f"{user_id}@expert-test.example"}
            )
        await db.prisma.organization.create(
            data={
                "id": organization_id,
                "name": "Expert scope",
                "slug": organization_id,
            }
        )
        await db.prisma.team.create(
            data={"id": team_id, "orgId": organization_id, "name": "Shared team"}
        )
        for user_id in users:
            await db.prisma.orgmember.create(
                data={"orgId": organization_id, "userId": user_id}
            )
            await db.prisma.teammember.create(
                data={"teamId": team_id, "userId": user_id}
            )
        expert_ids = {}
        for name, owner, visibility in [
            ("mine", users[0], ResourceVisibility.PRIVATE),
            ("other-private", users[1], ResourceVisibility.PRIVATE),
            ("other-shared", users[1], ResourceVisibility.TEAM),
            ("mine-shared", users[0], ResourceVisibility.TEAM),
        ]:
            expert = await db.prisma.expert.create(
                data={
                    "name": name,
                    "role": "Test",
                    "identity": "Test expert",
                    "ownerUserId": owner,
                    "organizationId": organization_id,
                    "teamId": team_id,
                    "visibility": visibility,
                }
            )
            expert_ids[name] = expert.id
        roster = await experts_db.list_experts(
            users[0],
            organization_id=organization_id,
            team_id_restriction=team_id,
            team_ids=[team_id],
            with_metrics=False,
        )
        assert [expert.id for expert in roster] == [expert_ids["mine"]]
        mine = await credentials._owned_expert(users[0], expert_ids["mine"])
        assert mine.id == expert_ids["mine"]
        for name in ["other-private", "other-shared", "mine-shared"]:
            with pytest.raises(ExpertNotFoundError):
                await credentials._owned_expert(users[0], expert_ids[name])
    finally:
        await db.prisma.user.delete_many(where={"id": {"in": users}})
        await db.prisma.organization.delete_many(where={"id": organization_id})
