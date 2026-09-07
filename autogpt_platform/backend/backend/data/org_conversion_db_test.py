import asyncio
from uuid import uuid4

import pytest
from prisma import Prisma

from backend.api.features.orgs import db as org_db
from backend.data import db
from backend.data.org_migration import create_personal_org


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome", ["success", "fail_before_creation", "fail_after_creation"]
)
async def test_personal_conversion_is_atomic_for_independent_readers(outcome, mocker):
    user_id = str(uuid4())
    reader = Prisma(auto_register=False, datasource={"url": db.DATABASE_URL})
    await reader.connect()
    try:
        await db.prisma.user.create(
            data={"id": user_id, "email": f"{user_id}@conversion.example"}
        )
        original = await create_personal_org(
            user_id, f"conversion-{user_id}", "Conversion"
        )
        entered, release = asyncio.Event(), asyncio.Event()
        create_replacement = org_db._create_personal_org_for_user

        async def paused_replacement(*args, **kwargs):
            replacement = None
            if outcome != "fail_before_creation":
                replacement = await create_replacement(*args, **kwargs)
            entered.set()
            await release.wait()
            if outcome != "success":
                raise RuntimeError("Replacement creation failed")
            return replacement

        mocker.patch.object(
            org_db, "_create_personal_org_for_user", side_effect=paused_replacement
        )
        conversion = asyncio.create_task(
            org_db.convert_personal_org(original.id, user_id)
        )
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            visible = await reader.organization.find_many(
                where={
                    "bootstrapUserId": user_id,
                    "isPersonal": True,
                    "deletedAt": None,
                }
            )
            assert [org.id for org in visible] == [original.id]
            default_org, _ = await org_db.get_user_default_team(user_id)
            assert default_org == original.id
        finally:
            release.set()
            result = (await asyncio.gather(conversion, return_exceptions=True))[0]

        personal = await reader.organization.find_many(
            where={
                "bootstrapUserId": user_id,
                "isPersonal": True,
                "deletedAt": None,
            }
        )
        assert len(personal) == 1
        all_owned = await reader.organization.find_many(
            where={"bootstrapUserId": user_id}
        )
        if outcome == "success":
            assert not isinstance(result, BaseException)
            assert len(all_owned) == 2 and personal[0].id != original.id
            replacement_id = personal[0].id
            assert (
                await reader.orgmember.count(
                    where={
                        "orgId": replacement_id,
                        "userId": user_id,
                        "isOwner": True,
                        "status": "ACTIVE",
                    }
                )
                == 1
            )
            teams = await reader.team.find_many(
                where={"orgId": replacement_id, "isDefault": True}
            )
            assert len(teams) == 1
            assert (
                await reader.teammember.count(
                    where={"teamId": teams[0].id, "userId": user_id, "status": "ACTIVE"}
                )
                == 1
            )
            assert (
                await reader.organizationprofile.count(
                    where={"organizationId": replacement_id}
                )
                == 1
            )
            assert await reader.orgbalance.count(where={"orgId": replacement_id}) == 1
        else:
            assert isinstance(result, RuntimeError)
            assert len(all_owned) == 1 and personal[0].id == original.id
    finally:
        await reader.organization.delete_many(where={"bootstrapUserId": user_id})
        await reader.user.delete_many(where={"id": user_id})
        await reader.disconnect()
