import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio
from prisma import Prisma

from backend.data.db import prisma

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        os.environ.get("RUN_ORG_ISOLATION_DB_TESTS") != "1",
        reason="Requires an isolated migrated PostgreSQL database",
    ),
]

_MIGRATIONS = Path(__file__).parents[4] / "migrations"


class _Rollback(Exception):
    pass


@pytest_asyncio.fixture
async def migration_db():
    # The client is shared with the rest of the session: leave it as found.
    connected_here = not prisma.is_connected()
    if connected_here:
        await prisma.connect()
    try:
        with pytest.raises(_Rollback):
            async with prisma.tx(timeout=timedelta(seconds=30)) as tx:
                yield tx
                raise _Rollback
    finally:
        if connected_here:
            await prisma.disconnect()


@pytest.mark.parametrize(
    "other_membership", ["foreign", "deleted", "newer", "tied", "foreign_only"]
)
async def test_backfills_choose_original_owned_personal_org(
    migration_db, other_membership
):
    db = migration_db
    user = str(uuid4())
    await db.user.create(data={"id": user, "email": f"{user}@example.invalid"})
    wrong_org, _ = await _membership(db, user, other_membership, preferred=False)
    expected_org, expected_team = (
        await _membership(db, user, "original", preferred=True)
        if other_membership != "foreign_only"
        else (None, None)
    )
    file_id = await _legacy_resources(db, user)

    for name in (
        "20261016173000_private_experts_skills_org",
        "20261016173500_skill_submission_org",
    ):
        await _apply_backfill(db, name)

    for table in (
        "Expert",
        "ExpertPod",
        "SkillHead",
        "SkillVersion",
        "SkillLearningSource",
        "SkillLearningReview",
        "SkillSuppression",
        "SkillUseEvent",
        "SkillListing",
    ):
        owner = (
            "ownerUserId"
            if table == "Expert"
            else "owningUserId" if table == "SkillListing" else "userId"
        )
        rows = await db.query_raw(
            f'SELECT "organizationId" FROM "{table}" WHERE "{owner}" = $1', user
        )
        assert len(rows) == 1
        assert rows[0]["organizationId"] == expected_org, table
        assert rows[0]["organizationId"] != wrong_org, table

    file = await db.userworkspacefile.find_unique(where={"id": file_id})
    assert file is not None
    assert (file.organizationId, file.teamId) == (expected_org, expected_team)
    assert file.isUserGlobalConfig == (expected_org is None)
    assert file.storagePath == "migration-test/legacy/SKILL.md"
    if expected_org:
        assert file.scopeKey == f"{expected_org}:{expected_team}"
    for table in (
        "SkillHead",
        "SkillVersion",
        "SkillLearningSource",
        "SkillLearningReview",
        "SkillSuppression",
        "SkillUseEvent",
    ):
        rows = await db.query_raw(
            f'SELECT "ownerKey" FROM "{table}" WHERE "userId" = $1', user
        )
        assert rows[0]["ownerKey"] == (
            f"org:{expected_org}:personal" if expected_org else "personal"
        ), table


async def _membership(db: Prisma, user: str, kind: str, *, preferred: bool):
    org = await db.organization.create(
        data={
            "name": kind,
            "slug": str(uuid4()),
            "isPersonal": True,
            "deletedAt": (
                datetime(2020, 1, 1, tzinfo=UTC) if kind == "deleted" else None
            ),
        }
    )
    team = await db.team.create(
        data={"name": "Home", "orgId": org.id, "isDefault": True}
    )
    # Insert the competing membership first; timestamps and ids, not row order,
    # must select the original owner. Tied timestamps exercise the id tie-break.
    year = 2021 if kind == "newer" else 2020 if preferred or kind == "tied" else 2019
    await db.orgmember.create(
        data={
            "id": ("a-" if preferred else "z-") + str(uuid4()),
            "orgId": org.id,
            "userId": user,
            "isOwner": kind not in ("foreign", "foreign_only"),
            "createdAt": datetime(year, 1, 1, tzinfo=UTC),
        }
    )
    if kind != "deleted":
        await db.teammember.create(
            data={"teamId": team.id, "userId": user, "isAdmin": True}
        )
    if kind in ("foreign", "foreign_only"):
        other = str(uuid4())
        await db.user.create(data={"id": other, "email": f"{other}@example.invalid"})
        await db.orgmember.create(
            data={"orgId": org.id, "userId": other, "isOwner": True}
        )
    return org.id, team.id


async def _legacy_resources(db: Prisma, user: str) -> str:
    workspace = await db.userworkspace.create(data={"userId": user})
    file = await db.userworkspacefile.create(
        data={
            "workspaceId": workspace.id,
            "name": "SKILL.md",
            "path": "/skills/legacy/SKILL.md",
            "storagePath": "migration-test/legacy/SKILL.md",
            "mimeType": "text/markdown",
            "sizeBytes": 1,
            "isUserGlobalConfig": True,
        }
    )
    await db.expert.create(
        data={
            "ownerUserId": user,
            "name": "Legacy",
            "role": "Legacy",
            "identity": "Private",
        }
    )
    await db.expertpod.create(data={"userId": user, "name": "Legacy"})
    head = await db.skillhead.create(
        data={"userId": user, "ownerKey": "personal", "skillName": "legacy"}
    )
    await db.skillversion.create(
        data={
            "userId": user,
            "ownerKey": "personal",
            "skillName": "legacy",
            "headId": head.id,
            "version": 1,
            "content": "Private",
            "contentHash": "fake-hash",
            "description": "Private",
            "origin": "imported",
        }
    )
    source = await db.skilllearningsource.create(
        data={
            "userId": user,
            "ownerKey": "personal",
            "sourceKind": "chat_session",
            "sourceId": str(uuid4()),
            "revision": "1",
        }
    )
    await db.skilllearningreview.create(
        data={
            "userId": user,
            "ownerKey": "personal",
            "sourceId": source.id,
            "sourceRevision": "1",
            "policyVersion": 1,
            "runId": str(uuid4()),
            "disposition": "skipped",
        }
    )
    await db.skillsuppression.create(
        data={
            "userId": user,
            "ownerKey": "personal",
            "skillName": "legacy",
            "behaviorFingerprint": "fake",
        }
    )
    await db.skilluseevent.create(
        data={
            "userId": user,
            "ownerKey": "personal",
            "skillName": "legacy",
            "kind": "loaded",
        }
    )
    await db.profile.create(
        data={
            "userId": user,
            "name": "Owner",
            "username": str(uuid4()),
            "description": "",
            "links": [],
        }
    )
    await db.skilllisting.create(data={"owningUserId": user, "slug": str(uuid4())})
    return file.id


async def _apply_backfill(db: Prisma, migration: str) -> None:
    for statement in (_MIGRATIONS / migration / "migration.sql").read_text().split(";"):
        sql = "\n".join(
            line
            for line in statement.splitlines()
            if not line.lstrip().startswith("--")
        ).strip()
        if sql.startswith("UPDATE") or (
            sql.startswith('ALTER TABLE "UserWorkspaceFile"') and "TRIGGER" in sql
        ):
            await db.execute_raw(sql)
