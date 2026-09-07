from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from backend.data import activity_event, db, org_migration


@pytest.mark.asyncio
async def test_persisted_events_preserve_actor_and_workspace_isolation():
    users = [str(uuid4()), str(uuid4())]
    try:
        for user_id in users:
            await db.prisma.user.create(
                data={"id": user_id, "email": f"{user_id}@activity-test.example"}
            )
        for title, actor, org, team in [
            ("visible", users[0], "org-a", "team-visible"),
            ("revoked", users[0], "org-a", "team-revoked"),
            ("org-home", users[0], "org-a", None),
            ("other-org", users[0], "org-b", "team-visible"),
            ("other-user", users[1], "org-a", "team-visible"),
            ("legacy", users[0], None, None),
        ]:
            event = await activity_event.create_activity_event(
                actor,
                activity_event.ActivityEventDraft(
                    category="FILE",
                    event_type="file_created",
                    title=title,
                    organization_id=org,
                    team_id=team,
                ),
            )
            assert (event.user_id, event.organization_id, event.team_id) == (
                actor,
                org,
                team,
            )
        since = datetime.now(timezone.utc) - timedelta(minutes=1)
        selected = await activity_event.list_activity_events(
            users[0], since, organization_id="org-a", team_id="team-visible"
        )
        assert {event.title for event in selected} == {"visible"}
        overview = await activity_event.list_activity_events(
            users[0], since, organization_id="org-a", team_ids=["team-visible"]
        )
        assert {event.title for event in overview} == {"visible", "org-home"}
        revoked = await activity_event.list_activity_events(
            users[0], since, organization_id="org-a", team_ids=[]
        )
        assert {event.title for event in revoked} == {"org-home"}
    finally:
        await db.prisma.user.delete_many(where={"id": {"in": users}})


@pytest.mark.asyncio
async def test_legacy_activity_scope_backfill_waits_for_owned_source_and_is_idempotent():
    users = [str(uuid4()), str(uuid4())]
    organization_id = str(uuid4())
    team_id = str(uuid4())
    try:
        for user_id in users:
            await db.prisma.user.create(
                data={"id": user_id, "email": f"{user_id}@activity-migration.example"}
            )
        await db.prisma.organization.create(
            data={
                "id": organization_id,
                "name": "Activity scope",
                "slug": organization_id,
            }
        )
        await db.prisma.team.create(
            data={"id": team_id, "orgId": organization_id, "name": "Work"}
        )
        await db.prisma.orgmember.create(
            data={"orgId": organization_id, "userId": users[0]}
        )
        await db.prisma.teammember.create(data={"teamId": team_id, "userId": users[0]})
        session = await db.prisma.chatsession.create(data={"userId": users[0]})
        home_session = await db.prisma.chatsession.create(
            data={"userId": users[0], "organizationId": organization_id}
        )
        graph = await db.prisma.agentgraph.create(data={"userId": users[0]})
        execution = await db.prisma.agentgraphexecution.create(
            data={
                "userId": users[0],
                "agentGraphId": graph.id,
                "agentGraphVersion": graph.version,
            }
        )
        events = {}
        for name, actor, source, org, team, execution_id in [
            ("owned", users[0], session.id, None, None, None),
            ("foreign", users[1], session.id, None, None, None),
            ("orphan", users[0], str(uuid4()), None, None, None),
            ("preserved", users[0], session.id, "existing-org", "existing-team", None),
            ("mismatched-refs", users[0], session.id, None, None, str(uuid4())),
            ("execution", users[0], None, None, None, execution.id),
            ("consistent-refs", users[0], session.id, None, None, execution.id),
            ("foreign-execution", users[1], None, None, None, execution.id),
            ("conflicting-scope", users[0], home_session.id, None, None, execution.id),
        ]:
            events[name] = await activity_event.create_activity_event(
                actor,
                activity_event.ActivityEventDraft(
                    category="FILE",
                    event_type="file_created",
                    title=name,
                    session_id=source,
                    organization_id=org,
                    team_id=team,
                    graph_exec_id=execution_id,
                ),
            )
        await org_migration._assign_activity_event_tenancy()
        pending = await db.prisma.activityevent.find_unique(
            where={"id": events["owned"].id}
        )
        assert pending is not None and pending.organizationId is None
        await db.prisma.chatsession.update(
            where={"id": session.id},
            data={"organizationId": organization_id, "teamId": team_id},
        )
        await db.prisma.agentgraphexecution.update(
            where={"id": execution.id},
            data={"organizationId": organization_id, "teamId": team_id},
        )
        await org_migration._assign_activity_event_tenancy()
        for name, event in events.items():
            row = await db.prisma.activityevent.find_unique(where={"id": event.id})
            assert row is not None
            expected = (
                (organization_id, team_id)
                if name in {"owned", "execution", "consistent-refs"}
                else (
                    ("existing-org", "existing-team")
                    if name == "preserved"
                    else (None, None)
                )
            )
            assert (row.organizationId, row.teamId) == expected
        assert await org_migration._assign_activity_event_tenancy() == 0
    finally:
        await db.prisma.user.delete_many(where={"id": {"in": users}})
        await db.prisma.organization.delete_many(where={"id": organization_id})
