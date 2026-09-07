from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import pytest

from backend.data import db
from backend.data.credit import UserCredit
from backend.data.credit_history import get_credit_history
from backend.data.org_migration import create_personal_org
from backend.data.tenancy import has_live_resource_access
from backend.util.json import SafeJson


@pytest.fixture
async def scoped_history():
    user_id = str(uuid4())
    organization_id = None
    try:
        await db.prisma.user.create(
            data={"id": user_id, "email": f"{user_id}@history-tenancy.example"}
        )
        org = await create_personal_org(user_id, f"history-{user_id}", "History")
        organization_id = org.id
        team = await db.prisma.team.find_first_or_raise(
            where={"orgId": org.id, "isDefault": True}
        )
        graph = await db.prisma.agentgraph.create(
            data={
                "userId": user_id,
                "name": "Confidential workflow",
                "organizationId": org.id,
                "teamId": team.id,
            }
        )
        library = await db.prisma.libraryagent.create(
            data={
                "userId": user_id,
                "agentGraphId": graph.id,
                "agentGraphVersion": graph.version,
                "isCreatedByUser": True,
                "organizationId": org.id,
                "teamId": team.id,
            }
        )
        run = await db.prisma.agentgraphexecution.create(
            data={
                "userId": user_id,
                "agentGraphId": graph.id,
                "agentGraphVersion": graph.version,
                "organizationId": org.id,
                "teamId": team.id,
                "executionStatus": "COMPLETED",
            }
        )
        session = await db.prisma.chatsession.create(
            data={
                "userId": user_id,
                "title": "Confidential conversation",
                "organizationId": org.id,
                "teamId": team.id,
            }
        )
        for source_id in (run.id, f"copilot-session-{session.id}"):
            await db.prisma.credittransaction.create(
                data={
                    "userId": user_id,
                    "transactionKey": str(uuid4()),
                    "amount": -12,
                    "type": "USAGE",
                    "metadata": SafeJson(
                        {"graph_exec_id": source_id, "graph_id": graph.id}
                    ),
                }
            )
        yield user_id, org.id, team.id, graph, library, run, session
    finally:
        await db.prisma.credittransaction.delete_many(where={"userId": user_id})
        await db.prisma.user.delete_many(where={"id": user_id})
        if organization_id:
            await db.prisma.organization.delete_many(where={"id": organization_id})


@pytest.mark.asyncio
async def test_revoked_team_history_retains_charges_without_private_metadata(
    scoped_history,
):
    user_id, organization_id, team_id, _, library, _, session = scoped_history
    before = await get_credit_history(user_id, viewer_organization_id=organization_id)
    assert {item.amount for item in before.transactions} == {-12}
    assert {item.library_agent_id for item in before.transactions} == {None, library.id}
    assert {item.conversation_id for item in before.transactions} == {None, session.id}
    for item in before.transactions:
        for url in (item.agent_url, item.execution_url, item.conversation_url):
            if url:
                params = parse_qs(urlsplit(url).query)
                assert params["organizationId"] == [organization_id]
                assert params["teamId"] == [team_id]

    await db.prisma.teammember.update_many(
        where={"userId": user_id, "teamId": team_id}, data={"status": "REMOVED"}
    )
    assert not await has_live_resource_access(user_id, organization_id, team_id, "view")

    after = await get_credit_history(user_id, viewer_organization_id=organization_id)
    assert len(after.transactions) == 2
    assert sum(item.amount for item in after.transactions) == -24
    assert all(item.agent_name is None for item in after.transactions)
    assert all(item.library_agent_id is None for item in after.transactions)
    assert all(item.conversation_id is None for item in after.transactions)
    assert all(item.conversation_title is None for item in after.transactions)
    assert all(not item.execution_available for item in after.transactions)
    assert all(item.agent_url is None for item in after.transactions)
    assert all(item.execution_url is None for item in after.transactions)
    assert all(item.conversation_url is None for item in after.transactions)


@pytest.mark.asyncio
@pytest.mark.parametrize("expert_session", [False, True])
async def test_other_org_history_context_does_not_use_private_library_fallback(
    scoped_history, expert_session
):
    user_id, source_org_id, source_team_id, _, _, _, session = scoped_history
    other_org_id = str(uuid4())
    try:
        await db.prisma.organization.create(
            data={"id": other_org_id, "slug": other_org_id, "name": "Other"}
        )
        await db.prisma.orgmember.create(
            data={"orgId": other_org_id, "userId": user_id, "isOwner": True}
        )
        if expert_session:
            expert = await db.prisma.expert.create(
                data={
                    "ownerUserId": user_id,
                    "name": "Private expert",
                    "role": "Researcher",
                    "identity": "Private expert",
                    "organizationId": source_org_id,
                    "teamId": source_team_id,
                }
            )
            await db.prisma.chatsession.update(
                where={"id": session.id}, data={"expertId": expert.id}
            )

        page = await get_credit_history(user_id, viewer_organization_id=other_org_id)

        assert len(page.transactions) == 2
        assert sum(item.amount for item in page.transactions) == -24
        assert all(item.agent_name is None for item in page.transactions)
        assert all(item.library_agent_id is None for item in page.transactions)
        assert all(item.conversation_id is None for item in page.transactions)
        assert all(item.agent_url is None for item in page.transactions)
        assert all(item.execution_url is None for item in page.transactions)
        assert all(item.conversation_url is None for item in page.transactions)
    finally:
        await db.prisma.organization.delete_many(where={"id": other_org_id})


@pytest.mark.asyncio
@pytest.mark.parametrize("resource_admin", [False, True])
async def test_personal_wallet_billing_delegate_is_not_the_owner_for_enrichment(
    scoped_history, resource_admin
):
    owner_id, organization_id, team_id, _, _, _, _ = scoped_history
    viewer_id = str(uuid4())
    try:
        await db.prisma.user.create(
            data={"id": viewer_id, "email": f"{viewer_id}@history-delegate.example"}
        )
        await db.prisma.orgmember.create(
            data={
                "orgId": organization_id,
                "userId": viewer_id,
                "isBillingManager": True,
                "isAdmin": resource_admin,
            }
        )
        if resource_admin:
            await db.prisma.teammember.create(
                data={"teamId": team_id, "userId": viewer_id, "isAdmin": True}
            )
        assert (
            await has_live_resource_access(viewer_id, organization_id, None, "view")
        ) is resource_admin

        page = await UserCredit(billing_user_id=owner_id).get_transaction_history(
            viewer_id, viewer_organization_id=organization_id
        )

        assert len(page.transactions) == 2
        assert sum(item.amount for item in page.transactions) == -24
        if resource_admin:
            assert {item.agent_name for item in page.transactions} == {
                None,
                "Confidential workflow",
            }
        else:
            assert all(item.agent_name is None for item in page.transactions)
        assert all(item.library_agent_id is None for item in page.transactions)
        assert all(item.conversation_id is None for item in page.transactions)
        assert all(item.conversation_title is None for item in page.transactions)
        assert all(item.conversation_url is None for item in page.transactions)
        assert all(item.agent_url is None for item in page.transactions)
    finally:
        await db.prisma.user.delete_many(where={"id": viewer_id})
