"""Tests for turning an admin's own expert into a marketplace template."""

import prisma.models
import pytest

from backend.api.features.experts import experts_db
from backend.api.features.experts.expert_zip import package_from_zip
from backend.api.features.experts.experts_db_test import (
    _create_seed_user,
    _delete_seeded_rows,
    _seed_own_library_agent,
    _seed_store_listing,
    _seeded_template_ids,
    _seeded_user_ids,
)
from backend.api.features.experts.publish import (
    UnpublishedWorkflowsError,
    publish_expert,
    published_template,
)
from backend.util.test import SpinTestServer


@pytest.fixture(autouse=True)
async def delete_rows_this_test_seeded():
    yield
    template_ids, user_ids = list(_seeded_template_ids), list(_seeded_user_ids)
    await _delete_seeded_rows(template_ids, user_ids)
    _seeded_template_ids.clear()
    _seeded_user_ids.clear()


async def _expert_with(user_id: str, **workflow) -> prisma.models.Expert:
    row = await experts_db.create_imported_expert(
        user_id,
        name="Maria Ops",
        role="Ops lead",
        tagline="Keeps the week moving",
        bio="Runs the weekly rhythm.",
        color="sky-300",
        categories=["operations"],
        identity="Careful and brief.",
        voice_preferences="Short sentences.",
        boundaries="Never spends without asking.",
        avatar_url="/experts/maria.svg",
        day_one=[],
        tool_profile=None,
    )
    _seeded_template_ids.append(row.id)
    if workflow:
        await prisma.models.ExpertWorkflow.prisma().create(
            data={"expertId": row.id, **workflow}
        )
    owned = await experts_db.get_owned_expert_row(user_id, row.id)
    assert owned is not None
    return owned


async def _published(template_id: str) -> prisma.models.Expert:
    row = await prisma.models.Expert.prisma().find_unique_or_raise(
        where={"id": template_id}, include={"Workflows": True}
    )
    _seeded_template_ids.append(row.id)
    return row


async def test_publishing_creates_an_ownerless_marketplace_template(
    server: SpinTestServer,
):
    admin = await _create_seed_user()
    version_id = await _seed_store_listing(server)
    expert = await _expert_with(
        admin.id, storeListingVersionId=version_id, scheduleCron="40 7 * * *"
    )

    template = await _published((await publish_expert(expert)).id)

    assert template.isTemplate is True
    assert template.ownerUserId is None
    assert template.publishedFromExpertId == expert.id
    assert template.name == "Maria Ops"
    assert template.tagline == "Keeps the week moving"
    assert template.avatarUrl == "/experts/maria.svg"
    assert [w.storeListingVersionId for w in template.Workflows or []] == [version_id]
    assert [w.scheduleCron for w in template.Workflows or []] == ["40 7 * * *"]


async def test_a_published_template_carries_the_package_it_was_published_from(
    server: SpinTestServer,
):
    """Download and hire both read this, so the marketplace keeps serving what
    was published even after the expert moves on."""
    admin = await _create_seed_user()
    expert = await _expert_with(admin.id)

    template = await _published((await publish_expert(expert)).id)

    assert template.publishedPackage is not None
    package = package_from_zip(template.publishedPackage.decode())
    assert package.manifest.identity.name == "Maria Ops"


async def test_publishing_again_refreshes_the_same_template(server: SpinTestServer):
    """One expert, one marketplace entry — republishing must not leave the old
    version standing beside the new one."""
    admin = await _create_seed_user()
    first_version = await _seed_store_listing(server)
    expert = await _expert_with(admin.id, storeListingVersionId=first_version)

    first = await publish_expert(expert)
    await prisma.models.Expert.prisma().update(
        where={"id": expert.id}, data={"name": "Maria Ops II"}
    )
    second_version = await _seed_store_listing(server)
    await prisma.models.ExpertWorkflow.prisma().delete_many(
        where={"expertId": expert.id}
    )
    await prisma.models.ExpertWorkflow.prisma().create(
        data={"expertId": expert.id, "storeListingVersionId": second_version}
    )
    refreshed = await experts_db.get_owned_expert_row(admin.id, expert.id)
    assert refreshed is not None
    second = await publish_expert(refreshed)

    assert second.id == first.id
    template = await _published(second.id)
    assert template.name == "Maria Ops II"
    assert [w.storeListingVersionId for w in template.Workflows or []] == [
        second_version
    ]
    assert (
        await prisma.models.Expert.prisma().count(
            where={"publishedFromExpertId": expert.id}
        )
        == 1
    )


async def test_an_agent_that_was_never_published_blocks_the_whole_publish(
    server: SpinTestServer,
):
    """A template pointing at a private graph would install nothing for
    whoever hired it."""
    admin = await _create_seed_user()
    library_agent_id, name = await _seed_own_library_agent(server, admin.id)
    expert = await _expert_with(admin.id, libraryAgentId=library_agent_id)

    with pytest.raises(UnpublishedWorkflowsError) as exc:
        await publish_expert(expert)

    assert exc.value.workflows == [name]
    assert await published_template(expert.id) is None


async def test_an_agent_published_after_it_was_installed_resolves(
    server: SpinTestServer,
):
    """The admin installed their own graph, then published it — the workflow
    has no storeListingVersionId but the listing exists."""
    admin = await _create_seed_user()
    version_id = await _seed_store_listing(server)
    version = await prisma.models.StoreListingVersion.prisma().find_unique_or_raise(
        where={"id": version_id}
    )
    agent = await prisma.models.LibraryAgent.prisma().find_first(
        where={"agentGraphId": version.agentGraphId}
    )
    assert agent is not None
    expert = await _expert_with(admin.id, libraryAgentId=agent.id)

    template = await _published((await publish_expert(expert)).id)

    assert [w.storeListingVersionId for w in template.Workflows or []] == [version_id]


async def test_a_published_template_is_offered_on_the_roster(server: SpinTestServer):
    admin = await _create_seed_user()
    expert = await _expert_with(admin.id)

    template = await publish_expert(expert)

    assert template.id in {t.id for t in await experts_db.list_templates()}
    found = await published_template(expert.id)
    assert found is not None and found.id == template.id
    _seeded_template_ids.append(template.id)
