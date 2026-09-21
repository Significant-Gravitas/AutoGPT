"""Publishing at the database boundary: which versions a template may point
at, and that the stored package names the same ones.

The DB-backed cases live in ``publish_test``; these pin the queries and the
rows written without a database.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import prisma.models
import pytest
import pytest_mock
from prisma import Base64

from backend.api.features.experts import package_export, publish
from backend.api.features.experts.expert_zip import package_from_zip
from backend.api.features.experts.package_model import (
    ExpertManifest,
    ExpertPackage,
    PackagedIdentity,
)
from backend.api.features.experts.publish import (
    UnpublishedWorkflowsError,
    publish_expert,
)
from backend.api.features.store.store_listing_versions import (
    installable_store_version_where,
)

pytestmark = pytest.mark.asyncio

ADMIN = "admin-1"


def _agent() -> prisma.models.LibraryAgent:
    return prisma.models.LibraryAgent.model_construct(
        id="agent-1", agentGraphId="graph-1", name="Weekly rollup", AgentGraph=None
    )


def _workflow(**overrides) -> prisma.models.ExpertWorkflow:
    values = {
        "id": "wf-1",
        "expertId": "expert-1",
        "storeListingVersionId": None,
        "StoreListingVersion": None,
        "LibraryAgent": _agent(),
        "scheduleCron": "40 7 * * *",
    }
    values.update(overrides)
    return prisma.models.ExpertWorkflow.model_construct(**values)


def _expert(*workflows: prisma.models.ExpertWorkflow) -> prisma.models.Expert:
    return prisma.models.Expert.model_construct(
        id="expert-1",
        name="Maria Ops",
        ownerUserId=ADMIN,
        avatarUrl=None,
        voicePreferences="Brief.",
        Workflows=list(workflows),
    )


def _version(version_id: str) -> prisma.models.StoreListingVersion:
    return prisma.models.StoreListingVersion.model_construct(
        id=version_id,
        name="Weekly rollup",
        subHeading="Rolls the week up",
        StoreListing=prisma.models.StoreListing.model_construct(
            id="listing-1",
            slug="weekly-rollup",
            CreatorProfile=prisma.models.Profile.model_construct(username="ada"),
        ),
    )


def _listing(active: str | None) -> prisma.models.StoreListing:
    return prisma.models.StoreListing.model_construct(
        id="listing-1", activeVersionId=active, isDeleted=False
    )


def _versions(mocker: pytest_mock.MockerFixture, found) -> MagicMock:
    client = MagicMock()
    client.find_first = AsyncMock(return_value=found)
    mocker.patch.object(
        prisma.models.StoreListingVersion, "prisma", return_value=client
    )
    return client


def _listings(mocker: pytest_mock.MockerFixture, found) -> MagicMock:
    client = MagicMock()
    client.find_first = AsyncMock(return_value=found)
    mocker.patch.object(prisma.models.StoreListing, "prisma", return_value=client)
    return client


def _transaction(mocker: pytest_mock.MockerFixture, existing=None) -> MagicMock:
    tx = MagicMock()
    template = prisma.models.Expert.model_construct(id="template-1")
    tx.expert.find_first = AsyncMock(return_value=existing)
    tx.expert.create = AsyncMock(return_value=template)
    tx.expert.update = AsyncMock(return_value=template)
    tx.expert.find_unique_or_raise = AsyncMock(return_value=template)
    tx.expertworkflow.delete_many = AsyncMock(return_value=0)
    tx.expertworkflow.create = AsyncMock()
    tx.expertpublishedpackage.upsert = AsyncMock()

    @asynccontextmanager
    async def fake_transaction():
        yield tx

    mocker.patch.object(publish, "transaction", fake_transaction)
    return tx


def _build(mocker: pytest_mock.MockerFixture) -> AsyncMock:
    return mocker.patch.object(
        publish,
        "build_expert_package",
        new_callable=AsyncMock,
        return_value=ExpertPackage(
            manifest=ExpertManifest(identity=PackagedIdentity(name="Maria Ops"))
        ),
    )


async def test_a_rows_own_version_is_held_to_the_installability_test(
    mocker: pytest_mock.MockerFixture,
):
    """A version since deleted, hidden or withdrawn blocks the publish, and
    nothing is built or written."""
    versions = _versions(mocker, None)
    _listings(mocker, None)
    build = _build(mocker)
    tx = _transaction(mocker)

    with pytest.raises(UnpublishedWorkflowsError) as exc:
        await publish_expert(
            _expert(_workflow(storeListingVersionId="v-gone")), user_id=ADMIN
        )

    assert exc.value.workflows == ["Weekly rollup"]
    assert versions.find_first.await_args_list[0].kwargs["where"] == {
        "id": "v-gone",
        **installable_store_version_where(),
    }
    build.assert_not_awaited()
    tx.expert.create.assert_not_awaited()
    tx.expertpublishedpackage.upsert.assert_not_awaited()


async def test_a_listing_whose_active_version_is_not_installable_blocks_the_publish(
    mocker: pytest_mock.MockerFixture,
):
    """The graph lookup does not trust ``hasApprovedVersion``: the active
    version it names has to pass the same test."""
    _listings(mocker, _listing(active="v-pending"))
    versions = _versions(mocker, None)
    build = _build(mocker)

    with pytest.raises(UnpublishedWorkflowsError) as exc:
        await publish_expert(_expert(_workflow()), user_id=ADMIN)

    assert exc.value.workflows == ["Weekly rollup"]
    assert versions.find_first.await_args.kwargs["where"] == {
        "id": "v-pending",
        **installable_store_version_where(),
    }
    build.assert_not_awaited()


async def test_an_agent_published_after_install_is_packaged_with_the_templates_version(
    mocker: pytest_mock.MockerFixture,
):
    """The stored zip must name the marketplace version the template's row
    points at, or Download → Import makes a private copy of what Hire
    installs from the marketplace."""
    _listings(mocker, _listing(active="v-new"))
    _versions(mocker, _version("v-new"))
    build = _build(mocker)
    tx = _transaction(mocker)
    mocker.patch.object(
        package_export, "_exported_graph", new_callable=AsyncMock
    ).return_value = None

    await publish_expert(_expert(_workflow()), user_id=ADMIN)

    packaged_row = build.await_args.args[0]
    assert build.await_args.kwargs == {"user_id": ADMIN}
    workflow = packaged_row.Workflows[0]
    assert workflow.storeListingVersionId == "v-new"
    assert workflow.StoreListingVersion.id == "v-new"
    # What the exporter writes for that row: the same version, by id, slug
    # and creator, that the template row below carries.
    reference = await package_export._workflow(workflow, ADMIN)
    assert reference is not None
    assert reference.store_listing_version_id == "v-new"
    assert reference.store_listing_slug == "weekly-rollup"
    assert reference.creator_username == "ada"
    assert reference.schedule_cron == "40 7 * * *"
    assert tx.expertworkflow.create.await_args.kwargs["data"] == {
        "expertId": "template-1",
        "storeListingVersionId": "v-new",
        "scheduleCron": "40 7 * * *",
    }


async def test_the_package_is_written_to_its_own_table(
    mocker: pytest_mock.MockerFixture,
):
    """Beside the template, not on it: a roster listing must never fetch it."""
    _versions(mocker, _version("v1"))
    _build(mocker)
    tx = _transaction(mocker)

    await publish_expert(_expert(_workflow(storeListingVersionId="v1")), user_id=ADMIN)

    fields = tx.expert.create.await_args.kwargs["data"]
    assert "publishedPackage" not in fields and "PublishedPackage" not in fields
    assert fields["publishedFromExpertId"] == "expert-1"
    upsert = tx.expertpublishedpackage.upsert.await_args.kwargs
    assert upsert["where"] == {"expertId": "template-1"}
    stored = package_from_zip(Base64.decode(upsert["data"]["create"]["package"]))
    assert stored.manifest.identity.name == "Maria Ops"
    assert upsert["data"]["update"]["package"] == upsert["data"]["create"]["package"]


async def test_publishing_again_refreshes_the_same_template_and_package(
    mocker: pytest_mock.MockerFixture,
):
    _versions(mocker, _version("v1"))
    _build(mocker)
    tx = _transaction(
        mocker, existing=prisma.models.Expert.model_construct(id="template-1")
    )

    await publish_expert(_expert(_workflow(storeListingVersionId="v1")), user_id=ADMIN)

    tx.expert.create.assert_not_awaited()
    assert tx.expert.update.await_args.kwargs["where"] == {"id": "template-1"}
    assert tx.expertpublishedpackage.upsert.await_args.kwargs["where"] == {
        "expertId": "template-1"
    }
