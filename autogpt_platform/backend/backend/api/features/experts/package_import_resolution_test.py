"""What an import resolves and persists, at the database boundary.

The DB-backed cases live in ``package_import_test``; these pin the queries and
the rows written without a database, so a change to the installability test,
the voice envelope or the schedule opt-out is caught by a mocked delegate.
"""

from unittest.mock import AsyncMock, MagicMock

import prisma.models
import pytest
import pytest_mock

from backend.api.features.experts import experts_db, package_import
from backend.api.features.experts.models import VoiceSample, decode_voice_preferences
from backend.api.features.experts.package_import import (
    ExpertImportEdits,
    ExpertImportWorkflowEdit,
    resolve_workflow,
)
from backend.api.features.experts.package_model import (
    ExpertManifest,
    ExpertPackage,
    PackagedIdentity,
    PackagedSoul,
    PackagedWorkflow,
)
from backend.api.features.store.store_listing_versions import (
    installable_store_version_where,
)
from backend.data.graph import Graph

pytestmark = pytest.mark.asyncio


def _versions(mocker: pytest_mock.MockerFixture, found=None) -> MagicMock:
    client = MagicMock()
    client.find_first = AsyncMock(return_value=found)
    mocker.patch.object(
        prisma.models.StoreListingVersion, "prisma", return_value=client
    )
    return client


def _listings(mocker: pytest_mock.MockerFixture, found=None) -> MagicMock:
    client = MagicMock()
    client.find_first = AsyncMock(return_value=found)
    mocker.patch.object(prisma.models.StoreListing, "prisma", return_value=client)
    return client


def _graph() -> Graph:
    return Graph(name="Digest", description="Rolls the week up")


def _version(version_id: str = "v1") -> prisma.models.StoreListingVersion:
    return prisma.models.StoreListingVersion.model_construct(id=version_id)


def _listing(
    active: str | None = "v-active", deleted: bool = False
) -> prisma.models.StoreListing:
    return prisma.models.StoreListing.model_construct(
        id="listing-1", activeVersionId=active, isDeleted=deleted
    )


# ---------------------------------------------------------------------------
# Resolution is the library's own installability test
# ---------------------------------------------------------------------------


async def test_a_version_id_is_held_to_the_librarys_installability_test(
    mocker: pytest_mock.MockerFixture,
):
    versions = _versions(mocker, _version())

    resolved = await resolve_workflow(
        PackagedWorkflow(name="Digest", store_listing_version_id="v1"), 0
    )

    assert resolved.source == "store"
    assert resolved.store_listing_version_id == "v1"
    where = versions.find_first.await_args.kwargs["where"]
    assert where == {"id": "v1", **installable_store_version_where()}
    # The parent listing's deletion is part of that test.
    assert where["StoreListing"] == {"is": {"isDeleted": False}}


async def test_a_version_a_hire_would_refuse_falls_back_to_the_graph(
    mocker: pytest_mock.MockerFixture,
):
    """A deleted listing, or a version since hidden or withdrawn, must land as
    the embedded copy rather than as a store install that fails."""
    _versions(mocker, None)
    _listings(mocker, None)

    resolved = await resolve_workflow(
        PackagedWorkflow(name="Digest", store_listing_version_id="v1", graph=_graph()),
        0,
    )

    assert resolved.source == "graph"
    assert resolved.store_listing_version_id is None


async def test_a_slug_resolves_to_its_listings_active_version_through_the_same_test(
    mocker: pytest_mock.MockerFixture,
):
    listings = _listings(mocker, _listing(active="v-active"))
    versions = _versions(mocker, _version("v-active"))

    resolved = await resolve_workflow(
        PackagedWorkflow(
            name="Digest", store_listing_slug="digest", creator_username="ada"
        ),
        0,
    )

    assert resolved.source == "store"
    assert resolved.store_listing_version_id == "v-active"
    listing_where = listings.find_first.await_args.kwargs["where"]
    assert listing_where["slug"] == "digest"
    assert listing_where["CreatorProfile"] == {"is": {"username": "ada"}}
    assert versions.find_first.await_args.kwargs["where"] == {
        "id": "v-active",
        **installable_store_version_where(),
    }


@pytest.mark.parametrize(
    "listing",
    [None, _listing(active=None), _listing(deleted=True)],
    ids=["no-listing", "no-active-version", "deleted-listing"],
)
async def test_a_slug_without_an_installable_active_version_is_not_a_store_source(
    mocker: pytest_mock.MockerFixture, listing: prisma.models.StoreListing | None
):
    _listings(mocker, listing)
    versions = _versions(mocker, None)

    resolved = await resolve_workflow(
        PackagedWorkflow(
            name="Digest", store_listing_slug="digest", creator_username="ada"
        ),
        0,
    )

    assert resolved.source == "unresolvable"
    versions.find_first.assert_not_awaited()


async def test_a_slug_whose_active_version_is_not_installable_is_not_a_store_source(
    mocker: pytest_mock.MockerFixture,
):
    """``activeVersionId`` alone is not enough: the version it names is put
    through the installability test before it counts."""
    _listings(mocker, _listing(active="v-pending"))
    _versions(mocker, None)

    resolved = await resolve_workflow(
        PackagedWorkflow(
            name="Digest",
            store_listing_slug="digest",
            creator_username="ada",
            graph=_graph(),
        ),
        0,
    )

    assert resolved.source == "graph"


# ---------------------------------------------------------------------------
# What the rows carry
# ---------------------------------------------------------------------------


def _install_mocks(mocker: pytest_mock.MockerFixture) -> tuple[MagicMock, AsyncMock]:
    mocker.patch.object(
        package_import, "_listing_version_id", new_callable=AsyncMock
    ).return_value = "v1"
    mocker.patch.object(
        package_import, "_library_agent", new_callable=AsyncMock
    ).return_value = MagicMock(id="agent-1", graph_id="graph-1", graph_version=1)
    rows = MagicMock()
    rows.create = AsyncMock(
        return_value=prisma.models.ExpertWorkflow.model_construct(id="row-1")
    )
    mocker.patch.object(prisma.models.ExpertWorkflow, "prisma", return_value=rows)
    mocker.patch.object(
        package_import, "get_user_by_id", new_callable=AsyncMock
    ).return_value = None
    schedule = mocker.patch.object(
        package_import.scheduling, "create_workflow_schedule", new_callable=AsyncMock
    )
    return rows, schedule


@pytest.mark.parametrize(
    "edits",
    [
        ExpertImportEdits(),
        ExpertImportEdits(
            workflows=[ExpertImportWorkflowEdit(index=0, schedule_enabled=False)]
        ),
    ],
    ids=["edits-omitted", "explicitly-off"],
)
async def test_a_cadence_the_user_turned_off_is_not_written_as_pending_setup(
    mocker: pytest_mock.MockerFixture, edits: ExpertImportEdits
):
    """A cron with no schedule id is what the credential-grant retry treats as
    "needs setup" and starts. Opting out has to mean no cron on the row, or it
    only delays the schedule until the next grant."""
    rows, schedule = _install_mocks(mocker)
    workflow = PackagedWorkflow(
        name="Digest", graph=_graph(), schedule_cron="0 7 * * *"
    )

    failed = await package_import._install_workflows(
        "user-1", "expert-1", [(0, workflow)], edits
    )

    assert failed == []
    assert rows.create.await_args.kwargs["data"]["scheduleCron"] is None
    schedule.assert_not_awaited()


async def test_a_cadence_the_user_turned_on_is_written_and_started(
    mocker: pytest_mock.MockerFixture,
):
    rows, schedule = _install_mocks(mocker)
    workflow = PackagedWorkflow(
        name="Digest", graph=_graph(), schedule_cron="0 7 * * *"
    )
    edits = ExpertImportEdits(
        workflows=[ExpertImportWorkflowEdit(index=0, schedule_enabled=True)]
    )

    await package_import._install_workflows(
        "user-1", "expert-1", [(0, workflow)], edits
    )

    assert rows.create.await_args.kwargs["data"]["scheduleCron"] == "0 7 * * *"
    assert schedule.await_args.kwargs["cron"] == "0 7 * * *"
    assert schedule.await_args.kwargs["workflow_row_id"] == "row-1"


async def test_voice_samples_survive_the_import_and_read_back_decoded(
    mocker: pytest_mock.MockerFixture,
):
    """The soul's samples are stored in the envelope the column already
    uses, and an owned row carrying that envelope reads back as description
    plus samples — never as raw JSON in the expert's voice."""
    samples = [
        VoiceSample(label="a", text="Short. Direct."),
        VoiceSample(label="b", text="Warm."),
    ]
    package = ExpertPackage(
        manifest=ExpertManifest(
            identity=PackagedIdentity(name="Maria Ops"),
            soul=PackagedSoul(voice_preferences="Brief.", voice_samples=samples),
        )
    )
    create = mocker.patch.object(
        experts_db, "create_imported_expert", new_callable=AsyncMock
    )
    create.return_value = prisma.models.Expert.model_construct(id="expert-1")
    mocker.patch.object(
        package_import, "install_package_skills", new_callable=AsyncMock
    ).return_value = []
    mocker.patch.object(
        package_import, "_install_workflows", new_callable=AsyncMock
    ).return_value = []
    read_back = mocker.patch.object(experts_db, "get_expert", new_callable=AsyncMock)

    stored = None

    async def _get_expert(user_id: str, expert_id: str):
        nonlocal stored
        stored = create.await_args.kwargs["voice_preferences"]
        return experts_db._to_model(_owned_row(stored))

    read_back.side_effect = _get_expert

    result = await package_import.import_package("user-1", package, ExpertImportEdits())

    assert stored is not None
    assert decode_voice_preferences(stored) == ("Brief.", samples)
    assert result.expert.voice_preferences == "Brief."
    assert result.expert.voice_samples == samples


async def test_a_voice_without_samples_is_stored_as_plain_text(
    mocker: pytest_mock.MockerFixture,
):
    package = ExpertPackage(
        manifest=ExpertManifest(
            identity=PackagedIdentity(name="Maria Ops"),
            soul=PackagedSoul(voice_preferences="Brief."),
        )
    )
    create = mocker.patch.object(
        experts_db, "create_imported_expert", new_callable=AsyncMock
    )
    create.return_value = prisma.models.Expert.model_construct(id="expert-1")
    mocker.patch.object(
        package_import, "install_package_skills", new_callable=AsyncMock
    ).return_value = []
    mocker.patch.object(
        package_import, "_install_workflows", new_callable=AsyncMock
    ).return_value = []
    mocker.patch.object(
        experts_db, "get_expert", new_callable=AsyncMock
    ).return_value = experts_db._to_model(_owned_row("Brief."))

    await package_import.import_package("user-1", package, ExpertImportEdits())

    assert create.await_args.kwargs["voice_preferences"] == "Brief."


def _owned_row(voice_preferences: str) -> prisma.models.Expert:
    return prisma.models.Expert.model_construct(
        id="expert-1",
        name="Maria Ops",
        avatarUrl=None,
        color="",
        role="Ops lead",
        tagline=None,
        bio=None,
        skills=[],
        categories=[],
        identity="Careful.",
        voicePreferences=voice_preferences,
        boundaries="",
        dayOne=None,
        isTemplate=False,
        sourceTemplateId=None,
        isArchived=False,
        weeklyBudget=None,
        schedulesPausedAt=None,
        podId=None,
        Workflows=[],
        Credentials=[],
    )
