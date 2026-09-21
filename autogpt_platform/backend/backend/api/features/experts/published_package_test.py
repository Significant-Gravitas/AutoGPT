"""The published package lives in its own table: ordinary expert reads never
fetch it, and the two paths that need it load it by template id."""

from unittest.mock import AsyncMock, MagicMock

import prisma.models
import pytest
import pytest_mock
from prisma import Base64

from backend.api.features.experts import experts_db, seed
from backend.api.features.experts.expert_zip import zip_from_package
from backend.api.features.experts.models import ExpertDayOneItem
from backend.api.features.experts.package_model import (
    ExpertManifest,
    ExpertPackage,
    PackagedIdentity,
)

pytestmark = pytest.mark.asyncio

ZIP = zip_from_package(
    ExpertPackage(manifest=ExpertManifest(identity=PackagedIdentity(name="Maria Ops")))
)


def _includes_package(include) -> bool:
    if not isinstance(include, dict):
        return False
    return "PublishedPackage" in include or any(
        _includes_package(value) for value in include.values()
    )


def test_no_expert_projection_carries_the_package():
    """Prisma fetches every scalar of a model on every read, so the blob has
    to be a different model — and no expert read may include it."""
    assert "publishedPackage" not in prisma.models.Expert.model_fields
    assert "package" in prisma.models.ExpertPublishedPackage.model_fields
    for include in (
        experts_db._TEMPLATE_WORKFLOW_INCLUDE,
        experts_db._ROSTER_WORKFLOW_INCLUDE,
        experts_db._WORKFLOW_INCLUDE,
        experts_db.EXPORT_INCLUDE,
    ):
        assert not _includes_package(include)


async def test_browsing_the_roster_reads_no_package(
    mocker: pytest_mock.MockerFixture,
):
    client = MagicMock()
    client.find_many = AsyncMock(return_value=[])
    client.find_first = AsyncMock(return_value=None)
    mocker.patch.object(prisma.models.Expert, "prisma", return_value=client)
    packages = MagicMock()
    packages.find_unique = AsyncMock()
    mocker.patch.object(
        prisma.models.ExpertPublishedPackage, "prisma", return_value=packages
    )

    await experts_db.list_templates()
    await experts_db.get_template("template-1")
    await experts_db.get_template_row("template-1")

    for call in client.find_many.await_args_list + client.find_first.await_args_list:
        assert not _includes_package(call.kwargs.get("include"))
    packages.find_unique.assert_not_awaited()


@pytest.mark.parametrize("stored", [True, False], ids=["published", "roster"])
async def test_the_package_is_loaded_by_template_id_from_its_own_table(
    mocker: pytest_mock.MockerFixture, stored: bool
):
    packages = MagicMock()
    packages.find_unique = AsyncMock(
        return_value=(
            prisma.models.ExpertPublishedPackage.model_construct(
                expertId="template-1", package=Base64.encode(ZIP)
            )
            if stored
            else None
        )
    )
    mocker.patch.object(
        prisma.models.ExpertPublishedPackage, "prisma", return_value=packages
    )

    package = await experts_db.get_published_package("template-1")

    assert package == (ZIP if stored else None)
    assert packages.find_unique.await_args.kwargs["where"] == {"expertId": "template-1"}


@pytest.mark.parametrize("stored", [True, False], ids=["published", "roster"])
async def test_a_hire_loads_the_package_explicitly_and_installs_its_skills(
    mocker: pytest_mock.MockerFixture, stored: bool
):
    template = prisma.models.Expert.model_construct(id="template-1")
    mocker.patch.object(
        experts_db, "get_published_package", new_callable=AsyncMock
    ).return_value = (ZIP if stored else None)
    install = mocker.patch.object(
        experts_db, "install_package_skills", new_callable=AsyncMock
    )
    install.return_value = []

    await experts_db._install_published_skills("user-1", "expert-1", template)

    if not stored:
        install.assert_not_awaited()
        return
    args = install.await_args.args
    assert args[:2] == ("user-1", "expert-1")
    assert args[2].manifest.identity.name == "Maria Ops"


async def test_the_seeder_leaves_any_template_with_a_package_alone(
    mocker: pytest_mock.MockerFixture,
):
    """Keyed on the package relation, which nothing clears, rather than on
    ``publishedFromExpertId``, which deleting the source expert nulls."""
    client = MagicMock()
    client.find_first = AsyncMock(return_value=None)
    client.create = AsyncMock(
        return_value=prisma.models.Expert.model_construct(id="template-1")
    )
    mocker.patch.object(prisma.models.Expert, "prisma", return_value=client)
    entry: seed.RosterEntry = {
        "name": "Maria Ops",
        "role": "Roster role",
        "job_title": "Roster job title",
        "tagline": "Roster tagline",
        "avatar_url": "/experts/maria.svg",
        "bio": "Roster bio.",
        "bundled_skills": [],
        "categories": ["marketing"],
        "identity": "Roster identity.",
        "voice_preferences": "Clear and confident.",
        "boundaries": "Never invent customer evidence.",
        "day_one": [ExpertDayOneItem(title="Social listening on your brand")],
        "preloads": [],
    }

    await seed._upsert_template(entry)

    where = client.find_first.await_args.kwargs["where"]
    assert where["name"] == "Maria Ops"
    assert where["PublishedPackage"] == {"is": None}
    assert "publishedPackage" not in where
