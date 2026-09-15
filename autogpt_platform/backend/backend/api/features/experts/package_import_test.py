"""Tests for reading an uploaded package — above all, that reading one writes
nothing, and that each workflow is honest about which source it would land as."""

import prisma.models
import pytest

from backend.api.features.experts.errors import ACTIVE_EXPERT_LIMIT
from backend.api.features.experts.experts_db_test import (
    _create_seed_user,
    _delete_seeded_rows,
    _seed_store_listing,
    _seeded_template_ids,
    _seeded_user_ids,
)
from backend.api.features.experts.package_import import (
    preview_package,
    resolve_workflow,
)
from backend.api.features.experts.package_model import (
    ExpertManifest,
    ExpertPackage,
    PackagedIdentity,
    PackagedSkill,
    PackagedWorkflow,
)
from backend.copilot.tools.skills import SkillFile, SkillPackage
from backend.data.graph import Graph
from backend.util.test import SpinTestServer

SKILL_MD = "---\nname: research\ndescription: Digs.\n---\n\n# Research\n"


@pytest.fixture(autouse=True)
async def delete_rows_this_test_seeded():
    """The same teardown ``experts_db_test`` uses; its autouse fixture does not
    reach this module, and the seed helpers do."""
    yield
    template_ids, user_ids = list(_seeded_template_ids), list(_seeded_user_ids)
    await _delete_seeded_rows(template_ids, user_ids)
    _seeded_template_ids.clear()
    _seeded_user_ids.clear()


def _graph() -> Graph:
    return Graph(name="Weekly rollup", description="Rolls the week up")


def _package(*workflows: PackagedWorkflow, **manifest) -> ExpertPackage:
    return ExpertPackage(
        manifest=ExpertManifest(
            identity=PackagedIdentity(name="Maria Ops"),
            workflows=list(workflows),
            **manifest,
        )
    )


async def _listing_of(version_id: str) -> prisma.models.StoreListing:
    version = await prisma.models.StoreListingVersion.prisma().find_unique_or_raise(
        where={"id": version_id},
        include={"StoreListing": {"include": {"CreatorProfile": True}}},
    )
    assert version.StoreListing is not None
    return version.StoreListing


# ---------------------------------------------------------------------------
# Which source a workflow lands as
# ---------------------------------------------------------------------------


async def test_a_version_id_this_marketplace_has_resolves_to_the_listing(
    server: SpinTestServer,
):
    version_id = await _seed_store_listing(server)

    resolution = await resolve_workflow(
        PackagedWorkflow(
            name="Morning digest",
            store_listing_version_id=version_id,
            graph=_graph(),
            schedule_cron="40 7 * * *",
        ),
        0,
    )

    assert resolution.source == "store"
    assert resolution.store_listing_version_id == version_id
    assert resolution.schedule_cron == "40 7 * * *"


async def test_a_slug_scoped_to_its_creator_resolves_when_the_version_id_does_not(
    server: SpinTestServer,
):
    """Version ids are per-installation; the slug plus the creator's username
    is the reference that survives a move between them."""
    version_id = await _seed_store_listing(server)
    listing = await _listing_of(version_id)
    assert listing.CreatorProfile is not None

    resolution = await resolve_workflow(
        PackagedWorkflow(
            name="Morning digest",
            store_listing_version_id="a-version-id-from-somewhere-else",
            store_listing_slug=listing.slug,
            creator_username=listing.CreatorProfile.username,
        ),
        0,
    )

    assert resolution.source == "store"
    assert resolution.store_listing_version_id == listing.activeVersionId


async def test_a_slug_belonging_to_another_creator_is_not_used(
    server: SpinTestServer,
):
    """(owner, slug) is what a listing is unique on, so a slug alone could
    resolve to somebody else's agent entirely."""
    version_id = await _seed_store_listing(server)
    listing = await _listing_of(version_id)

    resolution = await resolve_workflow(
        PackagedWorkflow(
            name="Morning digest",
            store_listing_slug=listing.slug,
            creator_username="somebody-else",
            graph=_graph(),
        ),
        0,
    )

    assert resolution.source == "graph"


async def test_an_unpublished_agent_falls_back_to_the_graph_in_the_file(
    server: SpinTestServer,
):
    resolution = await resolve_workflow(
        PackagedWorkflow(name="Weekly rollup", graph=_graph()), 3
    )

    assert resolution.source == "graph"
    assert resolution.index == 3
    assert resolution.store_listing_version_id is None


async def test_a_reference_to_nothing_with_no_copy_is_unresolvable(
    server: SpinTestServer,
):
    resolution = await resolve_workflow(
        PackagedWorkflow(name="Ghost", store_listing_version_id="not-here"), 0
    )

    assert resolution.source == "unresolvable"
    assert resolution.reason is not None


# ---------------------------------------------------------------------------
# The preview as a whole
# ---------------------------------------------------------------------------


async def test_an_unresolvable_workflow_is_a_warning_not_an_error(
    server: SpinTestServer,
):
    """The rest of the expert still imports, so this must not block confirm."""
    user = await _create_seed_user()

    preview = await preview_package(
        user.id,
        _package(PackagedWorkflow(name="Ghost", store_listing_version_id="not-here")),
    )

    assert [w.source for w in preview.workflows] == ["unresolvable"]
    assert [w.code for w in preview.warnings] == ["unresolvable_workflow"]
    assert "Ghost" in preview.warnings[0].message
    assert preview.errors == []


async def test_the_preview_lists_each_skills_files(server: SpinTestServer):
    user = await _create_seed_user()
    package = _package(skills=[PackagedSkill(slug="research", name="Research")])
    package = package.model_copy(
        update={
            "skills": {
                "research": SkillPackage(
                    skill_md=SKILL_MD,
                    files=[SkillFile(relative_path="refs/API.md", content=b"# API\n")],
                )
            }
        }
    )

    preview = await preview_package(user.id, package)

    assert [s.slug for s in preview.skills] == ["research"]
    assert [f.path for f in preview.skills[0].files] == ["SKILL.md", "refs/API.md"]
    assert preview.skills[0].files[1].size_bytes == 6


async def test_a_full_roster_is_an_error_that_disables_confirm(
    server: SpinTestServer, mocker
):
    mocker.patch(
        "backend.api.features.experts.package_import.count_active_experts",
        return_value=ACTIVE_EXPERT_LIMIT,
    )
    user = await _create_seed_user()

    preview = await preview_package(user.id, _package())

    assert [e.code for e in preview.errors] == ["active_expert_limit"]


async def test_previewing_a_package_writes_nothing(server: SpinTestServer):
    """The whole point of a separate parse step: the user sees the expert
    before anything of theirs changes."""
    user = await _create_seed_user()
    before = await prisma.models.Expert.prisma().count()

    await preview_package(
        user.id,
        _package(
            PackagedWorkflow(name="Weekly rollup", graph=_graph()),
            skills=[PackagedSkill(slug="research", name="Research")],
        ),
    )

    assert await prisma.models.Expert.prisma().count() == before
    assert (
        await prisma.models.ExpertWorkflow.prisma().count(
            where={"Expert": {"is": {"ownerUserId": user.id}}}
        )
        == 0
    )


@pytest.mark.parametrize(
    "workflow, expected",
    [
        pytest.param(
            {"store_listing_version_id": "not-here"},
            ["workflow_not_in_marketplace"],
            id="a-marketplace-agent-this-installation-does-not-have",
        ),
        pytest.param({}, [], id="an-agent-that-was-never-published"),
    ],
)
async def test_falling_back_to_the_graph_warns_only_when_a_listing_was_lost(
    server: SpinTestServer, workflow: dict, expected: list[str]
):
    """A silent downgrade would leave the user wondering why their marketplace
    agent stopped getting its creator's updates — but an agent that was never
    published had no listing to lose, so there is nothing to say."""
    user = await _create_seed_user()

    preview = await preview_package(
        user.id, _package(PackagedWorkflow(name="Digest", graph=_graph(), **workflow))
    )

    assert [w.source for w in preview.workflows] == ["graph"]
    assert [w.code for w in preview.warnings] == expected
    assert preview.errors == []
