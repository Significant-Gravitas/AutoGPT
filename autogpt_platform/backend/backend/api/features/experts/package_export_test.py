"""Tests for what an export carries out of this installation — and, mostly,
what it leaves behind."""

import json
from datetime import datetime, timezone

import prisma.enums
import prisma.models
import pytest
import pytest_mock

from backend.api.features.experts import package_export
from backend.api.features.experts.expert_zip import package_from_zip, zip_from_package
from backend.api.features.experts.models import VoiceSample, encode_voice_preferences
from backend.api.features.experts.package_export import (
    build_expert_package,
    package_filename,
)
from backend.api.features.experts.package_model import (
    MAX_MANIFEST_BYTES,
    ExpertPackageError,
    manifest_json,
)
from backend.copilot.tools.skills import ParsedSkill, SkillFile, SkillPackage
from backend.data.graph import GraphModel

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
OWNER = "user-kept-local"
ORG = "org-kept-local"
TEAM = "team-kept-local"
SKILL_MD = "---\nname: research\ndescription: Digs things up.\n---\n\n# Research\n"


def _expert(**overrides) -> prisma.models.Expert:
    return prisma.models.Expert.model_validate(
        {
            "id": "expert-1",
            "createdAt": NOW,
            "updatedAt": NOW,
            "ownerUserId": OWNER,
            "organizationId": ORG,
            "teamId": TEAM,
            "name": "Maria Ops",
            "color": "sky-300",
            "role": "Ops lead",
            "tagline": "Keeps the week moving",
            "bio": "Runs the weekly rhythm.",
            "identity": "Careful and brief.",
            "boundaries": "Never spends without asking.",
            "voicePreferences": "Short sentences.",
            "skills": ["Research"],
            "categories": ["productivity"],
            "isTemplate": False,
            "isArchived": False,
            "visibility": prisma.enums.ResourceVisibility.PRIVATE,
            "weeklyBudget": 4242,
            "toolProfile": json.dumps({"enabled": ["search"]}),
            "Workflows": [],
            **overrides,
        }
    )


def _workflow(**overrides) -> prisma.models.ExpertWorkflow:
    return prisma.models.ExpertWorkflow.model_validate(
        {"id": "wf-1", "createdAt": NOW, "expertId": "expert-1", **overrides}
    )


def _store_version(**overrides) -> prisma.models.StoreListingVersion:
    """Only the fields the exporter reads: a row built by prisma carries the
    other thirty, and naming them here would hide which ones matter."""
    return prisma.models.StoreListingVersion.model_construct(
        id="ver-1",
        name="Morning digest",
        subHeading="A digest every morning",
        **overrides,
    )


def _listing() -> prisma.models.StoreListing:
    return prisma.models.StoreListing.model_construct(
        id="listing-1",
        slug="morning-digest",
        CreatorProfile=prisma.models.Profile.model_construct(username="ada"),
    )


def _library_agent(**overrides) -> prisma.models.LibraryAgent:
    return prisma.models.LibraryAgent.model_construct(
        id="lib-1",
        userId=OWNER,
        agentGraphId="graph-9",
        agentGraphVersion=4,
        name=None,
        description=None,
        **overrides,
    )


@pytest.fixture(autouse=True)
def no_skills(mocker: pytest_mock.MockFixture):
    """Most tests are not about skills, and none of them should reach a real
    workspace."""
    mocker.patch.object(package_export, "list_user_skills", return_value=[])
    return mocker.patch.object(package_export, "read_user_skill_package")


def _graph(**overrides) -> GraphModel:
    return GraphModel.model_validate(
        {
            "id": "graph-9",
            "version": 4,
            "name": "Weekly rollup",
            "description": "Rolls the week up",
            "user_id": OWNER,
            "organization_id": ORG,
            "team_id": TEAM,
            "created_at": NOW,
            "nodes": [],
            "links": [],
            **overrides,
        }
    )


# ---------------------------------------------------------------------------
# What never leaves
# ---------------------------------------------------------------------------


async def test_nothing_local_to_this_installation_is_written():
    """The owner, the tenancy, the budget: all of it means something only
    here, and an importer that inherited any of it would be wrong."""
    package = await build_expert_package(_expert())

    raw = manifest_json(package.manifest).decode()
    assert OWNER not in raw and ORG not in raw and TEAM not in raw
    assert "4242" not in raw
    assert "expert-1" not in raw


async def test_the_soul_and_the_identity_survive_intact():
    package = await build_expert_package(
        _expert(
            voicePreferences=encode_voice_preferences(
                "Plain words", [VoiceSample(label="a", text="Morning!")]
            ),
            dayOne=json.dumps(
                [{"title": "Wire up the inbox", "description": "", "timing": ""}]
            ),
        )
    )

    manifest = package.manifest
    assert manifest.identity.name == "Maria Ops"
    assert manifest.identity.tagline == "Keeps the week moving"
    assert manifest.soul.identity == "Careful and brief."
    assert manifest.soul.voice_preferences == "Plain words"
    assert [s.text for s in manifest.soul.voice_samples] == ["Morning!"]
    assert [d.title for d in manifest.day_one] == ["Wire up the inbox"]
    assert manifest.tool_profile == {"enabled": ["search"]}


# ---------------------------------------------------------------------------
# Workflows
# ---------------------------------------------------------------------------


async def test_a_published_workflow_carries_its_reference_and_its_graph(
    mocker: pytest_mock.MockFixture,
):
    """The reference is what an import prefers — the published agent, its
    updates, its creator's attribution — but the graph rides along so the file
    still restores where that listing does not exist."""
    mocker.patch("backend.data.graph.get_graph", return_value=_graph())
    package = await build_expert_package(
        _expert(
            Workflows=[
                _workflow(
                    storeListingVersionId="ver-1",
                    StoreListingVersion=_store_version(StoreListing=_listing()),
                    LibraryAgent=_library_agent(),
                    scheduleCron="40 7 * * *",
                    scheduleId="sched-local",
                    libraryAgentId="lib-local",
                )
            ]
        )
    )

    workflow = package.manifest.workflows[0]
    assert workflow.store_listing_version_id == "ver-1"
    assert workflow.store_listing_slug == "morning-digest"
    assert workflow.creator_username == "ada"
    assert workflow.name == "Morning digest"
    assert workflow.graph is not None
    assert workflow.schedule_cron == "40 7 * * *"
    raw = manifest_json(package.manifest).decode()
    assert "sched-local" not in raw and "lib-local" not in raw


async def test_a_template_workflow_carries_the_reference_alone():
    """A roster template has no LibraryAgent, so there is no graph to read and
    no owner to read it as."""
    package = await build_expert_package(
        _expert(
            ownerUserId=None,
            isTemplate=True,
            Workflows=[
                _workflow(
                    storeListingVersionId="ver-1",
                    StoreListingVersion=_store_version(StoreListing=_listing()),
                )
            ],
        )
    )

    workflow = package.manifest.workflows[0]
    assert workflow.store_listing_version_id == "ver-1"
    assert workflow.graph is None


async def test_an_unpublished_workflow_carries_the_export_stripped_graph(
    mocker: pytest_mock.MockFixture,
):
    """An agent that was never published has no reference to travel as, so the
    graph itself goes — asked for the way an export must ask for it, with the
    original owner's credentials and webhooks taken out."""
    get_graph = mocker.patch(
        "backend.data.graph.get_graph", return_value=_graph(name="Weekly rollup")
    )

    package = await build_expert_package(
        _expert(
            Workflows=[
                _workflow(
                    libraryAgentId="lib-1",
                    LibraryAgent=_library_agent(),
                    scheduleCron="0 9 * * 1",
                )
            ]
        )
    )

    assert get_graph.await_args.kwargs["for_export"] is True
    assert get_graph.await_args.kwargs["include_subgraphs"] is True
    assert get_graph.await_args.kwargs["user_id"] == OWNER
    workflow = package.manifest.workflows[0]
    assert workflow.graph is not None
    assert workflow.graph.name == "Weekly rollup"
    assert workflow.schedule_cron == "0 9 * * 1"
    dumped = json.loads(manifest_json(package.manifest))["workflows"][0]["graph"]
    assert "user_id" not in dumped and "created_at" not in dumped


async def test_a_workflow_with_neither_a_listing_nor_an_agent_is_dropped():
    """It would arrive as a workflow the new owner could never run."""
    package = await build_expert_package(_expert(Workflows=[_workflow()]))
    assert package.manifest.workflows == []


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------


async def test_skills_travel_as_whole_packages(mocker: pytest_mock.MockFixture):
    mocker.patch.object(
        package_export,
        "list_user_skills",
        return_value=[ParsedSkill(name="Research", description="Digs.", body="")],
    )
    mocker.patch.object(
        package_export,
        "read_user_skill_package",
        return_value=SkillPackage(
            skill_md=SKILL_MD,
            files=[SkillFile(relative_path="refs/API.md", content=b"# API\n")],
        ),
    )

    package = await build_expert_package(_expert())

    assert [s.slug for s in package.manifest.skills] == ["research"]
    assert [f.relative_path for f in package.skills["research"].files] == [
        "refs/API.md"
    ]


async def test_a_skill_with_no_stored_package_is_left_out_of_both(
    mocker: pytest_mock.MockFixture,
):
    """A card without files would make the archive fail its own reader's
    manifest-versus-tree check."""
    mocker.patch.object(
        package_export,
        "list_user_skills",
        return_value=[ParsedSkill(name="Research", description="Digs.", body="")],
    )
    mocker.patch.object(package_export, "read_user_skill_package", return_value=None)

    package = await build_expert_package(_expert())

    assert package.manifest.skills == []
    assert package.skills == {}


# ---------------------------------------------------------------------------
# Caps and the file itself
# ---------------------------------------------------------------------------


async def test_a_manifest_over_the_cap_is_refused_before_a_file_is_written():
    """Better a 413 than a download that our own reader would refuse."""
    with pytest.raises(ExpertPackageError) as exc:
        await build_expert_package(
            _expert(toolProfile=json.dumps({"pad": "x" * (MAX_MANIFEST_BYTES + 1)}))
        )
    assert exc.value.over_limit


async def test_an_export_reads_back_as_the_same_expert(
    mocker: pytest_mock.MockFixture,
):
    mocker.patch.object(
        package_export,
        "list_user_skills",
        return_value=[ParsedSkill(name="Research", description="Digs.", body="")],
    )
    mocker.patch.object(
        package_export,
        "read_user_skill_package",
        return_value=SkillPackage(skill_md=SKILL_MD),
    )

    package = await build_expert_package(_expert())
    restored = package_from_zip(zip_from_package(package))

    assert restored.manifest == package.manifest
    assert restored.skills["research"].skill_md == SKILL_MD


def test_the_download_is_named_after_the_expert():
    assert package_filename("Maria Ops") == "maria-ops.expert.zip"
    assert package_filename("!!!") == "expert.expert.zip"
