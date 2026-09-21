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
    MAX_PACKAGE_SKILLS,
    MAX_PACKAGE_WORKFLOWS,
    ExpertPackageError,
    manifest_json,
)
from backend.copilot.tools.skills import (
    MAX_PACKAGE_BYTES,
    MAX_PACKAGE_FILE_BYTES,
    ParsedSkill,
    SkillFile,
    SkillPackage,
    SkillPackageError,
    prepare_user_skill,
    render_skill_markdown,
)
from backend.data.graph import GraphModel

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
OWNER = "user-kept-local"
ORG = "org-kept-local"
TEAM = "team-kept-local"
SKILL_MD = "---\nname: research\ndescription: Digs things up.\n---\n\n# Research\n"
DOWNLOADER = "user-downloading"


async def _build(row: prisma.models.Expert, user_id: str = OWNER):
    return await build_expert_package(row, user_id=user_id)


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
        **{
            "id": "ver-1",
            "name": "Morning digest",
            "subHeading": "A digest every morning",
            **overrides,
        }
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
    workspace or the Skills Hub."""
    mocker.patch.object(package_export, "list_user_skills", return_value=[])
    mocker.patch.object(
        package_export.experts_db, "bundled_skill_listings", return_value=[]
    )
    return mocker.patch.object(package_export, "read_user_skill_package")


def _hub_listing(slug: str, **version: object) -> prisma.models.SkillListing:
    """A live Hub listing as ``bundled_skill_listings`` returns it: the row
    with its active version, which is all an install reads."""
    return prisma.models.SkillListing.model_construct(
        id=f"listing-{slug}",
        slug=slug,
        ActiveVersion=prisma.models.SkillListingVersion.model_construct(
            **{
                "version": 3,
                "name": slug.title(),
                "description": f"{slug} description",
                "body": f"# {slug}\n",
                "triggers": ["t1"],
                **version,
            }
        ),
    )


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
    package = await _build(_expert())

    raw = manifest_json(package.manifest).decode()
    assert OWNER not in raw and ORG not in raw and TEAM not in raw
    assert "4242" not in raw
    assert "expert-1" not in raw


async def test_the_soul_and_the_identity_survive_intact():
    package = await _build(
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
    package = await _build(
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
    package = await _build(
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

    package = await _build(
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
    package = await _build(_expert(Workflows=[_workflow()]))
    assert package.manifest.workflows == []


def _published(i: int) -> prisma.models.ExpertWorkflow:
    return _workflow(
        id=f"wf-{i}",
        storeListingVersionId=f"ver-{i}",
        StoreListingVersion=_store_version(id=f"ver-{i}", StoreListing=_listing()),
    )


async def test_more_exportable_workflows_than_the_cap_is_refused_not_truncated():
    """Installing has no such cap, so a valid expert can be over it — and a
    package quietly missing runnable workflows is not a backup."""
    over = [_published(i) for i in range(MAX_PACKAGE_WORKFLOWS + 1)]

    with pytest.raises(ExpertPackageError, match="workflows") as exc:
        await _build(_expert(Workflows=over))
    assert exc.value.over_limit


async def test_workflows_without_a_source_do_not_count_toward_the_cap():
    at_cap = [_published(i) for i in range(MAX_PACKAGE_WORKFLOWS)]
    sourceless = [_workflow(id=f"none-{i}") for i in range(5)]

    package = await _build(_expert(Workflows=at_cap + sourceless))

    assert len(package.manifest.workflows) == MAX_PACKAGE_WORKFLOWS


async def test_workflow_reading_stops_at_the_first_excess_exportable_workflow(
    mocker: pytest_mock.MockFixture,
):
    """Every graph after the one that proved the expert over the cap is a
    fetch that only delays the 413."""
    get_graph = mocker.patch("backend.data.graph.get_graph", return_value=_graph())
    unpublished = [
        _workflow(id=f"wf-{i}", LibraryAgent=_library_agent())
        for i in range(MAX_PACKAGE_WORKFLOWS + 10)
    ]

    with pytest.raises(ExpertPackageError, match="workflows") as exc:
        await _build(_expert(Workflows=unpublished))

    assert exc.value.over_limit
    assert get_graph.call_count == MAX_PACKAGE_WORKFLOWS + 1


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

    package = await _build(_expert())

    assert [s.slug for s in package.manifest.skills] == ["research"]
    assert [f.relative_path for f in package.skills["research"].files] == [
        "refs/API.md"
    ]


async def test_more_skills_than_the_cap_is_refused_not_truncated(
    mocker: pytest_mock.MockFixture,
):
    """The folder listing is not held to the package cap, so it can run over."""
    mocker.patch.object(
        package_export,
        "list_user_skills",
        return_value=[
            ParsedSkill(name=f"skill-{i}", description="d", body="")
            for i in range(MAX_PACKAGE_SKILLS + 1)
        ],
    )
    mocker.patch.object(
        package_export,
        "read_user_skill_package",
        return_value=SkillPackage(skill_md=SKILL_MD),
    )

    with pytest.raises(ExpertPackageError, match="skills") as exc:
        await _build(_expert())
    assert exc.value.over_limit


async def test_a_stored_skill_the_skill_download_refuses_is_refused_here_too(
    mocker: pytest_mock.MockFixture,
):
    """A tree over the files cap raises out of the skill store; that is a 413
    on this route as well, not a 500."""
    mocker.patch.object(
        package_export,
        "list_user_skills",
        return_value=[ParsedSkill(name="Research", description="Digs.", body="")],
    )
    mocker.patch.object(
        package_export,
        "read_user_skill_package",
        side_effect=SkillPackageError("more than 100 files", over_limit=True),
    )

    with pytest.raises(ExpertPackageError, match="research") as exc:
        await _build(_expert())
    assert exc.value.over_limit


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

    package = await _build(_expert())

    assert package.manifest.skills == []
    assert package.skills == {}


# ---------------------------------------------------------------------------
# A roster template's skills come from the Skills Hub
# ---------------------------------------------------------------------------


async def test_a_template_carries_the_hub_skills_it_bundles_as_a_hire_installs_them(
    mocker: pytest_mock.MockFixture,
):
    """Downloading a marketplace template must not lose the skills a hire of
    it would come with — and they are read behind the same gate a hire uses,
    keyed by the downloader."""
    bundled = mocker.patch.object(
        package_export.experts_db,
        "bundled_skill_listings",
        return_value=[_hub_listing("brand-voice-guide"), _hub_listing("outreach")],
    )
    listed = mocker.patch.object(package_export, "list_user_skills")
    template = _expert(id="template-1", ownerUserId=None, isTemplate=True)

    package = await _build(template, user_id=DOWNLOADER)

    bundled.assert_awaited_once_with(DOWNLOADER, "template-1")
    listed.assert_not_called()
    assert [(s.slug, s.name, s.description) for s in package.manifest.skills] == [
        ("brand-voice-guide", "brand-voice-guide", "brand-voice-guide description"),
        ("outreach", "outreach", "outreach description"),
    ]
    installed = prepare_user_skill(
        name="brand-voice-guide",
        description="brand-voice-guide description",
        body="# brand-voice-guide\n",
        triggers=["t1"],
        version="3",
    )
    assert package.skills["brand-voice-guide"].skill_md == render_skill_markdown(
        installed
    )
    assert package.skills["brand-voice-guide"].files == []


async def test_a_bundled_listing_a_hire_would_refuse_to_install_is_left_out(
    mocker: pytest_mock.MockFixture,
):
    """The install validates before it writes; a listing it would reject
    cannot be in the package either, or the file would list a skill the hire
    never has."""
    mocker.patch.object(
        package_export.experts_db,
        "bundled_skill_listings",
        return_value=[_hub_listing("empty", body=""), _hub_listing("outreach")],
    )

    package = await _build(
        _expert(ownerUserId=None, isTemplate=True), user_id=DOWNLOADER
    )

    assert [s.slug for s in package.manifest.skills] == ["outreach"]
    assert list(package.skills) == ["outreach"]


async def test_a_template_bundled_skill_round_trips_through_the_archive(
    mocker: pytest_mock.MockFixture,
):
    mocker.patch.object(
        package_export.experts_db,
        "bundled_skill_listings",
        return_value=[_hub_listing("brand-voice-guide")],
    )

    package = await _build(
        _expert(ownerUserId=None, isTemplate=True), user_id=DOWNLOADER
    )
    restored = package_from_zip(zip_from_package(package))

    assert restored.manifest == package.manifest
    assert restored.skills == package.skills


async def test_an_owned_expert_never_reads_the_hub(mocker: pytest_mock.MockFixture):
    bundled = mocker.patch.object(package_export.experts_db, "bundled_skill_listings")

    await _build(_expert())

    bundled.assert_not_called()


# ---------------------------------------------------------------------------
# Caps and the file itself
# ---------------------------------------------------------------------------


def _skill_of(*sizes: int) -> SkillPackage:
    return SkillPackage(
        skill_md=SKILL_MD,
        files=[
            SkillFile(relative_path=f"assets/f{i}.bin", content=b"\0" * size)
            for i, size in enumerate(sizes)
        ],
    )


def _stored_skills(mocker: pytest_mock.MockFixture, skills: dict[str, SkillPackage]):
    mocker.patch.object(
        package_export,
        "list_user_skills",
        return_value=[
            ParsedSkill(name=slug, description="d", body="") for slug in skills
        ],
    )
    mocker.patch.object(
        package_export,
        "read_user_skill_package",
        side_effect=lambda _owner, slug, **_: skills[slug],
    )


async def test_skills_that_are_each_legal_but_together_over_the_cap_are_refused(
    mocker: pytest_mock.MockFixture,
):
    """The reader bounds the whole tree at once, so two skills that each fit
    would download and then fail on re-import."""
    half = [MAX_PACKAGE_FILE_BYTES] * 6  # 12 MiB, under the 20 MiB per-skill cap
    _stored_skills(mocker, {"first": _skill_of(*half), "second": _skill_of(*half)})

    with pytest.raises(ExpertPackageError, match="unpacks to") as exc:
        await _build(_expert())
    assert exc.value.over_limit


async def test_skill_reading_stops_at_the_first_skill_that_overflows_the_package(
    mocker: pytest_mock.MockFixture,
):
    """Fifty stored skills that each fit could otherwise be read whole — a
    gibibyte held in memory — just to answer 413; and neither the avatar nor
    a graph is worth fetching once the skills alone do not fit."""
    half = [MAX_PACKAGE_FILE_BYTES] * 6  # 12 MiB, under the 20 MiB per-skill cap
    _stored_skills(mocker, {f"skill-{i}": _skill_of(*half) for i in range(4)})
    reads = mocker.patch.object(
        package_export,
        "read_user_skill_package",
        side_effect=lambda _owner, slug, **_: _skill_of(*half),
    )
    avatar = mocker.patch.object(package_export, "packaged_avatar")
    get_graph = mocker.patch("backend.data.graph.get_graph")

    with pytest.raises(ExpertPackageError, match="unpacks to") as exc:
        await _build(
            _expert(
                avatarUrl="https://cdn.example.com/maria.png",
                Workflows=[_workflow(LibraryAgent=_library_agent())],
            )
        )

    assert exc.value.over_limit
    assert reads.call_count == 2
    avatar.assert_not_called()
    get_graph.assert_not_called()


async def test_skill_reading_stops_at_the_first_skill_past_the_count_cap(
    mocker: pytest_mock.MockFixture,
):
    _stored_skills(
        mocker,
        {f"skill-{i}": _skill_of() for i in range(MAX_PACKAGE_SKILLS + 10)},
    )
    reads = mocker.patch.object(
        package_export,
        "read_user_skill_package",
        side_effect=lambda _owner, slug, **_: _skill_of(),
    )

    with pytest.raises(ExpertPackageError, match="skills") as exc:
        await _build(_expert())

    assert exc.value.over_limit
    assert reads.call_count == MAX_PACKAGE_SKILLS + 1


async def test_a_template_whose_bundled_skills_alone_overflow_is_refused(
    mocker: pytest_mock.MockFixture,
):
    half = [MAX_PACKAGE_FILE_BYTES] * 6
    mocker.patch.object(
        package_export.experts_db,
        "bundled_skill_listings",
        return_value=[_hub_listing(f"listing-{i}") for i in range(3)],
    )
    installable = mocker.patch.object(
        package_export.skill_db,
        "installable_skill",
        side_effect=lambda listing: (
            ParsedSkill(name=listing.slug, description="d", body=""),
            _skill_of(*half),
        ),
    )

    with pytest.raises(ExpertPackageError, match="unpacks to") as exc:
        await _build(_expert(ownerUserId=None, isTemplate=True), user_id=DOWNLOADER)

    assert exc.value.over_limit
    assert installable.call_count == 2


async def test_a_stored_skill_file_over_the_cap_is_refused(
    mocker: pytest_mock.MockFixture,
):
    _stored_skills(mocker, {"research": _skill_of(MAX_PACKAGE_FILE_BYTES + 1)})

    with pytest.raises(ExpertPackageError, match="research") as exc:
        await _build(_expert())
    assert exc.value.over_limit


async def test_a_stored_root_skill_md_over_the_file_cap_is_refused(
    mocker: pytest_mock.MockFixture,
):
    """The skill store accepts a root of this size; the archive reader does
    not, so the export is the place to say so."""
    oversized = SkillPackage(skill_md=SKILL_MD + "x" * MAX_PACKAGE_FILE_BYTES)
    _stored_skills(mocker, {"research": oversized})

    with pytest.raises(ExpertPackageError, match="research.*SKILL.md") as exc:
        await _build(_expert())
    assert exc.value.over_limit


async def test_an_export_exactly_at_the_cap_reads_back(
    mocker: pytest_mock.MockFixture,
):
    _stored_skills(mocker, {"research": _skill_of(1)})
    probe = await _build(_expert())
    room = MAX_PACKAGE_BYTES - probe.size_bytes + 1
    sizes = [MAX_PACKAGE_FILE_BYTES] * (room // MAX_PACKAGE_FILE_BYTES)
    sizes.append(room - sum(sizes))
    _stored_skills(mocker, {"research": _skill_of(*sizes)})

    package = await _build(_expert())
    restored = package_from_zip(zip_from_package(package))

    assert package.size_bytes == MAX_PACKAGE_BYTES
    assert (
        restored.skills["research"].size_bytes == package.skills["research"].size_bytes
    )


async def test_a_manifest_over_the_cap_is_refused_before_a_file_is_written():
    """Better a 413 than a download that our own reader would refuse."""
    with pytest.raises(ExpertPackageError) as exc:
        await _build(
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

    package = await _build(_expert())
    restored = package_from_zip(zip_from_package(package))

    assert restored.manifest == package.manifest
    assert restored.skills["research"].skill_md == SKILL_MD


def test_the_download_is_named_after_the_expert():
    assert package_filename("Maria Ops") == "maria-ops.expert.zip"
    assert package_filename("!!!") == "expert.expert.zip"
