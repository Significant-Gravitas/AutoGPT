"""Tests for reading an uploaded package — above all, that reading one writes
nothing, and that each workflow is honest about which source it would land as."""

import uuid

import prisma.models
import pytest

from backend.api.features.experts.errors import ACTIVE_EXPERT_LIMIT
from backend.api.features.experts.expert_zip import package_from_zip, zip_from_package
from backend.api.features.experts.experts_db_test import (
    _create_seed_user,
    _delete_seeded_rows,
    _seed_store_listing,
    _seeded_template_ids,
    _seeded_user_ids,
)
from backend.api.features.experts.package_import import (
    ExpertImportEdits,
    ExpertImportResult,
    ExpertImportWorkflowEdit,
    import_package,
    preview_package,
    resolve_workflow,
)
from backend.api.features.experts.package_model import (
    ExpertManifest,
    ExpertPackage,
    PackagedAvatar,
    PackagedIdentity,
    PackagedSkill,
    PackagedSoul,
    PackagedWorkflow,
)
from backend.blocks.basic import StoreValueBlock
from backend.blocks.io import AgentInputBlock
from backend.copilot.tools.skills import (
    SkillFile,
    SkillLimitError,
    SkillPackage,
    read_user_skill_package,
)
from backend.data import graph as graph_db
from backend.data.graph import Graph, Link, Node
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


# ---------------------------------------------------------------------------
# Creating the expert
# ---------------------------------------------------------------------------


def _two_node_graph() -> Graph:
    """A graph with a link, so a round trip proves more than an empty shell:
    reassignment has to rewrite the link's endpoints along with the nodes."""
    source = Node(block_id=AgentInputBlock().id, input_default={"name": "input_1"})
    sink = Node(block_id=StoreValueBlock().id)
    return Graph(
        name=f"Weekly rollup {uuid.uuid4().hex[:8]}",
        description="Rolls the week up",
        nodes=[source, sink],
        links=[
            Link(
                source_id=source.id,
                sink_id=sink.id,
                source_name="result",
                sink_name="input",
            )
        ],
    )


def _skill_package() -> ExpertPackage:
    return ExpertPackage(
        manifest=ExpertManifest(
            identity=PackagedIdentity(name="Maria Ops", role="Ops lead"),
            soul=PackagedSoul(identity="Careful and brief."),
            skills=[
                PackagedSkill(slug="research", name="Research"),
                PackagedSkill(slug="digest", name="Digest"),
            ],
        ),
        skills={
            "research": SkillPackage(
                skill_md=SKILL_MD,
                files=[SkillFile(relative_path="refs/API.md", content=b"# API\n")],
            ),
            "digest": SkillPackage(skill_md=SKILL_MD.replace("research", "digest")),
        },
    )


async def _import(user_id: str, package: ExpertPackage, **edits) -> ExpertImportResult:
    result = await import_package(user_id, package, ExpertImportEdits(**edits))
    _seeded_template_ids.append(result.expert.id)
    return result


async def test_an_import_creates_the_callers_own_expert(server: SpinTestServer):
    user = await _create_seed_user()

    result = await _import(user.id, _package())

    assert result.expert.name == "Maria Ops"
    assert result.expert.source_template_id is None
    assert result.expert.is_template is False
    assert result.failed_skills == [] and result.failed_workflows == []


async def test_the_dialogs_rename_wins_over_the_file(server: SpinTestServer):
    user = await _create_seed_user()

    result = await _import(user.id, _package(), name="Maria (imported)")

    assert result.expert.name == "Maria (imported)"


async def test_skills_are_written_into_the_experts_own_folder(server: SpinTestServer):
    user = await _create_seed_user()

    result = await _import(user.id, _skill_package())

    assert result.failed_skills == []
    stored = await read_user_skill_package(
        user.id, "research", expert_id=result.expert.id
    )
    assert stored is not None
    assert [f.relative_path for f in stored.files] == ["refs/API.md"]
    assert sorted(result.expert.skills) == ["digest", "research"]


async def test_a_skill_the_user_removed_is_not_written(server: SpinTestServer):
    user = await _create_seed_user()

    result = await _import(user.id, _skill_package(), removed_skill_slugs=["digest"])

    assert result.expert.skills == ["research"]
    assert (
        await read_user_skill_package(user.id, "digest", expert_id=result.expert.id)
        is None
    )


async def test_a_skill_that_cannot_be_written_is_reported_not_fatal(
    server: SpinTestServer, mocker
):
    """An expert missing one of its skills is far more use than no expert."""
    user = await _create_seed_user()
    mocker.patch(
        "backend.api.features.experts.package_import.store_user_skill",
        side_effect=SkillLimitError("full"),
    )

    result = await _import(user.id, _skill_package())

    assert result.failed_skills == ["Research", "Digest"]
    assert result.expert.id


async def test_a_published_workflow_is_installed_from_the_marketplace(
    server: SpinTestServer,
):
    user = await _create_seed_user()
    version_id = await _seed_store_listing(server)

    result = await _import(
        user.id,
        _package(
            PackagedWorkflow(name="Morning digest", store_listing_version_id=version_id)
        ),
    )

    assert result.failed_workflows == []
    workflow = result.expert.workflows[0]
    assert workflow.store_listing_version_id == version_id
    assert workflow.library_agent_id is not None


async def test_an_embedded_graph_becomes_the_importers_own_agent(
    server: SpinTestServer,
):
    """Ids are reassigned, links included, so the copy is the importer's own
    graph rather than a second reference to somebody else's."""
    user = await _create_seed_user()
    packaged = _two_node_graph()

    result = await _import(
        user.id, _package(PackagedWorkflow(name=packaged.name, graph=packaged))
    )

    assert result.failed_workflows == []
    workflow = result.expert.workflows[0]
    assert workflow.store_listing_version_id is None
    assert workflow.graph_id is not None and workflow.graph_id != packaged.id
    stored = await graph_db.get_graph(workflow.graph_id, None, user_id=user.id)
    assert stored is not None
    assert len(stored.nodes) == 2 and len(stored.links) == 1
    assert {n.id for n in stored.nodes} == {
        stored.links[0].source_id,
        stored.links[0].sink_id,
    }
    assert {n.id for n in stored.nodes}.isdisjoint({n.id for n in packaged.nodes})


async def test_a_workflow_the_user_removed_is_not_installed(server: SpinTestServer):
    user = await _create_seed_user()
    version_id = await _seed_store_listing(server)

    result = await _import(
        user.id,
        _package(
            PackagedWorkflow(name="Kept", store_listing_version_id=version_id),
            PackagedWorkflow(name="Dropped", graph=_two_node_graph()),
        ),
        removed_workflow_indices=[1],
    )

    assert [w.store_listing_version_id for w in result.expert.workflows] == [version_id]


async def test_a_workflow_that_cannot_be_installed_is_reported_not_fatal(
    server: SpinTestServer,
):
    user = await _create_seed_user()

    result = await _import(
        user.id,
        _package(PackagedWorkflow(name="Ghost", store_listing_version_id="not-here")),
    )

    assert result.failed_workflows == ["Ghost"]
    assert result.expert.workflows == []


async def test_a_schedule_is_only_created_when_the_user_turned_it_on(
    server: SpinTestServer, mocker
):
    """The cadence travels in the file, but starting it is the importer's
    decision — a file should not silently begin running on upload."""
    create = mocker.patch(
        "backend.api.features.experts.package_import.scheduling.create_workflow_schedule",
        return_value=True,
    )
    user = await _create_seed_user()
    version_id = await _seed_store_listing(server)
    package = _package(
        PackagedWorkflow(
            name="Morning digest",
            store_listing_version_id=version_id,
            schedule_cron="40 7 * * *",
        )
    )

    await _import(user.id, package)
    create.assert_not_awaited()

    await _import(
        user.id,
        package,
        workflows=[ExpertImportWorkflowEdit(index=0, schedule_enabled=True)],
    )
    assert create.await_args.kwargs["cron"] == "40 7 * * *"


async def test_an_embedded_avatar_goes_through_the_media_pipeline(
    server: SpinTestServer, mocker
):
    """A package is user-supplied input whoever it came from, so its picture
    gets the magic-byte check and the virus scan like any other upload."""
    upload = mocker.patch(
        "backend.api.features.experts.package_import.upload_media",
        return_value="/api/store/media/u/images/a.png",
    )
    user = await _create_seed_user()
    package = _package(avatar=PackagedAvatar(kind="file", path="avatar.png"))
    package = package.model_copy(update={"avatar_bytes": b"\x89PNG"})

    result = await _import(user.id, package)

    assert result.expert.avatar_url == "/api/store/media/u/images/a.png"
    assert upload.await_args.args[0] == user.id


@pytest.mark.parametrize(
    "url, expected",
    [
        ("/experts/maria.svg", "/experts/maria.svg"),
        ("https://cdn.example/maria.png", None),
    ],
    ids=["site-relative-kept", "absolute-dropped"],
)
async def test_a_url_avatar_is_kept_only_when_it_is_ours(
    server: SpinTestServer, url: str, expected: str | None
):
    user = await _create_seed_user()

    result = await _import(
        user.id, _package(avatar=PackagedAvatar(kind="url", url=url))
    )

    assert result.expert.avatar_url == expected


async def test_an_exported_expert_survives_the_whole_round_trip(
    server: SpinTestServer,
):
    """Export, zip, parse, import — with a real multi-node graph, which is the
    only way the embedded-graph path is proven end to end."""
    user = await _create_seed_user()
    packaged = _two_node_graph()
    original = _package(
        PackagedWorkflow(name=packaged.name, graph=packaged),
        skills=[PackagedSkill(slug="research", name="Research")],
    ).model_copy(update={"skills": {"research": SkillPackage(skill_md=SKILL_MD)}})

    restored = package_from_zip(zip_from_package(original))
    preview = await preview_package(user.id, restored)
    result = await _import(user.id, restored)

    assert [w.source for w in preview.workflows] == ["graph"]
    assert result.failed_workflows == [] and result.failed_skills == []
    assert result.expert.name == "Maria Ops"
    assert len(result.expert.workflows) == 1
    assert result.expert.skills == ["research"]
