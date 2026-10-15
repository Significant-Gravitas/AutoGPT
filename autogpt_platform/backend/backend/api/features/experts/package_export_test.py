import io
import json
import zipfile
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

import backend.util.json
from backend.api.features.experts import package_export
from backend.api.features.experts.models import (
    Expert,
    ExpertBundledSkill,
    ExpertDayOneItem,
    ExpertRoutine,
    ExpertTemplate,
    ExpertWorkflowRef,
    VoiceSample,
)
from backend.api.features.experts.package_export import (
    build_template_package,
    zip_from_expert_package,
)
from backend.api.features.experts.package_model import ExpertManifest
from backend.api.features.store.skill_model import SkillVersionPackage
from backend.blocks.basic import StoreValueBlock
from backend.copilot.tools.skills import SkillFile
from backend.data.graph import GraphModel, Link, NodeModel
from backend.util.exceptions import NotFoundError

TEMPLATE_ID = "tmpl-1"
SKILL_MD = {
    "write-briefs": "---\nname: write-briefs\ndescription: Drafts briefs\n---\nBody A\n",
    "run-checks": "---\nname: run-checks\ndescription: Runs checks\n---\nBody B\n",
}
SCRIPT = b"#!/bin/sh\necho checked\n"


def _workflow_ref(listing_id: str | None, name: str, cron: str | None) -> dict:
    return ExpertWorkflowRef(
        id=f"ew-{name}",
        store_listing_version_id=listing_id,
        library_agent_id=None,
        graph_id=None,
        name=name,
        description=f"{name} description",
        schedule_cron=cron,
        integration_providers=["github"] if cron else [],
    ).model_dump()


def _template(workflows: list[dict] | None = None) -> Expert:
    return Expert(
        id=TEMPLATE_ID,
        name="Ada Lovelace",
        avatar_url="/experts/ada.png",
        color="violet",
        role="Research assistant",
        job_title="Analyst",
        tagline="Finds the signal",
        bio="Ada reads everything.",
        skills=[],
        categories=["research"],
        identity="You are Ada.",
        voice_preferences="Plain and short.",
        voice_samples=[VoiceSample(label="Crisp", text="Done.")],
        day_one=[ExpertDayOneItem(title="Read your docs", timing="Day 1")],
        boundaries="No legal advice.",
        protected_soul_rules=["rule"],
        is_template=True,
        source_template_id=None,
        is_archived=False,
        workflows=workflows
        or [
            _workflow_ref("slv-digest", "Daily digest", "0 9 * * *"),
            _workflow_ref("slv-report", "Weekly report", None),
        ],
    )


def _graph(graph_id: str, name: str) -> GraphModel:
    nodes = [
        NodeModel(
            id=f"{graph_id}-n{i}",
            block_id=StoreValueBlock().id,
            input_default={"input": i},
            graph_id=graph_id,
            graph_version=1,
        )
        for i in (1, 2)
    ]
    return GraphModel(
        id=graph_id,
        version=1,
        name=name,
        description=f"{name} graph",
        user_id="",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        nodes=nodes,
        links=[
            Link(
                source_id=nodes[0].id,
                sink_id=nodes[1].id,
                source_name="output",
                sink_name="data",
            )
        ],
    )


GRAPHS = {
    "slv-digest": _graph("g-digest", "Daily digest"),
    "slv-report": _graph("g-report", "Weekly report"),
}


def _version(slug: str, files: list[SkillFile]) -> SkillVersionPackage:
    return SkillVersionPackage(
        version_id=f"v-{slug}",
        listing_id=f"l-{slug}",
        slug=slug,
        version=1,
        package_sha256=None,
        skill_markdown=SKILL_MD[slug],
        files=files,
    )


LIVE_PACKAGES = {
    "l-write-briefs": _version("write-briefs", []),
    "l-run-checks": _version(
        "run-checks",
        [
            SkillFile(
                relative_path="scripts/run.sh", content=SCRIPT, is_executable=True
            ),
            SkillFile(relative_path="references/GUIDE.md", content=b"# Guide\n"),
        ],
    ),
}


async def _get_agent(listing_id: str) -> GraphModel:
    if listing_id not in GRAPHS:
        raise NotFoundError(f"Store listing version {listing_id} not found")
    return GRAPHS[listing_id]


@pytest.fixture
def sources(monkeypatch: pytest.MonkeyPatch) -> dict[str, AsyncMock]:
    bundled = [
        ExpertBundledSkill(
            id=f"l-{slug}", slug=slug, title=slug.title(), description=f"{slug} card"
        )
        for slug in SKILL_MD
    ]

    async def with_bundled_skills(templates, user_id):
        return [
            ExpertTemplate(**t.model_dump(), bundled_skills=bundled) for t in templates
        ]

    mocks = {
        "get_template": AsyncMock(return_value=_template()),
        "with_bundled_skills": AsyncMock(side_effect=with_bundled_skills),
        "get_live_packages": AsyncMock(return_value=LIVE_PACKAGES),
        "get_agent": AsyncMock(side_effect=_get_agent),
        "list_template_routines": AsyncMock(
            return_value=[
                ExpertRoutine(
                    id="r-1",
                    expert_id=TEMPLATE_ID,
                    key="morning",
                    title="Morning sweep",
                    prompt="Sweep the inbox.",
                    crons=["0 8 * * 1-5"],
                    asks=["Which inbox?"],
                    session_mode="THREAD",
                )
            ]
        ),
    }
    for name in ("get_template", "with_bundled_skills"):
        monkeypatch.setattr(package_export.experts_db, name, mocks[name])
    monkeypatch.setattr(
        package_export.skill_db, "get_live_packages", mocks["get_live_packages"]
    )
    monkeypatch.setattr(package_export.store_db, "get_agent", mocks["get_agent"])
    monkeypatch.setattr(
        package_export.routines,
        "list_template_routines",
        mocks["list_template_routines"],
    )
    return mocks


async def _zip() -> zipfile.ZipFile:
    package = await build_template_package(TEMPLATE_ID, "user-1")
    return zipfile.ZipFile(io.BytesIO(zip_from_expert_package(package)))


def _manifest(archive: zipfile.ZipFile) -> ExpertManifest:
    return ExpertManifest.model_validate_json(archive.read("expert.json"))


async def test_zip_holds_exactly_the_manifest_skills_and_workflows(sources):
    archive = await _zip()
    assert sorted(archive.namelist()) == [
        "expert.json",
        "skills/run-checks/SKILL.md",
        "skills/run-checks/references/GUIDE.md",
        "skills/run-checks/scripts/run.sh",
        "skills/write-briefs/SKILL.md",
        "workflows/01-daily-digest.json",
        "workflows/02-weekly-report.json",
    ]


async def test_every_bundled_skill_is_packaged_whole(sources):
    archive = await _zip()
    for slug, skill_md in SKILL_MD.items():
        assert archive.read(f"skills/{slug}/SKILL.md").decode() == skill_md
    assert archive.read("skills/run-checks/scripts/run.sh") == SCRIPT
    script_mode = archive.getinfo("skills/run-checks/scripts/run.sh").external_attr
    assert (script_mode >> 16) & 0o111
    guide_mode = archive.getinfo("skills/run-checks/references/GUIDE.md")
    assert not (guide_mode.external_attr >> 16) & 0o111
    cards = _manifest(archive).skills
    assert [(c.slug, c.path) for c in cards] == [
        ("write-briefs", "skills/write-briefs/"),
        ("run-checks", "skills/run-checks/"),
    ]
    sources["with_bundled_skills"].assert_awaited_once()
    assert sources["with_bundled_skills"].await_args.args[1] == "user-1"


async def test_every_workflow_is_its_agent_file(sources):
    archive = await _zip()
    entries = _manifest(archive).workflows
    assert [e.path for e in entries] == [
        "workflows/01-daily-digest.json",
        "workflows/02-weekly-report.json",
    ]
    for entry, listing_id in zip(entries, ["slv-digest", "slv-report"]):
        raw = archive.read(entry.path)
        assert raw.decode() == backend.util.json.dumps(GRAPHS[listing_id])
        graph = json.loads(raw)
        assert graph["name"] and graph["description"] is not None
        assert len(graph["nodes"]) == 2 and len(graph["links"]) == 1
        assert entry.store_listing_version_id == listing_id
    assert entries[0].schedule_cron == "0 9 * * *"
    assert entries[0].integration_providers == ["github"]
    assert entries[1].schedule_cron is None


async def test_manifest_carries_persona_soul_and_routines(sources):
    manifest = _manifest(await _zip())
    assert manifest.format_version == 1
    assert (manifest.name, manifest.role, manifest.job_title) == (
        "Ada Lovelace",
        "Research assistant",
        "Analyst",
    )
    assert manifest.avatar_url == "/experts/ada.png"
    assert manifest.identity == "You are Ada."
    assert [s.label for s in manifest.voice_samples] == ["Crisp"]
    assert [d.title for d in manifest.day_one] == ["Read your docs"]
    assert manifest.boundaries == "No legal advice."
    [routine] = manifest.routines
    assert (routine.key, routine.crons, routine.asks) == (
        "morning",
        ["0 8 * * 1-5"],
        ["Which inbox?"],
    )


async def test_two_builds_are_byte_identical(sources):
    first = zip_from_expert_package(await build_template_package(TEMPLATE_ID, "u"))
    second = zip_from_expert_package(await build_template_package(TEMPLATE_ID, "u"))
    assert first == second


async def test_missing_template_is_not_found(sources):
    sources["get_template"].return_value = None
    with pytest.raises(NotFoundError):
        await build_template_package(TEMPLATE_ID, "user-1")


async def test_a_delisted_workflow_is_left_out_and_the_rest_stay(sources):
    sources["get_template"].return_value = _template(
        [
            _workflow_ref("slv-gone", "Gone", None),
            _workflow_ref("slv-report", "Weekly report", None),
            _workflow_ref(None, "Unlisted", None),
        ]
    )
    archive = await _zip()
    assert [e.path for e in _manifest(archive).workflows] == [
        "workflows/01-weekly-report.json"
    ]
    assert [n for n in archive.namelist() if n.startswith("workflows/")] == [
        "workflows/01-weekly-report.json"
    ]


async def test_skills_are_loaded_by_the_listings_the_page_shows(sources):
    await _zip()
    sources["get_live_packages"].assert_awaited_once_with(
        ["l-write-briefs", "l-run-checks"]
    )


async def test_a_skill_that_leaves_the_hub_mid_download_is_left_out(sources):
    sources["get_live_packages"].return_value = {
        "l-run-checks": LIVE_PACKAGES["l-run-checks"]
    }
    archive = await _zip()
    assert [c.slug for c in _manifest(archive).skills] == ["run-checks"]
    assert not any(n.startswith("skills/write-briefs/") for n in archive.namelist())


async def test_no_skills_when_the_hub_bundles_none(sources):
    async def none_bundled(templates, user_id):
        return [ExpertTemplate(**t.model_dump()) for t in templates]

    sources["with_bundled_skills"].side_effect = none_bundled
    archive = await _zip()
    assert _manifest(archive).skills == []
    assert not any(n.startswith("skills/") for n in archive.namelist())
