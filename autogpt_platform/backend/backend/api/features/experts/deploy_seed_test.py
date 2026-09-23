"""The deploy-time seed: refuses a schema that is behind, and converges.

The DB tests seed a two-skill catalog from a temp folder and a one-expert
roster under a random name, so they never touch the live roster or need the
private catalog repo.
"""

import asyncio
import functools
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import prisma.models
import pytest
import pytest_asyncio

from backend.api.features.experts import deploy_seed, experts_db, seed
from backend.api.features.store import skill_seed
from backend.data.db import prisma as db_client
from backend.data.user import get_or_create_user
from backend.util.test import SpinTestServer


@pytest_asyncio.fixture(autouse=True)
async def absorb_a_stale_event_loop(server: SpinTestServer):
    """An earlier test can leave the shared Prisma client bound to a loop that
    has since closed; only the first query on the new loop fails, and the engine
    re-establishes itself. Spend that failure here rather than in a test."""
    try:
        await db_client.execute_raw("SELECT 1")
    except RuntimeError as error:
        if "Event loop is closed" not in str(error):
            raise


def _write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_skill(root: Path, slug: str, body: str) -> None:
    _write(
        root,
        f"skills/{slug}/SKILL.md",
        f'---\nname: {slug}\ndescription: "The {slug} skill."\n---\n\n{body}\n',
    )


def _write_catalog(root: Path, slugs: list[str]) -> None:
    lines = ["skills:"]
    for slug in slugs:
        lines += [f"  - slug: {slug}", "    categories: [content]"]
        _write_skill(root, slug, f"# {slug}\n\nFirst edition.")
    _write(root, "catalog.yml", "\n".join(lines) + "\n")


def _entry(name: str, bundled: list[str]) -> seed.RosterEntry:
    return {
        "name": name,
        "role": "SEO & Content",
        "job_title": "SEO Content Manager",
        "tagline": "Takes a keyword from brief to article.",
        "avatar_url": "/experts/maria.svg",
        "bio": "An SEO and content strategist.",
        "bundled_skills": bundled,
        "categories": ["marketing"],
        "identity": "You are a content strategist.",
        "voice_preferences": "Clear and confident.",
        "voice_samples": [],
        "boundaries": "Never invent customer evidence.",
        "day_one": [],
        "preloads": [],
        "routines": [],
    }


@pytest.fixture
async def stub_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A catalog of two skills and a roster of one expert bundling both."""
    suffix = uuid.uuid4().hex[:8]
    slugs = [f"deploy-seed-a-{suffix}", f"deploy-seed-b-{suffix}"]
    name = f"Deploy Seed {suffix}"
    _write_catalog(tmp_path, slugs)
    monkeypatch.setattr(
        skill_seed,
        "seed_catalog_skills",
        functools.partial(skill_seed.seed_catalog_skills, tmp_path),
    )
    monkeypatch.setattr(seed, "ROSTER", [_entry(name, slugs)])
    monkeypatch.setattr(experts_db, "is_feature_enabled", AsyncMock(return_value=False))
    users: list[str] = []
    yield {"root": tmp_path, "slugs": slugs, "name": name, "users": users}

    templates = await prisma.models.Expert.prisma().find_many(
        where={"isTemplate": True, "name": name}
    )
    template_ids = [t.id for t in templates]
    await prisma.models.Expert.prisma().delete_many(
        where={"sourceTemplateId": {"in": template_ids}}
    )
    await prisma.models.Expert.prisma().delete_many(where={"id": {"in": template_ids}})
    await prisma.models.SkillListingVersion.prisma().delete_many(
        where={"SkillListing": {"is": {"slug": {"in": slugs}}}}
    )
    await prisma.models.SkillListing.prisma().delete_many(where={"slug": {"in": slugs}})
    await prisma.models.User.prisma().delete_many(where={"id": {"in": users}})


async def _snapshot(name: str, slugs: list[str]) -> dict:
    """Every row the deploy seed writes, minus the timestamps it bumps."""
    templates = await prisma.models.Expert.prisma().find_many(
        where={"isTemplate": True, "name": name},
        include={"BundledSkills": True},
        order={"id": "asc"},
    )
    hires = await prisma.models.Expert.prisma().find_many(
        where={"sourceTemplateId": {"in": [t.id for t in templates]}},
        order={"id": "asc"},
    )
    listings = await prisma.models.SkillListing.prisma().find_many(
        where={"slug": {"in": slugs}},
        include={"Versions": True},
        order={"slug": "asc"},
    )
    return {
        "templates": [
            (
                t.id,
                t.name,
                t.role,
                t.tagline,
                t.avatarUrl,
                sorted((b.skillListingId, b.position) for b in (t.BundledSkills or [])),
            )
            for t in templates
        ],
        "hires": [(h.id, h.name, h.avatarUrl, h.tagline) for h in hires],
        "listings": [
            (
                listing.id,
                listing.slug,
                listing.activeVersionId,
                sorted(
                    ((v.id, v.version, v.body) for v in (listing.Versions or [])),
                    key=lambda version: version[1],
                ),
            )
            for listing in listings
        ],
    }


async def _hire(env: dict, template_id: str) -> prisma.models.Expert:
    user = await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"deploy-seed-{uuid.uuid4().hex[:8]}@example.com",
            "name": "Seed Owner",
        }
    )
    env["users"].append(user.id)
    result = await experts_db.hire_expert(user.id, template_id, None)
    hired = await prisma.models.Expert.prisma().find_unique(
        where={"id": result.expert.id}
    )
    assert hired is not None
    return hired


def test_migrations_on_disk_lists_only_migration_folders(tmp_path: Path):
    _write(tmp_path, "20260101000000_first/migration.sql", "SELECT 1;")
    _write(tmp_path, "20260102000000_second/migration.sql", "SELECT 1;")
    (tmp_path / "20260103000000_empty").mkdir()
    _write(tmp_path, "migration_lock.toml", 'provider = "postgresql"\n')

    assert deploy_seed.migrations_on_disk(tmp_path) == {
        "20260101000000_first",
        "20260102000000_second",
    }


def test_migrations_dir_points_at_the_backend_migrations():
    assert (deploy_seed.MIGRATIONS_DIR / "migration_lock.toml").is_file()
    assert deploy_seed.migrations_on_disk()


@pytest.mark.asyncio(loop_scope="session")
async def test_a_current_schema_passes(server: SpinTestServer):
    await deploy_seed.assert_schema_current()


@pytest.mark.asyncio(loop_scope="session")
async def test_a_pending_migration_stops_the_seed_before_any_write(
    server: SpinTestServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    for name in deploy_seed.migrations_on_disk():
        _write(tmp_path, f"{name}/migration.sql", "SELECT 1;")
    _write(tmp_path, "29990101000000_not_applied/migration.sql", "SELECT 1;")
    monkeypatch.setattr(deploy_seed, "MIGRATIONS_DIR", tmp_path)
    skills = AsyncMock()
    roster = AsyncMock()
    monkeypatch.setattr(skill_seed, "seed_catalog_skills", skills)
    monkeypatch.setattr(seed, "seed_roster", roster)

    with pytest.raises(
        deploy_seed.SchemaBehindError, match="pending: 29990101000000_not_applied"
    ):
        await deploy_seed.seed_experts()

    skills.assert_not_awaited()
    roster.assert_not_awaited()


@pytest.mark.asyncio(loop_scope="session")
async def test_a_failed_migration_stops_the_seed(monkeypatch: pytest.MonkeyPatch):
    applied = [
        {
            "migration_name": name,
            "finished_at": datetime.now(timezone.utc),
            "rolled_back_at": None,
        }
        for name in deploy_seed.migrations_on_disk()
    ]
    applied[-1] = {**applied[-1], "finished_at": None}
    monkeypatch.setattr(
        deploy_seed.database,
        "query_raw_with_schema",
        AsyncMock(return_value=applied),
    )

    with pytest.raises(
        deploy_seed.SchemaBehindError,
        match=f"failed: {applied[-1]['migration_name']};",
    ):
        await deploy_seed.assert_schema_current()


@pytest.mark.asyncio(loop_scope="session")
async def test_running_the_deploy_seed_twice_changes_nothing(
    server: SpinTestServer, stub_environment: dict
):
    env = stub_environment
    (template_id,) = await deploy_seed.seed_experts()
    hired = await _hire(env, template_id)
    await prisma.models.Expert.prisma().update(
        where={"id": hired.id},
        data={"name": "My content lead", "avatarUrl": "/avatars/mine.svg"},
    )
    first = await _snapshot(env["name"], env["slugs"])

    assert await deploy_seed.seed_experts() == [template_id]

    assert await _snapshot(env["name"], env["slugs"]) == first
    assert len(first["templates"]) == 1
    assert len(first["templates"][0][5]) == 2
    assert first["hires"] == [
        (hired.id, "My content lead", "/avatars/mine.svg", hired.tagline)
    ]
    assert all(len(listing[3]) == 1 for listing in first["listings"])


@pytest.mark.asyncio(loop_scope="session")
async def test_overlapping_deploys_create_one_template(
    server: SpinTestServer, stub_environment: dict
):
    env = stub_environment
    first, second = await asyncio.gather(
        deploy_seed.seed_experts(), deploy_seed.seed_experts()
    )

    assert first == second
    templates = await prisma.models.Expert.prisma().find_many(
        where={"isTemplate": True, "name": env["name"]}
    )
    assert len(templates) == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_a_changed_skill_gets_a_new_version_on_the_next_deploy(
    server: SpinTestServer, stub_environment: dict, monkeypatch: pytest.MonkeyPatch
):
    env = stub_environment
    changed, dropped = env["slugs"]
    (template_id,) = await deploy_seed.seed_experts()
    before = await _snapshot(env["name"], env["slugs"])

    _write_skill(env["root"], changed, f"# {changed}\n\nSecond edition.")
    monkeypatch.setattr(seed, "ROSTER", [_entry(env["name"], [changed])])
    assert await deploy_seed.seed_experts() == [template_id]
    after = await _snapshot(env["name"], env["slugs"])

    listing_before = {listing[1]: listing for listing in before["listings"]}
    listing_after = {listing[1]: listing for listing in after["listings"]}
    (old_version,) = listing_before[changed][3]
    kept, new_version = listing_after[changed][3]
    assert listing_after[changed][0] == listing_before[changed][0]
    assert kept == old_version
    assert (new_version[1], listing_after[changed][2]) == (2, new_version[0])
    assert "Second edition." in new_version[2]
    assert listing_after[dropped] == listing_before[dropped]
    assert after["templates"][0][0] == template_id
    assert after["templates"][0][5] == [(listing_after[changed][0], 0)]


@pytest.mark.asyncio(loop_scope="session")
async def test_a_missing_preload_workflow_fails_before_any_write(
    server: SpinTestServer, stub_environment: dict, monkeypatch: pytest.MonkeyPatch
):
    env = stub_environment
    missing = f"unpublished-workflow-{uuid.uuid4().hex[:8]}"
    entry = _entry(env["name"], env["slugs"])
    entry["preloads"] = [{"slug": missing, "cron": None}]
    monkeypatch.setattr(seed, "ROSTER", [entry])

    with pytest.raises(RuntimeError, match=f"missing roster listings for: {missing}"):
        await deploy_seed.seed_experts()

    assert await _snapshot(env["name"], env["slugs"]) == {
        "templates": [],
        "hires": [],
        "listings": [],
    }
