"""A catalog edit reaches installed copies: whole when the copy is untouched,
three-way merged when its owner edited it (``copilot/tools/skill_updates``).
Seeds a one-skill catalog from a temp folder, installs it into a fake
workspace, then seeds a new edition. Lives here rather than beside the module
because ``copilot/tools`` tests run without a database."""

import uuid
from pathlib import Path

import prisma.models
import pytest

from backend.api.features.store import skill_db, skill_seed
from backend.copilot.tools.skills import (
    ParsedSkill,
    _skill_md_path,
    list_user_skills,
    read_user_skill_with_body,
    store_user_skill,
)
from backend.copilot.tools.skills_test import _FakeWorkspaceManager, _patch_skills_path
from backend.data.user import get_or_create_user
from backend.util.test import SpinTestServer

DESCRIPTION = "Turn a keyword into a brief."
BODY = (
    "# Brief\n\nStep one: read the page.\nStep two: list keywords.\n\n"
    "Step three: draft.\nStep four: review."
)


class Catalog:
    def __init__(self, root: Path):
        self.root = root
        self.slug = f"update-probe-{uuid.uuid4().hex[:8]}"

    async def publish(
        self, body: str = BODY, description: str = DESCRIPTION, triggers: str = "brief"
    ) -> None:
        (self.root / "skills" / self.slug).mkdir(parents=True, exist_ok=True)
        (self.root / "catalog.yml").write_text(
            f"skills:\n  - slug: {self.slug}\n    categories: [content]\n"
        )
        (self.root / "skills" / self.slug / "SKILL.md").write_text(
            f'---\nname: {self.slug}\ndescription: "{description}"\n'
            f"triggers: [{triggers}]\n---\n\n{body}\n"
        )
        await skill_seed.seed_catalog_skills(self.root)

    async def versions(self) -> list[prisma.models.SkillListingVersion]:
        return await prisma.models.SkillListingVersion.prisma().find_many(
            where={"SkillListing": {"is": {"slug": self.slug}}},
            order={"version": "asc"},
        )


@pytest.fixture
async def catalog(server: SpinTestServer, tmp_path: Path):
    made = Catalog(tmp_path)
    yield made
    await prisma.models.SkillListingVersion.prisma().delete_many(
        where={"SkillListing": {"is": {"slug": made.slug}}}
    )
    await prisma.models.SkillListing.prisma().delete_many(where={"slug": made.slug})


@pytest.fixture
async def owner(server: SpinTestServer):
    return await get_or_create_user(
        {
            "sub": str(uuid.uuid4()),
            "email": f"skill-updates-{uuid.uuid4().hex[:8]}@example.com",
            "name": "Skill Owner",
        }
    )


async def _read(owner_id: str, slug: str) -> tuple[ParsedSkill, ParsedSkill]:
    """The copy as the index lists it (after any update), and with its body."""
    (listed,) = [s for s in await list_user_skills(owner_id) if s.name == slug]
    copy = await read_user_skill_with_body(owner_id, slug)
    assert copy is not None
    return listed, copy


@pytest.mark.asyncio(loop_scope="session")
async def test_the_seed_adds_a_version_only_when_the_content_changes(catalog):
    await catalog.publish()
    await catalog.publish()
    (first,) = await catalog.versions()

    await catalog.publish(body=BODY.replace("draft.", "draft it."))
    old, new = await catalog.versions()

    assert (old.id, old.version, old.body, old.updatedAt) == (
        first.id,
        1,
        first.body,
        first.updatedAt,
    )
    assert new.version == 2
    assert "draft it." in new.body
    listing = await prisma.models.SkillListing.prisma().find_unique(
        where={"slug": catalog.slug}
    )
    assert listing is not None and listing.activeVersionId == new.id


@pytest.mark.asyncio(loop_scope="session")
async def test_an_untouched_copy_takes_the_new_version(catalog, owner):
    await catalog.publish()
    with _patch_skills_path(_FakeWorkspaceManager()):
        await skill_db.install_marketplace_skill(owner.id, catalog.slug)
        await catalog.publish(
            body=BODY.replace("review.", "review against the brief."),
            description="Turn a keyword into a ranked brief.",
        )
        listed, copy = await _read(owner.id, catalog.slug)

    assert "review against the brief." in copy.body
    assert copy.description == "Turn a keyword into a ranked brief."
    assert listed.installed_version == "2"


@pytest.mark.asyncio(loop_scope="session")
async def test_an_edited_copy_keeps_its_edits_and_takes_the_rest(catalog, owner):
    await catalog.publish()
    workspace = _FakeWorkspaceManager()
    with _patch_skills_path(workspace):
        await skill_db.install_marketplace_skill(owner.id, catalog.slug)
        await store_user_skill(
            owner.id,
            name=catalog.slug,
            description="My own description.",
            body=BODY.replace("read the page.", "read the page twice."),
            triggers=["brief"],
        )
        await catalog.publish(
            body=BODY.replace("review.", "review against the brief."),
            description="Turn a keyword into a ranked brief.",
            triggers="brief, seo brief",
        )
        listed, copy = await _read(owner.id, catalog.slug)
        metadata = workspace.metadata[_skill_md_path(catalog.slug)]

    assert "read the page twice." in copy.body
    assert "review against the brief." in copy.body
    assert copy.description == "My own description."
    assert copy.triggers == ("brief", "seo brief")
    assert listed.installed_version == "2"
    assert [c["field"] for c in metadata["update_conflicts"]] == ["description"]


@pytest.mark.asyncio(loop_scope="session")
async def test_an_overlapping_edit_keeps_the_owners_lines(catalog, owner):
    await catalog.publish()
    workspace = _FakeWorkspaceManager()
    mine = BODY.replace("Step three: draft.", "Step three: draft by hand.")
    with _patch_skills_path(workspace):
        await skill_db.install_marketplace_skill(owner.id, catalog.slug)
        await store_user_skill(
            owner.id, name=catalog.slug, description=DESCRIPTION, body=mine
        )
        await catalog.publish(
            body=BODY.replace("Step three: draft.", "Step three: use the template.")
        )
        _, copy = await _read(owner.id, catalog.slug)
        metadata = workspace.metadata[_skill_md_path(catalog.slug)]

    assert copy.body.strip() == mine
    assert metadata["update_conflicts"] == [
        {
            "field": "body",
            "base": "Step three: draft.\n",
            "ours": "Step three: use the template.\n",
            "theirs": "Step three: draft by hand.\n",
        }
    ]


@pytest.mark.asyncio(loop_scope="session")
async def test_an_owner_edit_keeps_the_install_mark(catalog, owner):
    await catalog.publish()
    with _patch_skills_path(_FakeWorkspaceManager()):
        await skill_db.install_marketplace_skill(owner.id, catalog.slug)
        edited = await store_user_skill(
            owner.id, name=catalog.slug, description="Mine.", body=BODY
        )
        (listed,) = [
            s for s in await list_user_skills(owner.id) if s.name == catalog.slug
        ]

    assert edited.installed_version == "1"
    assert (listed.origin, listed.installed_version) == ("user", "1")


@pytest.mark.asyncio(loop_scope="session")
async def test_a_community_listing_does_not_update_installed_copies(catalog, owner):
    await catalog.publish()
    await catalog.publish(body=BODY + "\nMore.")
    assert [
        u.latest.version for u in await skill_db.get_skill_updates({catalog.slug: 1})
    ] == [2]

    await prisma.models.SkillListing.prisma().update(
        where={"slug": catalog.slug}, data={"owningUserId": owner.id}
    )

    assert await skill_db.get_skill_updates({catalog.slug: 1}) == []
