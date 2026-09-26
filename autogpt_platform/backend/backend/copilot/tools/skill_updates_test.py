"""Marketplace copies catch up with the marketplace on a cold turn: an
unedited copy is replaced, an edited one is merged with the owner's side
winning, a retired one is left alone and flagged. All of it rides on the
baseline an install records and the checksums the workspace already keeps."""

import hashlib
from unittest.mock import AsyncMock, patch

import pytest

from backend.api.features.store.skill_model import (
    ActiveSkillVersion,
    SkillVersionPackage,
)
from backend.copilot.tools import skills as skills_module
from backend.copilot.tools.skills import (
    _RECONCILE_WRITE_BUDGET,
    SKILL_ORIGIN_MARKETPLACE,
    SKILL_ORIGIN_USER,
    UPDATE_AVAILABLE,
    UPDATE_MERGED,
    UPDATE_RETIRED,
    SkillBaseline,
    SkillFile,
    build_skills_update_notice,
    list_user_skills,
    render_skills_index,
    store_user_skill,
)
from backend.copilot.tools.skills_test import _FakeWorkspaceManager, _patch_skills_path
from backend.data.skill_package import package_tree_sha256

USER = "user-1"


def _skill_md(slug: str, body: str) -> str:
    return f"---\nname: {slug}\ndescription: {slug} description\n---\n\n{body}"


def _package(
    slug: str,
    version_id: str,
    body: str,
    files: dict[str, bytes] | None = None,
) -> SkillVersionPackage:
    """*version_id* is "v<n>"; the id stored is unique per slug, as real
    version rows are, so several skills can share a version number."""
    text = _skill_md(slug, body)
    siblings = [
        SkillFile(relative_path=path, content=content)
        for path, content in sorted((files or {}).items())
    ]
    hashed = [("SKILL.md", hashlib.sha256(text.encode()).hexdigest(), False)]
    hashed += [
        (f.relative_path, hashlib.sha256(f.content).hexdigest(), False)
        for f in siblings
    ]
    return SkillVersionPackage(
        version_id=f"{slug}-{version_id}",
        listing_id=f"listing-{slug}",
        slug=slug,
        version=int(version_id[1:]),
        package_sha256=package_tree_sha256(hashed),
        skill_markdown=text,
        files=siblings,
    )


def _active(
    package: SkillVersionPackage, *, retired: bool = False
) -> ActiveSkillVersion:
    return ActiveSkillVersion(
        listing_id=package.listing_id,
        version_id=package.version_id,
        package_sha256=package.package_sha256,
        retired=retired,
    )


def _baseline(package: SkillVersionPackage) -> SkillBaseline:
    return SkillBaseline(package.listing_id, package.version_id, package.package_sha256)


async def _install(package: SkillVersionPackage) -> None:
    await store_user_skill(
        USER,
        name=package.slug,
        description="",
        body="",
        files=list(package.files),
        origin=SKILL_ORIGIN_MARKETPLACE,
        skill_markdown=package.skill_markdown,
        baseline=_baseline(package),
    )


def _marketplace(
    patched: _patch_skills_path, *packages: SkillVersionPackage, retired=()
):
    patched.skill_db.get_active_versions = AsyncMock(
        return_value={p.slug: _active(p, retired=p.slug in retired) for p in packages}
    )
    patched.skill_db.get_version_packages = AsyncMock(
        return_value={p.version_id: p for p in packages}
    )


@pytest.mark.asyncio
async def test_an_install_records_the_baseline_on_the_root():
    fake = _FakeWorkspaceManager()
    v1 = _package("cold-email", "v1", "# Cold\n", {"references/a.md": b"a\n"})
    with _patch_skills_path(fake):
        await _install(v1)
        [skill] = await list_user_skills(USER)

    assert fake.files["/skills/cold-email/SKILL.md"] == v1.skill_markdown.encode()
    meta = fake.metadata["/skills/cold-email/SKILL.md"]
    assert meta["listing_version_id"] == v1.version_id
    assert meta["package_sha256"] == v1.package_sha256
    assert skill.baseline == _baseline(v1)
    assert skill.origin == SKILL_ORIGIN_MARKETPLACE
    assert skill.update is None


@pytest.mark.asyncio
async def test_an_unedited_copy_behind_the_marketplace_is_fast_forwarded():
    fake = _FakeWorkspaceManager()
    v1 = _package(
        "cold-email",
        "v1",
        "# Cold\n",
        {"references/a.md": b"a\n", "references/gone.md": b"x\n"},
    )
    v2 = _package(
        "cold-email",
        "v2",
        "# Cold, revised\n",
        {"references/a.md": b"a v2\n", "references/new.md": b"n\n"},
    )
    with _patch_skills_path(fake) as patched:
        await _install(v1)
        _marketplace(patched, v2)
        [skill] = await list_user_skills(USER)

    assert fake.files["/skills/cold-email/SKILL.md"] == v2.skill_markdown.encode()
    assert fake.files["/skills/cold-email/references/a.md"] == b"a v2\n"
    assert fake.files["/skills/cold-email/references/new.md"] == b"n\n"
    assert "/skills/cold-email/references/gone.md" not in fake.files
    assert skill.baseline == _baseline(v2)
    assert skill.update is None


@pytest.mark.asyncio
async def test_a_current_copy_is_left_alone():
    fake = _FakeWorkspaceManager()
    v1 = _package("cold-email", "v1", "# Cold\n")
    with _patch_skills_path(fake) as patched:
        await _install(v1)
        _marketplace(patched, v1)
        [skill] = await list_user_skills(USER)

    patched.skill_db.get_version_packages.assert_not_called()
    assert skill.update is None


@pytest.mark.asyncio
async def test_an_owners_edit_keeps_the_copy_a_marketplace_copy_with_its_baseline():
    fake = _FakeWorkspaceManager()
    v1 = _package("cold-email", "v1", "# Cold\n", {"references/a.md": b"a\n"})
    with _patch_skills_path(fake):
        await _install(v1)
        await store_user_skill(
            USER,
            name="cold-email",
            description="cold-email description",
            body="# Cold\n\nMy note.\n",
        )
        [skill] = await list_user_skills(USER)

    meta = fake.metadata["/skills/cold-email/SKILL.md"]
    assert meta["skill_origin"] == SKILL_ORIGIN_MARKETPLACE
    assert meta["listing_version_id"] == v1.version_id
    assert skill.origin == SKILL_ORIGIN_MARKETPLACE
    assert skill.baseline == _baseline(v1)
    # A single-file edit leaves the siblings in place.
    assert fake.files["/skills/cold-email/references/a.md"] == b"a\n"


@pytest.mark.asyncio
async def test_an_edited_copy_is_merged_with_the_update():
    fake = _FakeWorkspaceManager()
    v1 = _package(
        "cold-email",
        "v1",
        "# Cold\n\nintro\nkeep\noutro\n",
        {"references/a.md": b"a\n"},
    )
    v2 = _package(
        "cold-email",
        "v2",
        "# Cold\n\nnew intro\nkeep\noutro\n",
        {"references/a.md": b"a v2\n"},
    )
    with _patch_skills_path(fake) as patched:
        await _install(v1)
        await store_user_skill(
            USER,
            name="cold-email",
            description="cold-email description",
            body="# Cold\n\nintro\nkeep\nmy addition\noutro\n",
        )
        _marketplace(patched, v1, v2)
        [skill] = await list_user_skills(USER)

    body = fake.files["/skills/cold-email/SKILL.md"].decode()
    assert "new intro\nkeep\nmy addition\noutro" in body
    assert fake.files["/skills/cold-email/references/a.md"] == b"a v2\n"
    assert skill.update == UPDATE_MERGED
    assert skill.baseline is not None
    assert (skill.baseline.version_id, skill.baseline.merged_from) == (
        v2.version_id,
        v1.version_id,
    )
    assert skill.baseline.merge_conflicts == ()


@pytest.mark.asyncio
async def test_a_conflict_keeps_the_owners_side_and_says_so():
    fake = _FakeWorkspaceManager()
    v1 = _package("cold-email", "v1", "# Cold\n\nline one\nline two\n")
    v2 = _package("cold-email", "v2", "# Cold\n\nline one\nline two (upstream)\n")
    with _patch_skills_path(fake) as patched:
        await _install(v1)
        await store_user_skill(
            USER,
            name="cold-email",
            description="cold-email description",
            body="# Cold\n\nline one\nline two (mine)\n",
        )
        _marketplace(patched, v1, v2)
        [skill] = await list_user_skills(USER)

    assert "line two (mine)" in fake.files["/skills/cold-email/SKILL.md"].decode()
    assert "upstream" not in fake.files["/skills/cold-email/SKILL.md"].decode()
    assert skill.update == UPDATE_MERGED
    assert skill.baseline is not None and skill.baseline.merge_conflicts == (
        "SKILL.md",
    )


@pytest.mark.asyncio
async def test_a_retired_listing_leaves_the_copy_and_flags_it():
    fake = _FakeWorkspaceManager()
    v1 = _package("cold-email", "v1", "# Cold\n")
    with _patch_skills_path(fake) as patched:
        await _install(v1)
        _marketplace(patched, v1, retired={"cold-email"})
        [skill] = await list_user_skills(USER)

    assert fake.files["/skills/cold-email/SKILL.md"] == v1.skill_markdown.encode()
    assert skill.update == UPDATE_RETIRED


@pytest.mark.asyncio
async def test_a_copy_without_a_baseline_is_stamped_when_it_matches_the_marketplace():
    fake = _FakeWorkspaceManager()
    v1 = _package("cold-email", "v1", "# Cold\n", {"references/a.md": b"a\n"})
    with _patch_skills_path(fake) as patched:
        # Installed before baselines existed: same bytes, no install record.
        await store_user_skill(
            USER,
            name="cold-email",
            description="",
            body="",
            files=list(v1.files),
            origin=SKILL_ORIGIN_MARKETPLACE,
            skill_markdown=v1.skill_markdown,
        )
        assert "listing_version_id" not in fake.metadata["/skills/cold-email/SKILL.md"]
        _marketplace(patched, v1)
        [skill] = await list_user_skills(USER)

    assert skill.baseline == _baseline(v1)
    assert skill.update is None


@pytest.mark.asyncio
async def test_a_copy_without_a_baseline_that_differs_is_only_flagged():
    fake = _FakeWorkspaceManager()
    v1 = _package("cold-email", "v1", "# Cold\n")
    with _patch_skills_path(fake) as patched:
        await store_user_skill(
            USER,
            name="cold-email",
            description="",
            body="",
            files=[],
            origin=SKILL_ORIGIN_MARKETPLACE,
            skill_markdown=_skill_md(
                "cold-email", "# Cold, but not what is published\n"
            ),
        )
        _marketplace(patched, v1)
        [skill] = await list_user_skills(USER)

    assert skill.baseline is None
    assert skill.update == UPDATE_AVAILABLE
    assert "not what is published" in fake.files["/skills/cold-email/SKILL.md"].decode()


@pytest.mark.asyncio
async def test_a_turn_rewrites_a_bounded_number_of_copies():
    fake = _FakeWorkspaceManager()
    olds = [
        _package(f"skill-{i}", "v1", f"# {i}\n")
        for i in range(_RECONCILE_WRITE_BUDGET + 2)
    ]
    news = [
        _package(f"skill-{i}", "v2", f"# {i} v2\n")
        for i in range(_RECONCILE_WRITE_BUDGET + 2)
    ]
    with _patch_skills_path(fake) as patched:
        for old in olds:
            await _install(old)
        _marketplace(patched, *news)
        skills = await list_user_skills(USER)

    moved = [s for s in skills if s.baseline and s.baseline.version_id.endswith("-v2")]
    waiting = [s for s in skills if s.update == UPDATE_AVAILABLE]
    assert len(moved) == _RECONCILE_WRITE_BUDGET
    assert len(waiting) == 2


@pytest.mark.asyncio
async def test_a_reinstall_of_the_same_package_rewrites_nothing_but_the_root():
    class Counting(_FakeWorkspaceManager):
        def __init__(self):
            super().__init__()
            self.writes: list[str] = []

        async def write_file(self, **kwargs):
            self.writes.append(kwargs["path"])
            return await super().write_file(**kwargs)

    fake = Counting()
    v1 = _package(
        "cold-email",
        "v1",
        "# Cold\n",
        {"references/a.md": b"a\n", "references/b.md": b"b\n"},
    )
    with _patch_skills_path(fake):
        await _install(v1)
        fake.writes.clear()
        await _install(v1)

    assert fake.writes == ["/skills/cold-email/SKILL.md"]


def test_the_index_names_each_update_state():
    skills = [
        skills_module.ParsedSkill(
            name="a", description="A.", body="", update=UPDATE_AVAILABLE
        ),
        skills_module.ParsedSkill(
            name="b", description="B.", body="", update=UPDATE_MERGED
        ),
        skills_module.ParsedSkill(
            name="c", description="C.", body="", update=UPDATE_RETIRED
        ),
        skills_module.ParsedSkill(name="d", description="D.", body=""),
    ]
    lines = render_skills_index(skills).splitlines()
    assert "newer marketplace version" in lines[0] and "customized" in lines[0]
    assert "merged" in lines[1]
    assert "no longer offered" in lines[2]
    assert "note:" not in lines[3]


@pytest.mark.asyncio
async def test_the_next_turn_is_told_which_bodies_changed():
    fake = _FakeWorkspaceManager()
    v1 = _package("cold-email", "v1", "# Cold\n")
    with _patch_skills_path(fake), patch.object(
        skills_module, "_consume_updated", AsyncMock(return_value=["cold-email"])
    ), patch.object(
        skills_module, "is_skills_feature_enabled", AsyncMock(return_value=True)
    ):
        await _install(v1)
        prior = [
            "<available_skills>\n- name: cold-email — x\n</available_skills>\n\nhi"
        ]
        notice = await build_skills_update_notice(USER, prior_contents=prior)

    assert "Updated skills: cold-email" in notice
    assert "read_skill" in notice


@pytest.mark.asyncio
async def test_an_owner_skill_is_never_touched_by_the_reconcile():
    fake = _FakeWorkspaceManager()
    v1 = _package("cold-email", "v1", "# Cold\n")
    with _patch_skills_path(fake) as patched:
        await store_user_skill(
            USER, name="cold-email", description="mine", body="# Mine\n"
        )
        _marketplace(patched, v1)
        [skill] = await list_user_skills(USER)

    assert skill.origin == SKILL_ORIGIN_USER
    assert skill.update is None
    assert fake.files["/skills/cold-email/SKILL.md"].decode().endswith("# Mine\n")
    patched.skill_db.get_active_versions.assert_not_called()
