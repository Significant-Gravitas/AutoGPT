"""Seed the platform-authored skill listings from the skills catalog.

Run as ``python -m backend.api.features.store.skill_seed``, like the expert
roster seed. Idempotent: re-running rewrites each listing's live version in
place rather than stacking a new one, so the marketplace is edited by editing
the catalog and seeding again.

The catalog is the private ``Significant-Gravitas/skills-catalog`` repo. Its
``catalog.yml`` names every listing with its categories and the integrations
its instructions assume; ``skills/<slug>/`` holds the SKILL.md beside the
references it points at. Each SKILL.md is parsed with the same
:func:`parse_skill_markdown` the copilot and the upload endpoint use, so a
seeded skill cannot drift from the format an installed skill has, and the
package files go through the same :func:`validate_package` an upload does.

Environment:

``SKILLS_CATALOG_PATH``
    A local checkout to seed from instead of GitHub.
``SKILLS_CATALOG_REPO`` / ``SKILLS_CATALOG_REF``
    The repo (``owner/name``) and branch, tag or commit to download.
``SKILLS_CATALOG_TOKEN``
    A GitHub token that can read the repo; ``GITHUB_TOKEN`` is the fallback.
"""

import asyncio
import io
import logging
import os
import tarfile
import tempfile
from pathlib import Path
from typing import TypedDict

import httpx
import prisma
import prisma.enums
import prisma.models
import yaml

from backend.copilot.tools.skills import (
    ParsedSkill,
    SkillFile,
    SkillPackage,
    _validate_name,
    parse_skill_markdown,
    validate_package,
)
from backend.data import db as database

from .categories import validate_canonical_categories
from .skill_submission_db import snapshot_version_files

logger = logging.getLogger(__name__)

DEFAULT_CATALOG_REPO = "Significant-Gravitas/skills-catalog"
DEFAULT_CATALOG_REF = "main"
CATALOG_FILE = "catalog.yml"
SKILLS_DIR = "skills"
_CONTENT_DIR = Path(__file__).parent / "starter_skills"


class CatalogEntry(TypedDict):
    slug: str
    categories: list[str]
    required_providers: list[str]


STARTER_SKILLS: list[CatalogEntry] = [
    {
        "slug": "brand-voice-guide",
        "categories": ["content"],
        "required_providers": [],
    },
    {
        "slug": "outreach-playbook",
        "categories": ["sales"],
        "required_providers": ["google"],
    },
    {
        "slug": "seo-content-brief",
        "categories": ["marketing", "content"],
        "required_providers": [],
    },
    {
        "slug": "on-page-seo-audit",
        "categories": ["marketing"],
        "required_providers": [],
    },
    {
        "slug": "content-repurposing",
        "categories": ["marketing", "content"],
        "required_providers": [],
    },
    {
        "slug": "competitor-teardown",
        "categories": ["research", "marketing"],
        "required_providers": [],
    },
    {
        "slug": "icp-and-positioning",
        "categories": ["marketing", "research"],
        "required_providers": [],
    },
    {
        "slug": "lifecycle-email-map",
        "categories": ["marketing"],
        "required_providers": [],
    },
    {
        "slug": "email-deliverability-guardrails",
        "categories": ["marketing"],
        "required_providers": [],
    },
]


async def seed_catalog_skills(catalog_dir: Path | None = None) -> list[str]:
    """Upsert every catalog listing. Returns the listing ids.

    With no *catalog_dir* the catalog is downloaded from GitHub, or read from
    ``SKILLS_CATALOG_PATH`` when that is set.
    """
    if catalog_dir is None:
        local = os.environ.get("SKILLS_CATALOG_PATH")
        if local:
            return await seed_catalog_skills(Path(local))
        with tempfile.TemporaryDirectory() as tmp:
            return await seed_catalog_skills(_download_catalog(Path(tmp)))

    entries = load_catalog(catalog_dir)
    catalog_slugs = {entry["slug"] for entry in entries}
    loaded = [(entry, *_load(catalog_dir, entry)) for entry in entries]
    loaded += [
        (entry, *_load_starter(entry))
        for entry in STARTER_SKILLS
        if entry["slug"] not in catalog_slugs
    ]
    return await _seed_loaded(loaded)


async def seed_starter_skills() -> list[str]:
    """Upsert the checked-in skills needed by the expert roster."""
    loaded = [(entry, *_load_starter(entry)) for entry in STARTER_SKILLS]
    return await _seed_loaded(loaded)


async def _seed_loaded(
    loaded: list[tuple[CatalogEntry, ParsedSkill, list[SkillFile]]],
) -> list[str]:
    """Validate all packages before writing any listing."""
    listing_ids = []
    for entry, parsed, files in loaded:
        listing = await _upsert_listing(entry, parsed, files)
        listing_ids.append(listing.id)
        logger.info(
            f"Seeded skill '{entry['slug']}' (#{listing.id})"
            + (f" with {len(files)} package files" if files else "")
        )
    return listing_ids


def load_catalog(root: Path) -> list[CatalogEntry]:
    """The catalog's entries, each with a canonical category set."""
    raw = yaml.safe_load((root / CATALOG_FILE).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{CATALOG_FILE} must contain a mapping")
    skills = raw.get("skills") or []
    if not isinstance(skills, list):
        raise ValueError(f"{CATALOG_FILE}: skills must be a list")
    entries: list[CatalogEntry] = []
    seen: set[str] = set()
    for item in skills:
        if not isinstance(item, dict):
            raise ValueError(f"{CATALOG_FILE}: every skill must be a mapping")
        slug = str(item.get("slug") or "").strip()
        if not slug:
            raise ValueError(f"{CATALOG_FILE}: an entry has no slug")
        if error := _validate_name(slug):
            raise ValueError(f"{CATALOG_FILE}: '{slug}' {error}")
        if slug in seen:
            raise ValueError(f"{CATALOG_FILE}: '{slug}' is listed twice")
        seen.add(slug)
        raw_categories = item.get("categories")
        categories = [] if raw_categories is None else raw_categories
        if not isinstance(categories, list) or not all(
            isinstance(category, str) for category in categories
        ):
            raise ValueError(
                f"{CATALOG_FILE}: '{slug}' categories must be a list of strings"
            )
        raw_required_providers = item.get("required_providers")
        required_providers = (
            [] if raw_required_providers is None else raw_required_providers
        )
        if not isinstance(required_providers, list) or not all(
            isinstance(provider, str) for provider in required_providers
        ):
            raise ValueError(
                f"{CATALOG_FILE}: '{slug}' required_providers must be a list of strings"
            )
        entries.append(
            CatalogEntry(
                slug=slug,
                categories=validate_canonical_categories(categories),
                required_providers=required_providers,
            )
        )
    if not entries:
        raise ValueError(f"{CATALOG_FILE} lists no skills")
    return entries


async def _upsert_listing(
    entry: CatalogEntry, parsed: ParsedSkill, files: list[SkillFile]
) -> prisma.models.SkillListing:
    async with database.transaction() as tx:
        listing = await prisma.models.SkillListing.prisma(tx).find_unique(
            where={"slug": entry["slug"]}, include={"ActiveVersion": True}
        )
        if listing is None:
            listing = await prisma.models.SkillListing.prisma(tx).create(
                data={"slug": entry["slug"], "hasApprovedVersion": True},
                include={"ActiveVersion": True},
            )
        else:
            if listing.owningUserId is not None or listing.owningOrgId is not None:
                raise ValueError(
                    f"catalog skill '{entry['slug']}' conflicts with an owned listing"
                )
            listing = (
                await prisma.models.SkillListing.prisma(tx).update(
                    where={"id": listing.id},
                    data={"hasApprovedVersion": True, "isDeleted": False},
                    include={"ActiveVersion": True},
                )
                or listing
            )
        version = await _upsert_version(tx, listing, entry, parsed, files)
        if listing.activeVersionId != version.id:
            listing = (
                await prisma.models.SkillListing.prisma(tx).update(
                    where={"id": listing.id},
                    data={"activeVersionId": version.id},
                    include={"ActiveVersion": True},
                )
                or listing
            )
        return listing


async def _upsert_version(
    tx: prisma.Prisma,
    listing: prisma.models.SkillListing,
    entry: CatalogEntry,
    parsed: ParsedSkill,
    files: list[SkillFile],
) -> prisma.models.SkillListingVersion:
    """Rewrite the listing's live version in place, package and all.

    A catalog skill is platform-authored, so there is no review to preserve
    and no creator waiting on a version history — editing the catalog should
    change what installers get, not add a row.
    """
    metadata = parsed.extra.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    license_value = parsed.extra.get("license")
    content: dict = {
        "name": parsed.name,
        "description": parsed.description,
        "body": parsed.body,
        "triggers": list(parsed.triggers),
        "categories": entry["categories"],
        "requiredProviders": entry["required_providers"],
        "sourceSkillSlug": entry["slug"],
        "sourceRepo": _optional_str(metadata.get("source")),
        "sourceUrl": _optional_str(metadata.get("source_url")),
        "license": _optional_str(license_value),
        "isAvailable": True,
        "isDeleted": False,
        "submissionStatus": prisma.enums.SubmissionStatus.APPROVED,
    }
    existing = listing.ActiveVersion
    if existing is not None:
        updated = await prisma.models.SkillListingVersion.prisma(tx).update(
            where={"id": existing.id}, data=content
        )
        if updated is not None:
            await snapshot_version_files(updated.id, files, tx)
            return updated
    created = await prisma.models.SkillListingVersion.prisma(tx).create(
        data={**content, "skillListingId": listing.id}
    )
    await snapshot_version_files(created.id, files, tx)
    return created


def _optional_str(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _load(root: Path, entry: CatalogEntry) -> tuple[ParsedSkill, list[SkillFile]]:
    """A catalog skill's SKILL.md and the files beside it, validated as the
    package an installer will receive."""
    slug = entry["slug"]
    directory = root / SKILLS_DIR / slug
    skill_md = directory / "SKILL.md"
    named = f"{SKILLS_DIR}/{slug}/SKILL.md"
    if not skill_md.is_file():
        raise ValueError(f"{named} is missing")
    text = skill_md.read_text(encoding="utf-8")
    parsed = parse_skill_markdown(text)
    if parsed is None:
        raise ValueError(f"{named} is not a valid SKILL.md")
    if parsed.name != slug:
        raise ValueError(
            f"{named} declares name '{parsed.name}'; the frontmatter name is "
            "the installed skill's name and must match the catalog slug"
        )
    files = _package_files(directory)
    validate_package(SkillPackage(skill_md=text, files=files))
    return parsed, files


def _load_starter(entry: CatalogEntry) -> tuple[ParsedSkill, list[SkillFile]]:
    slug = entry["slug"]
    directory = _CONTENT_DIR / slug
    is_package = directory.is_dir()
    root = directory / "SKILL.md" if is_package else _CONTENT_DIR / f"{slug}.md"
    named = f"starter_skills/{root.relative_to(_CONTENT_DIR)}"
    text = root.read_text(encoding="utf-8")
    parsed = parse_skill_markdown(text)
    if parsed is None:
        raise ValueError(f"{named} is not a valid SKILL.md")
    if parsed.name != slug:
        raise ValueError(
            f"{named} declares name '{parsed.name}'; the frontmatter name is "
            "the installed skill's name and must match the listing slug"
        )
    files = _package_files(directory) if is_package else []
    validate_package(SkillPackage(skill_md=text, files=files))
    return parsed, files


def _package_files(directory: Path) -> list[SkillFile]:
    """Every file beside the directory's ``SKILL.md``, by its relative path.
    The executable bit rides along so a seeded script stays runnable."""
    root = directory / "SKILL.md"
    return [
        SkillFile(
            relative_path=path.relative_to(directory).as_posix(),
            content=path.read_bytes(),
            is_executable=os.access(path, os.X_OK),
        )
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path != root
    ]


def _download_catalog(into: Path) -> Path:
    """Fetch the catalog repo's tarball from GitHub and unpack it under *into*,
    returning the checkout root."""
    repo = os.environ.get("SKILLS_CATALOG_REPO") or DEFAULT_CATALOG_REPO
    ref = os.environ.get("SKILLS_CATALOG_REF") or DEFAULT_CATALOG_REF
    token = os.environ.get("SKILLS_CATALOG_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError(
            "SKILLS_CATALOG_TOKEN (or GITHUB_TOKEN) is required to download "
            f"{repo}; set SKILLS_CATALOG_PATH to seed from a local checkout"
        )
    logger.info(f"Downloading {repo}@{ref}")
    response = httpx.get(
        f"https://api.github.com/repos/{repo}/tarball/{ref}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "autogpt-platform-skill-seed",
        },
        follow_redirects=True,
        timeout=60,
    )
    response.raise_for_status()
    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
        _extract_catalog_archive(tar, into)
    # GitHub wraps the tree in one "<owner>-<repo>-<sha>" directory.
    roots = [p for p in into.iterdir() if p.is_dir()]
    if len(roots) != 1:
        raise RuntimeError(f"unexpected tarball layout for {repo}: {roots}")
    return roots[0]


def _extract_catalog_archive(archive: tarfile.TarFile, into: Path) -> None:
    """Extract with the data filter, including on Python versions before its backport."""
    if hasattr(tarfile, "data_filter"):
        archive.extractall(into, filter="data")
        return

    root = into.resolve()
    members = archive.getmembers()
    for member in members:
        target = (into / member.name).resolve()
        if not (member.isfile() or member.isdir()) or (
            target != root and root not in target.parents
        ):
            raise RuntimeError(f"unsafe catalog archive member: {member.name}")
    archive.extractall(into, members=members)


async def main() -> None:
    await database.connect()
    try:
        await seed_catalog_skills()
    finally:
        await database.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
