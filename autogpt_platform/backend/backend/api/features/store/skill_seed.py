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
from datetime import timedelta
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
    SkillPackageError,
    _validate_name,
    parse_skill_markdown,
    validate_package,
    validate_skill_content,
)
from backend.data import db as database

from .categories import validate_canonical_categories
from .retired_skill_slugs import RETIRED_STARTER_SLUGS
from .skill_submission_db import snapshot_version_files

logger = logging.getLogger(__name__)

DEFAULT_CATALOG_REPO = "Significant-Gravitas/skills-catalog"
DEFAULT_CATALOG_REF = "main"
CATALOG_FILE = "catalog.yml"
SKILLS_DIR = "skills"
_CONTENT_DIR = Path(__file__).parent / "starter_skills"
# One transaction for the whole catalog, so a failure leaves the marketplace
# as it was. The default 30s covers a handful of listings; a full catalog is
# several queries per listing against a remote database.
SEED_TRANSACTION_TIMEOUT = timedelta(minutes=10)


class CatalogEntry(TypedDict):
    slug: str
    categories: list[str]
    required_providers: list[str]


# The onboarding skill each expert opens with. Every other skill a hire gets
# now comes from the catalog, so these are the only listings still authored
# here: they set up the persona, its preferences and its standing work, which
# no upstream skill can supply.
STARTER_SKILLS: list[CatalogEntry] = [
    {
        "slug": "bookkeeping-getting-started",
        "categories": ["finance", "operations"],
        "required_providers": [],
    },
    {
        "slug": "investor-relations-getting-started",
        "categories": ["finance", "operations"],
        "required_providers": [],
    },
    {
        "slug": "kpi-analysis-getting-started",
        "categories": ["research"],
        "required_providers": [],
    },
    {
        "slug": "recruiting-getting-started",
        "categories": ["operations"],
        "required_providers": [],
    },
    {
        "slug": "procurement-getting-started",
        "categories": ["operations", "finance"],
        "required_providers": [],
    },
    {
        "slug": "contract-ops-getting-started",
        "categories": ["operations"],
        "required_providers": [],
    },
    {
        "slug": "dependency-security-getting-started",
        "categories": ["development"],
        "required_providers": [],
    },
    {
        "slug": "customer-success-getting-started",
        "categories": ["support"],
        "required_providers": [],
    },
    {
        "slug": "deal-desk-getting-started",
        "categories": ["sales"],
        "required_providers": [],
    },
    {
        "slug": "robin-getting-started",
        "categories": ["support", "operations"],
        "required_providers": [
            "google",
            "mcp_gong",
            "mcp_granola",
            "mcp_linear",
            "notion",
            "slack",
        ],
    },
    {
        "slug": "max-getting-started",
        "categories": ["sales"],
        "required_providers": [
            "google",
            "hubspot",
            "mcp_gong",
            "mcp_granola",
            "notion",
            "slack",
        ],
    },
    {
        "slug": "anika-getting-started",
        "categories": ["sales", "operations"],
        "required_providers": ["google", "hubspot", "mcp_granola", "notion", "slack"],
    },
    {
        "slug": "daniel-getting-started",
        "categories": ["finance", "operations"],
        "required_providers": ["google", "hubspot", "notion", "slack", "stripe"],
    },
    {
        "slug": "alex-getting-started",
        "categories": ["development", "research"],
        "required_providers": ["github", "google", "mcp_linear", "notion", "slack"],
    },
    {
        "slug": "sofia-getting-started",
        "categories": ["operations"],
        "required_providers": [
            "google",
            "mcp_granola",
            "mcp_linear",
            "notion",
            "slack",
        ],
    },
    {
        "slug": "james-getting-started",
        "categories": ["operations"],
        "required_providers": ["google", "mcp_linear", "notion", "slack"],
    },
    {
        "slug": "maya-getting-started",
        "categories": ["marketing", "content"],
        "required_providers": ["google", "notion", "slack", "hubspot"],
    },
    {
        "slug": "zara-getting-started",
        "categories": ["marketing", "sales"],
        "required_providers": [
            "apollo",
            "google",
            "hubspot",
            "mcp_amplitude",
            "mcp_gong",
            "mcp_linear",
            "slack",
            "stripe",
        ],
    },
    {
        "slug": "support-getting-started",
        "categories": ["support", "operations"],
        "required_providers": [],
    },
    {
        "slug": "product-getting-started",
        "categories": ["research", "operations"],
        "required_providers": [],
    },
    {
        "slug": "paid-ads-getting-started",
        "categories": ["marketing", "operations"],
        "required_providers": [],
    },
    {
        "slug": "communications-getting-started",
        "categories": ["marketing", "operations"],
        "required_providers": [],
    },
    {
        "slug": "code-quality-getting-started",
        "categories": ["development", "operations"],
        "required_providers": [],
    },
    {
        "slug": "people-ops-getting-started",
        "categories": ["operations"],
        "required_providers": [],
    },
    {
        "slug": "revops-getting-started",
        "categories": ["sales", "operations"],
        "required_providers": [],
    },
    {
        "slug": "compliance-ops-getting-started",
        "categories": ["operations"],
        "required_providers": [],
    },
    {
        "slug": "executive-assistant-getting-started",
        "categories": ["support", "operations"],
        "required_providers": [],
    },
]


async def seed_catalog_skills(catalog_dir: Path | None = None) -> list[str]:
    """Upsert every catalog listing. Returns the listing ids.

    An explicit *catalog_dir* seeds only that tree. A normal run downloads the
    catalog, or reads ``SKILLS_CATALOG_PATH``, and also keeps checked-in skills
    that the expert roster needs until the catalog holds the same slug.
    """
    if catalog_dir is not None:
        return await _seed_catalog(catalog_dir, include_starters=False)

    local = os.environ.get("SKILLS_CATALOG_PATH")
    if local:
        return await _seed_catalog(Path(local), include_starters=True)
    with tempfile.TemporaryDirectory() as tmp:
        return await _seed_catalog(_download_catalog(Path(tmp)), include_starters=True)


async def _seed_catalog(root: Path, *, include_starters: bool) -> list[str]:
    entries = load_catalog(root)
    loaded = [(entry, *_load(root, entry)) for entry in entries]
    if not include_starters:
        return await _seed_loaded(loaded)

    catalog_slugs = {entry["slug"] for entry in entries}
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


async def _delist_retired_starters(tx: prisma.Prisma) -> list[str]:
    """Delist the retired slugs that are still listed. Returns those slugs."""
    listings = await prisma.models.SkillListing.prisma(tx).find_many(
        where={
            "slug": {"in": RETIRED_STARTER_SLUGS},
            "isDeleted": False,
            "owningUserId": None,
            "owningOrgId": None,
        }
    )
    for listing in listings:
        await prisma.models.SkillListing.prisma(tx).update(
            where={"id": listing.id}, data={"isDeleted": True}
        )
        if listing.activeVersionId is not None:
            await prisma.models.SkillListingVersion.prisma(tx).update(
                where={"id": listing.activeVersionId},
                data={"isAvailable": False},
            )
    return [listing.slug for listing in listings]


async def _seed_loaded(
    loaded: list[tuple[CatalogEntry, ParsedSkill, list[SkillFile]]],
) -> list[str]:
    """Write a set whose packages have all been loaded and checked."""
    listing_ids = []
    async with database.transaction(timeout=SEED_TRANSACTION_TIMEOUT) as tx:
        for entry, parsed, files in loaded:
            listing = await _upsert_listing(tx, entry, parsed, files)
            listing_ids.append(listing.id)
            logger.info(
                f"Seeded skill '{entry['slug']}' (#{listing.id})"
                + (f" with {len(files)} package files" if files else "")
            )
        delisted = await _delist_retired_starters(tx)
        if delisted:
            logger.info(f"Delisted {len(delisted)} retired skill(s): {delisted}")
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
    tx: prisma.Prisma,
    entry: CatalogEntry,
    parsed: ParsedSkill,
    files: list[SkillFile],
) -> prisma.models.SkillListing:
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
        "sourceRepo": _attribution_value(parsed, metadata, "source"),
        "sourceUrl": _attribution_value(parsed, metadata, "source_url"),
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


def _attribution_value(
    parsed: ParsedSkill, metadata: dict[object, object], key: str
) -> str | None:
    return _optional_str(metadata.get(key)) or _optional_str(parsed.extra.get(key))


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
    validate_skill_content(parsed.description, parsed.body, parsed.triggers)
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
    validate_skill_content(parsed.description, parsed.body, parsed.triggers)
    files = _package_files(directory) if is_package else []
    validate_package(SkillPackage(skill_md=text, files=files))
    return parsed, files


def _package_files(directory: Path) -> list[SkillFile]:
    """Every file beside the directory's ``SKILL.md``, by its relative path.
    The executable bit rides along so a seeded script stays runnable."""
    root = directory / "SKILL.md"
    files = []
    for path in sorted(directory.rglob("*")):
        relative_path = path.relative_to(directory).as_posix()
        if path.is_symlink():
            raise SkillPackageError(f"file '{relative_path}' may not be a symlink")
        if path.is_file() and path != root:
            files.append(
                SkillFile(
                    relative_path=relative_path,
                    content=path.read_bytes(),
                    is_executable=os.access(path, os.X_OK),
                )
            )
    return files


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
