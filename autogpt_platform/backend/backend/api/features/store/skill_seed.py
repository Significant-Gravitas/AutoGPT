"""Catalogue package validation; legacy unversioned seed commands are disabled.

Publish through backend.api.features.store.catalog_release instead. This module
never writes marketplace records or falls back to bundled application content.
"""

import asyncio
import io
import logging
import os
import tarfile
from pathlib import Path
from typing import TypedDict

import httpx
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

from .categories import validate_canonical_categories

logger = logging.getLogger(__name__)
DEFAULT_CATALOG_REPO = "Significant-Gravitas/skills-catalog"
DEFAULT_CATALOG_REF = "main"
CATALOG_FILE = "catalog.yml"
SKILLS_DIR = "skills"


class CatalogEntry(TypedDict):
    slug: str
    categories: list[str]
    required_providers: list[str]


async def seed_catalog_skills(catalog_dir: Path | None = None) -> list[str]:
    raise RuntimeError(
        "Unversioned skill seeding is disabled. Use the catalogue release preview/apply command."
    )


async def seed_starter_skills() -> list[str]:
    raise RuntimeError(
        "Bundled starter skills were removed. Use the catalogue release preview/apply command."
    )


async def main() -> None:
    await seed_catalog_skills()


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


if __name__ == "__main__":
    asyncio.run(main())
