"""Read one commit of the skills catalog into validated packages and experts.

``Significant-Gravitas/skills-catalog`` holds ``catalog.yml`` (each package's
categories and required integrations), ``skills/<slug>/`` (the package),
``experts/<key>.yml`` (each roster template) and ``release.json``, which binds
every file to a SHA-256. :func:`load_release` verifies all of it: every byte
the manifest names is hashed and compared, every package passes the checks
an upload gets, every expert file parses. Nothing here touches the database;
:mod:`backend.api.features.store.skill_catalog` publishes what this returns.

Environment:

``SKILLS_CATALOG_PATH``
    A local checkout to use instead of GitHub.
``SKILLS_CATALOG_REPO`` / ``SKILLS_CATALOG_REF``
    The repo (``owner/name``) and branch, tag or commit to download.
``SKILLS_CATALOG_TOKEN``
    A GitHub token, optional for the public catalog; ``GITHUB_TOKEN`` is the
    fallback. Raises the API rate limit and is required for a private fork.
"""

import io
import json
import logging
import os
import re
import tarfile
from pathlib import Path

import httpx
import yaml
from pydantic import BaseModel, ConfigDict
from typing_extensions import TypedDict

from backend.api.features.experts.roster import (
    EXPERT_FILE_SUFFIX,
    EXPERTS_DIR,
    RosterError,
    expert_file,
    parse_expert,
)
from backend.api.features.experts.roster_types import RosterEntry
from backend.copilot.service import strip_server_injected_tags
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
from backend.data.skill_package import SKILL_MD, file_sha256, package_tree_sha256

from .categories import validate_canonical_categories

logger = logging.getLogger(__name__)

DEFAULT_CATALOG_REPO = "Significant-Gravitas/skills-catalog"
DEFAULT_CATALOG_REF = "main"
CATALOG_FILE = "catalog.yml"
RELEASE_FILE = "release.json"
SKILLS_DIR = "skills"
RELEASE_SCHEMA_VERSION = 2

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX256_RE = re.compile(r"^[0-9a-f]{64}$")


class CatalogError(ValueError):
    """The checkout is inconsistent with its manifest, or a package would
    fail the checks an install applies."""


class CatalogEntry(TypedDict):
    slug: str
    categories: list[str]
    required_providers: list[str]
    # `platform`, or `owner/repo/path` for a vendored skill.
    source: str | None
    license: str | None


class LoadedPackage(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    slug: str
    entry: CatalogEntry
    parsed: ParsedSkill
    # The SKILL.md exactly as the catalog holds it; installed verbatim.
    skill_markdown: str
    # Everything beside SKILL.md.
    files: list[SkillFile]
    # The catalog's tree_sha256, recomputed here from the bytes.
    package_sha256: str


class LoadedRelease(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    release_key: str
    manifest_sha256: str
    packages: list[LoadedPackage]
    retirements: list[str]
    experts: list[RosterEntry]
    retired_experts: list[str]

    @property
    def slugs(self) -> set[str]:
        return {package.slug for package in self.packages}


class CatalogSource(BaseModel):
    model_config = ConfigDict(frozen=True)

    repository: str
    revision: str
    root: Path


# ---------------------------------------------------------------------------
# Loading a checkout
# ---------------------------------------------------------------------------


def load_release(root: Path) -> LoadedRelease:
    manifest_bytes = _read_bytes(root / RELEASE_FILE)
    manifest = _parse_manifest(manifest_bytes)
    release_key = _require_str(manifest, "release_key")

    catalog_bytes = _read_bytes(root / CATALOG_FILE)
    if manifest.get("catalog_sha256") != file_sha256(catalog_bytes):
        raise CatalogError(f"{CATALOG_FILE} does not match its hash in {RELEASE_FILE}")
    entries = {entry["slug"]: entry for entry in load_catalog(root)}

    raw_packages = manifest.get("packages")
    if not isinstance(raw_packages, list) or not raw_packages:
        raise CatalogError(f"{RELEASE_FILE}: packages must be a non-empty list")
    manifest_slugs = [_require_str(raw, "slug") for raw in raw_packages]
    if len(set(manifest_slugs)) != len(manifest_slugs):
        raise CatalogError(f"{RELEASE_FILE}: a package is listed twice")
    if set(manifest_slugs) != set(entries):
        missing = sorted(set(entries) - set(manifest_slugs))
        extra = sorted(set(manifest_slugs) - set(entries))
        raise CatalogError(
            f"{RELEASE_FILE} and {CATALOG_FILE} disagree: "
            f"not in manifest {missing}, not in catalog {extra}"
        )
    packages = [_load_package(root, entries[raw["slug"]], raw) for raw in raw_packages]

    retirements = _require_slug_list(manifest, "retirements")
    if overlap := sorted(set(retirements) & set(entries)):
        raise CatalogError(f"retired but still in the catalog: {overlap}")
    experts = _load_experts(root, manifest, set(entries))
    retired_experts = _require_slug_list(manifest, "retired_experts")
    if overlap := sorted(set(retired_experts) & {e["key"] for e in experts}):
        raise CatalogError(f"retired but still in the roster: {overlap}")

    return LoadedRelease(
        release_key=release_key,
        manifest_sha256=file_sha256(manifest_bytes),
        packages=packages,
        retirements=retirements,
        experts=experts,
        retired_experts=retired_experts,
    )


def _parse_manifest(manifest_bytes: bytes) -> dict:
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as exc:
        raise CatalogError(f"{RELEASE_FILE}: not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise CatalogError(f"{RELEASE_FILE} must contain an object")
    if manifest.get("schema_version") != RELEASE_SCHEMA_VERSION:
        raise CatalogError(
            f"{RELEASE_FILE}: schema_version {manifest.get('schema_version')!r} "
            f"is not {RELEASE_SCHEMA_VERSION}, the only schema this backend publishes"
        )
    if manifest.get("system_packages"):
        raise CatalogError("system_packages are not supported")
    return manifest


def load_catalog(root: Path) -> list[CatalogEntry]:
    """The catalog's entries, each with a canonical category set."""
    raw = yaml.safe_load(_read_bytes(root / CATALOG_FILE).decode("utf-8")) or {}
    if not isinstance(raw, dict):
        raise CatalogError(f"{CATALOG_FILE} must contain a mapping")
    skills = raw.get("skills") or []
    if not isinstance(skills, list):
        raise CatalogError(f"{CATALOG_FILE}: skills must be a list")
    entries: list[CatalogEntry] = []
    seen: set[str] = set()
    for item in skills:
        entry = _catalog_entry(item)
        if entry["slug"] in seen:
            raise CatalogError(f"{CATALOG_FILE}: '{entry['slug']}' is listed twice")
        seen.add(entry["slug"])
        entries.append(entry)
    if not entries:
        raise CatalogError(f"{CATALOG_FILE} lists no skills")
    return entries


def _catalog_entry(item: object) -> CatalogEntry:
    if not isinstance(item, dict):
        raise CatalogError(f"{CATALOG_FILE}: every skill must be a mapping")
    slug = str(item.get("slug") or "").strip()
    if not slug:
        raise CatalogError(f"{CATALOG_FILE}: an entry has no slug")
    if error := _validate_name(slug):
        raise CatalogError(f"{CATALOG_FILE}: '{slug}' {error}")
    categories = item.get("categories") or []
    providers = item.get("required_providers") or []
    for label, value in (("categories", categories), ("required_providers", providers)):
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise CatalogError(
                f"{CATALOG_FILE}: '{slug}' {label} must be a list of strings"
            )
    try:
        canonical = validate_canonical_categories(categories)
    except ValueError as exc:
        raise CatalogError(f"{CATALOG_FILE}: '{slug}': {exc}") from exc
    return CatalogEntry(
        slug=slug,
        categories=canonical,
        required_providers=providers,
        source=_optional_str(item.get("source")),
        license=_optional_str(item.get("license")),
    )


def _load_package(root: Path, entry: CatalogEntry, raw: dict) -> LoadedPackage:
    slug = entry["slug"]
    directory = root / SKILLS_DIR / slug
    if directory.is_symlink() or not directory.is_dir():
        raise CatalogError(f"{SKILLS_DIR}/{slug}/ is missing")
    expected = _manifest_files(slug, raw)
    on_disk = _walk_package(directory, slug)
    if set(on_disk) != set(expected):
        missing = sorted(set(expected) - set(on_disk))
        extra = sorted(set(on_disk) - set(expected))
        raise CatalogError(
            f"{slug}: checkout and manifest disagree: missing {missing}, unlisted {extra}"
        )
    hashed: list[tuple[str, str, bool]] = []
    for path, content in on_disk.items():
        digest, executable = expected[path]
        if file_sha256(content) != digest:
            raise CatalogError(f"{slug}: '{path}' does not match its hash")
        hashed.append((path, digest, executable))
    package_sha256 = package_tree_sha256(hashed)
    if raw.get("tree_sha256") != package_sha256:
        raise CatalogError(f"{slug}: tree_sha256 does not match the files")
    text, parsed = _parse_skill_md(slug, on_disk[SKILL_MD])
    files = [
        SkillFile(relative_path=path, content=content, is_executable=expected[path][1])
        for path, content in sorted(on_disk.items())
        if path != SKILL_MD
    ]
    try:
        validate_skill_content(parsed.description, parsed.body, list(parsed.triggers))
        validate_package(SkillPackage(skill_md=text, files=files))
    except (ValueError, SkillPackageError) as exc:
        raise CatalogError(f"{slug}: {exc}") from exc
    return LoadedPackage(
        slug=slug,
        entry=entry,
        parsed=parsed,
        skill_markdown=text,
        files=files,
        package_sha256=package_sha256,
    )


def _manifest_files(slug: str, raw: dict) -> dict[str, tuple[str, bool]]:
    raw_files = raw.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise CatalogError(f"{slug}: files must be a non-empty list")
    expected: dict[str, tuple[str, bool]] = {}
    for item in raw_files:
        path = _require_str(item, "path")
        if path in expected:
            raise CatalogError(f"{slug}: file '{path}' is listed twice")
        digest = _require_str(item, "sha256")
        if not _HEX256_RE.fullmatch(digest):
            raise CatalogError(f"{slug}: '{path}' has a malformed sha256")
        if not isinstance(item.get("executable"), bool):
            raise CatalogError(f"{slug}: '{path}' executable must be a boolean")
        expected[path] = (digest, item["executable"])
    if SKILL_MD not in expected:
        raise CatalogError(f"{slug}: manifest lists no {SKILL_MD}")
    return expected


def _parse_skill_md(slug: str, content: bytes) -> tuple[str, ParsedSkill]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CatalogError(f"{slug}: {SKILL_MD} is not UTF-8") from exc
    if strip_server_injected_tags(text) != text:
        raise CatalogError(f"{slug}: {SKILL_MD} carries reserved server tags")
    parsed = parse_skill_markdown(text)
    if parsed is None:
        raise CatalogError(f"{slug}: {SKILL_MD} is not a valid SKILL.md")
    if parsed.name != slug:
        raise CatalogError(
            f"{slug}: {SKILL_MD} declares name '{parsed.name}'; the frontmatter "
            "name is the installed skill's name and must match the slug"
        )
    return text, parsed


def _walk_package(directory: Path, slug: str) -> dict[str, bytes]:
    """Every regular file under the package folder by relative posix path."""
    files: dict[str, bytes] = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory).as_posix()
        if path.is_symlink():
            raise CatalogError(f"{slug}: '{relative}' is a symlink")
        if path.is_file():
            files[relative] = path.read_bytes()
    return files


def _load_experts(
    root: Path, manifest: dict, active_slugs: set[str]
) -> list[RosterEntry]:
    raw_experts = manifest.get("experts")
    if not isinstance(raw_experts, list):
        raise CatalogError(f"{RELEASE_FILE}: experts must be a list")
    experts: list[RosterEntry] = []
    keys: set[str] = set()
    for raw in raw_experts:
        key = _require_str(raw, "key")
        if key in keys:
            raise CatalogError(f"{RELEASE_FILE}: expert '{key}' is listed twice")
        keys.add(key)
        experts.append(_load_expert(root, key, raw, active_slugs))
    directory = root / EXPERTS_DIR
    if directory.is_dir():
        unlisted = sorted(
            path.name[: -len(EXPERT_FILE_SUFFIX)]
            for path in directory.glob(f"*{EXPERT_FILE_SUFFIX}")
            if path.name[: -len(EXPERT_FILE_SUFFIX)] not in keys
        )
        if unlisted:
            raise CatalogError(f"experts not in {RELEASE_FILE}: {unlisted}")
    names = [entry["name"].lower() for entry in experts]
    if len(set(names)) != len(names):
        raise CatalogError("two experts share a display name")
    return experts


def _load_expert(
    root: Path, key: str, raw: dict, active_slugs: set[str]
) -> RosterEntry:
    named = f"{EXPERTS_DIR}/{key}{EXPERT_FILE_SUFFIX}"
    content = _read_bytes(expert_file(root, key))
    if raw.get("sha256") != file_sha256(content):
        raise CatalogError(f"{named} does not match its hash")
    try:
        entry = parse_expert(content.decode("utf-8"), expected_key=key)
    except (RosterError, UnicodeDecodeError) as exc:
        raise CatalogError(f"{named}: {exc}") from exc
    if raw.get("skills") != entry["bundled_skills"]:
        raise CatalogError(
            f"{RELEASE_FILE}: expert '{key}' skills differ from its file"
        )
    if unknown := [s for s in entry["bundled_skills"] if s not in active_slugs]:
        raise CatalogError(
            f"expert '{key}' bundles skills not in the catalog: {unknown}"
        )
    return entry


def _read_bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise CatalogError(f"{path.name} is missing")
    return path.read_bytes()


def _require_str(mapping: object, key: str) -> str:
    if not isinstance(mapping, dict) or not isinstance(mapping.get(key), str):
        raise CatalogError(f"{RELEASE_FILE}: '{key}' must be a string")
    value = mapping[key].strip()
    if not value:
        raise CatalogError(f"{RELEASE_FILE}: '{key}' is empty")
    return value


def _require_slug_list(manifest: dict, key: str) -> list[str]:
    value = manifest.get(key, [])
    if not isinstance(value, list) or not all(isinstance(s, str) for s in value):
        raise CatalogError(f"{RELEASE_FILE}: '{key}' must be a list of strings")
    if len(set(value)) != len(value):
        raise CatalogError(f"{RELEASE_FILE}: '{key}' has a duplicate")
    return sorted(value)


def _optional_str(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


# ---------------------------------------------------------------------------
# Fetching a commit onto disk
# ---------------------------------------------------------------------------


def resolve_catalog(into: Path) -> CatalogSource:
    """The checkout to publish: ``SKILLS_CATALOG_PATH`` if set, else the
    configured repo and ref downloaded under *into*."""
    local = os.environ.get("SKILLS_CATALOG_PATH")
    if local:
        root = Path(local)
        return CatalogSource(
            repository=f"file:{root}", revision=local_revision(root), root=root
        )
    repo = os.environ.get("SKILLS_CATALOG_REPO") or DEFAULT_CATALOG_REPO
    ref = os.environ.get("SKILLS_CATALOG_REF") or DEFAULT_CATALOG_REF
    return fetch_catalog(into, repo=repo, ref=ref)


def fetch_catalog(into: Path, *, repo: str, ref: str) -> CatalogSource:
    """Download *repo* at *ref* into *into*. The source names the exact
    commit the ref resolved to, so a release log entry is reproducible even
    when the ref was a branch."""
    headers = github_headers()
    revision = ref if _SHA_RE.fullmatch(ref) else resolve_ref(repo, ref, headers)
    logger.info(f"Downloading {repo}@{revision}")
    response = httpx.get(
        f"https://api.github.com/repos/{repo}/tarball/{revision}",
        headers=headers,
        follow_redirects=True,
        timeout=120,
    )
    response.raise_for_status()
    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
        _extract_catalog_archive(tar, into)
    # GitHub wraps the tree in one "<owner>-<repo>-<sha>" directory.
    roots = [p for p in into.iterdir() if p.is_dir()]
    if len(roots) != 1:
        raise CatalogError(f"unexpected tarball layout for {repo}: {roots}")
    return CatalogSource(repository=repo, revision=revision, root=roots[0])


def github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "autogpt-platform-skills-catalog",
    }
    token = os.environ.get("SKILLS_CATALOG_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def resolve_ref(repo: str, ref: str, headers: dict[str, str]) -> str:
    """The commit *ref* points at now. A 40-hex ref is returned as is."""
    if _SHA_RE.fullmatch(ref):
        return ref
    response = httpx.get(
        f"https://api.github.com/repos/{repo}/commits/{ref}",
        headers={**headers, "Accept": "application/vnd.github.sha"},
        follow_redirects=True,
        timeout=30,
    )
    response.raise_for_status()
    sha = response.text.strip()
    if not _SHA_RE.fullmatch(sha):
        raise CatalogError(f"could not resolve {repo}@{ref} to a commit: {sha!r}")
    return sha


def local_revision(root: Path) -> str:
    """The checkout's HEAD when it is a git checkout, else a marker."""
    head = root / ".git" / "HEAD"
    try:
        ref = head.read_text(encoding="utf-8").strip()
        if ref.startswith("ref: "):
            return (root / ".git" / ref[5:]).read_text(encoding="utf-8").strip()
        return ref
    except OSError:
        return "local"


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
            raise CatalogError(f"unsafe catalog archive member: {member.name}")
    archive.extractall(into, members=members)
