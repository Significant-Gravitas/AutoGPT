"""Bind an exact, clean Git checkout to validated immutable package bytes."""

import hashlib
import re
import subprocess
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from backend.api.features.store.catalog_release_model import (
    ReleaseManifest,
    StrictModel,
    digest,
)
from backend.api.features.store.skill_seed import (
    CatalogEntry,
    _attribution_value,
    _load,
    _optional_str,
    load_catalog,
)
from backend.copilot.service import strip_server_injected_tags
from backend.copilot.tools.skills import (
    SkillFile,
    SkillPackage,
    parse_skill_markdown,
    validate_package,
    validate_skill_content,
)


class LoadedPackage(StrictModel):
    slug: str
    package_sha256: str
    skill_markdown: str
    name: str
    description: str
    body: str
    triggers: list[str]
    categories: list[str]
    required_providers: list[str]
    source_repo: str | None
    source_url: str | None
    license: str | None
    catalogue_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    files: list[SkillFile] = Field(default_factory=list)


class LoadedRelease(StrictModel):
    revision: str
    release_id: str
    manifest_sha256: str
    manifest: ReleaseManifest
    packages: dict[str, LoadedPackage]


class CatalogueMetadataEntry(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)
    slug: str


class CatalogueMetadataDocument(StrictModel):
    skills: list[CatalogueMetadataEntry]


def load_release(root: Path, revision: str) -> LoadedRelease:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be an exact 40-character Git commit SHA")
    if _git(root, "rev-parse", "HEAD").strip() != revision:
        raise ValueError("checkout does not match the requested revision")
    if _git(root, "status", "--porcelain", "--untracked-files=all").strip():
        raise ValueError("catalogue checkout must be clean, including untracked files")
    tracked = _tracked_files(root, revision)
    raw = (root / "release.json").read_bytes()
    _verify_blob(tracked, "release.json", raw)
    manifest = ReleaseManifest.model_validate_json(raw)
    catalog = (root / "catalog.yml").read_bytes()
    _verify_blob(tracked, "catalog.yml", catalog)
    if hashlib.sha256(catalog).hexdigest() != manifest.catalog_sha256:
        raise ValueError("catalog.yml hash mismatch")
    entries = {entry["slug"]: entry for entry in load_catalog(root)}
    metadata = {
        entry.slug: TypeAdapter(dict[str, JsonValue]).validate_python(
            entry.model_dump()
        )
        for entry in CatalogueMetadataDocument.model_validate(
            yaml.safe_load(catalog)
        ).skills
    }
    if (root / "catalog.yml").read_bytes() != catalog:
        raise ValueError("catalogue changed while loading")
    if set(entries) != {package.slug for package in manifest.packages}:
        raise ValueError("release packages must exactly match catalog.yml")
    packages = {
        package.slug: _load_package(
            root,
            package.slug,
            entries[package.slug],
            manifest,
            tracked,
            metadata[package.slug],
        )
        for package in manifest.packages
    }
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    return LoadedRelease(
        revision=revision,
        release_id=digest({"revision": revision, "manifest_sha256": manifest_sha256}),
        manifest_sha256=manifest_sha256,
        manifest=manifest,
        packages=packages,
    )


def _load_package(
    root: Path,
    slug: str,
    entry: CatalogEntry,
    manifest: ReleaseManifest,
    tracked: dict[str, tuple[str, str]],
    catalogue_metadata: dict[str, JsonValue],
) -> LoadedPackage:
    directory = root / "skills" / slug
    if directory.is_symlink() or directory.parent.is_symlink():
        raise ValueError("package directories may not be symlinks")
    _, files = _load(root, entry)
    declared = next(package for package in manifest.packages if package.slug == slug)
    actual = {"SKILL.md": (directory / "SKILL.md").read_bytes()}
    actual.update({file.relative_path: file.content for file in files})
    if set(actual) != {file.path for file in declared.files}:
        raise ValueError(f"package {slug} file inventory differs from its manifest")
    for file in declared.files:
        path = f"skills/{slug}/{file.path}"
        if hashlib.sha256(actual[file.path]).hexdigest() != file.sha256:
            raise ValueError(f"package {slug} file hash mismatch: {file.path}")
        _verify_blob(tracked, path, actual[file.path])
        if tracked[path][0] != ("100755" if file.executable else "100644"):
            raise ValueError(f"package {slug} Git file mode mismatch: {file.path}")
    markdown = actual["SKILL.md"].decode("utf-8")
    if strip_server_injected_tags(markdown) != markdown:
        raise ValueError(f"package {slug} contains reserved server context tags")
    parsed = parse_skill_markdown(markdown.replace("\r\n", "\n").replace("\r", "\n"))
    if parsed is None or parsed.name != slug:
        raise ValueError(f"package {slug} has invalid frontmatter")
    validate_skill_content(parsed.description, parsed.body, parsed.triggers)
    files = [
        SkillFile(
            relative_path=file.path,
            content=actual[file.path],
            is_executable=file.executable,
        )
        for file in declared.files
        if file.path != "SKILL.md"
    ]
    validate_package(SkillPackage(skill_md=markdown, files=files))
    metadata = parsed.extra.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    return LoadedPackage(
        slug=slug,
        package_sha256=digest(
            {"tree_sha256": declared.tree_sha256, "entry": catalogue_metadata}
        ),
        skill_markdown=markdown,
        name=parsed.name,
        description=parsed.description,
        body=parsed.body,
        triggers=list(parsed.triggers),
        categories=entry["categories"],
        required_providers=entry["required_providers"],
        source_repo=_attribution_value(parsed, metadata, "source"),
        source_url=_attribution_value(parsed, metadata, "source_url"),
        license=_optional_str(parsed.extra.get("license")),
        catalogue_metadata=catalogue_metadata,
        files=files,
    )


def _tracked_files(root: Path, revision: str) -> dict[str, tuple[str, str]]:
    return {
        record.split("\t", 1)[1]: (
            record.split(" ", 2)[0],
            record.split("\t", 1)[0].split(" ")[2],
        )
        for record in _git(root, "ls-tree", "-r", "-z", revision).split("\0")
        if record
    }


def _verify_blob(
    tracked: dict[str, tuple[str, str]], path: str, content: bytes
) -> None:
    blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    if path not in tracked or tracked[path][1] != blob:
        raise ValueError(f"file differs from the pinned Git commit: {path}")


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
