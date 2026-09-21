"""The ``.expert.zip`` transport for a whole expert.

An expert package is three things and nothing else: ``expert.json``,
``skills/<slug>/**`` and at most one ``avatar.<ext>``. Any other member is
refused by name, which is how the ticket's rule — memory, conversations and
workspace files never leave and never arrive — is enforced on the file rather
than trusted to the writer.

Both entry points are CPU-bound and synchronous; callers run them off the event
loop.
"""

import io
import zipfile

from pydantic import ValidationError

from backend.api.features.experts.package_model import (
    AVATAR_PATHS,
    MAX_AVATAR_BYTES,
    MAX_MANIFEST_BYTES,
    MAX_PACKAGE_SKILLS,
    ExpertManifest,
    ExpertPackage,
    ExpertPackageError,
    check_skill_slug,
    manifest_json,
    validate_expert_package,
)
from backend.api.features.zip_members import add_member, checked_members, mode
from backend.copilot.tools.skills import (
    MAX_PACKAGE_BYTES,
    MAX_PACKAGE_FILE_BYTES,
    SkillFile,
    SkillPackage,
    SkillPackageError,
    validate_package,
)

MANIFEST_NAME = "expert.json"
SKILLS_PREFIX = "skills/"
ROOT_SKILL_MD = "SKILL.md"
# A compressed archive already past the uncompressed cap cannot hold a package
# that fits, so one number bounds both the request body and the tree.
MAX_ZIP_BYTES = MAX_PACKAGE_BYTES
# The manifest may be larger than any one skill file, so the archive-wide
# member cap is the largest of the three kinds; each member is then held to
# its own kind's cap once sorted, still from the central directory.
MAX_MEMBER_BYTES = max(MAX_MANIFEST_BYTES, MAX_PACKAGE_FILE_BYTES, MAX_AVATAR_BYTES)

_SHAPE = "an expert package is expert.json, skills/ and an optional avatar"
_NAMED_PATHS = 10


def package_from_zip(data: bytes) -> ExpertPackage:
    """Unpack an uploaded archive into a validated :class:`ExpertPackage`.

    Every refusal is an :class:`ExpertPackageError`, ``over_limit`` set where
    the REST edge owes a 413. Sizes come from the central directory, so a zip
    bomb is refused without a member being decompressed.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, OSError) as exc:
        raise ExpertPackageError(f"upload is not a readable zip archive: {exc}")
    with archive:
        members = _unwrapped(
            checked_members(
                archive.infolist(),
                max_file_bytes=MAX_MEMBER_BYTES,
                max_total_bytes=MAX_PACKAGE_BYTES,
                error=ExpertPackageError,
            )
        )
        manifest_info, skill_members, avatar = _partitioned(members)
        _check_declared_sizes(manifest_info, skill_members, avatar)
        # A member's CRC is verified as it decompresses, so a corrupt archive
        # opens cleanly and fails here.
        try:
            manifest = _manifest(archive.read(manifest_info))
            for slug in sorted(skill_members):
                try:
                    check_skill_slug(slug)
                except ValueError as exc:
                    raise ExpertPackageError(str(exc))
            skills = {
                slug: _skill(archive, slug, files)
                for slug, files in sorted(skill_members.items())
            }
            avatar_bytes = archive.read(avatar[1]) if avatar else None
        except (zipfile.BadZipFile, OSError) as exc:
            raise ExpertPackageError(f"archive member could not be read: {exc}")
    _check_skills_agree(manifest, skills)
    _check_avatar_agrees(manifest, avatar[0] if avatar else None)
    return ExpertPackage(
        manifest=manifest,
        skills=skills,
        avatar_bytes=avatar_bytes,
        avatar_mime=AVATAR_PATHS[avatar[0]] if avatar else None,
    )


def zip_from_package(package: ExpertPackage) -> bytes:
    """Write an expert out: the manifest, every skill under its slug, and the
    avatar when it travels as bytes.

    Members go in sorted with a fixed stamp, so exporting an unchanged expert
    twice produces the same file and a download can be checksummed.

    A package :func:`package_from_zip` would refuse is refused here instead of
    written — including one whose content is incompressible enough that the
    archive itself would be over the cap an upload is held to.
    """
    validate_expert_package(package)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        add_member(archive, MANIFEST_NAME, manifest_json(package.manifest), False)
        for slug, skill in sorted(package.skills.items()):
            root = f"{SKILLS_PREFIX}{slug}/"
            add_member(archive, root + ROOT_SKILL_MD, skill.skill_md.encode(), False)
            for entry in sorted(skill.files, key=lambda f: f.relative_path):
                path = root + entry.relative_path
                add_member(archive, path, entry.content, entry.is_executable)
        avatar = package.manifest.avatar
        if package.avatar_bytes is not None and avatar and avatar.path:
            add_member(archive, avatar.path, package.avatar_bytes, False)
    data = buffer.getvalue()
    if len(data) > MAX_ZIP_BYTES:
        raise ExpertPackageError(
            f"archive would be {len(data)} bytes; the limit is {MAX_ZIP_BYTES}",
            over_limit=True,
        )
    return data


_Members = dict[str, zipfile.ZipInfo]
_Avatar = tuple[str, zipfile.ZipInfo] | None


def _unwrapped(members: _Members) -> _Members:
    """Strip the wrapping directory a "download zip" button adds, so
    ``maria-export/expert.json`` is the package root.

    Only when one directory holds every member and the manifest is the one
    inside it, so a package whose own root is present is never rewritten.
    """
    tops = {name.split("/", 1)[0] for name in members}
    if len(tops) == 1 and (top := tops.pop()) and f"{top}/{MANIFEST_NAME}" in members:
        members = {name[len(top) + 1 :]: info for name, info in members.items()}
    if MANIFEST_NAME not in members:
        raise ExpertPackageError(
            f"archive has no {MANIFEST_NAME} at its root — {_SHAPE}"
        )
    return members


def _partitioned(
    members: _Members,
) -> tuple[zipfile.ZipInfo, dict[str, _Members], _Avatar]:
    """Split the archive into the three things a package may hold, refusing
    everything else by name so the user sees what was in the file."""
    skills: dict[str, _Members] = {}
    avatars: list[tuple[str, zipfile.ZipInfo]] = []
    unknown: list[str] = []
    for path, info in sorted(members.items()):
        slug, _, relative = path.removeprefix(SKILLS_PREFIX).partition("/")
        if path == MANIFEST_NAME:
            continue
        if path.startswith(SKILLS_PREFIX) and relative:
            skills.setdefault(slug, {})[relative] = info
        elif path in AVATAR_PATHS:
            avatars.append((path, info))
        elif "/" not in path and path.startswith("avatar."):
            raise ExpertPackageError(
                f"'{path}' is not a supported avatar format; use one of "
                f"{sorted(AVATAR_PATHS)}"
            )
        else:
            unknown.append(path)
    if unknown:
        raise ExpertPackageError(
            f"archive contains entries that are not part of an expert package: "
            f"{_listed(unknown)} — {_SHAPE}"
        )
    if len(avatars) > 1:
        raise ExpertPackageError("archive carries more than one avatar")
    if len(skills) > MAX_PACKAGE_SKILLS:
        raise ExpertPackageError(
            f"archive carries {len(skills)} skills; the limit is "
            f"{MAX_PACKAGE_SKILLS}",
            over_limit=True,
        )
    return members[MANIFEST_NAME], skills, avatars[0] if avatars else None


def _listed(paths: list[str]) -> str:
    shown = ", ".join(f"'{path[:120]}'" for path in paths[:_NAMED_PATHS])
    remaining = len(paths) - _NAMED_PATHS
    return f"{shown} and {remaining} more" if remaining > 0 else shown


def _check_declared_sizes(
    manifest: zipfile.ZipInfo, skills: dict[str, _Members], avatar: _Avatar
) -> None:
    """Each kind of member against its own cap, from the sizes the central
    directory declares, before any member is decompressed."""
    if manifest.file_size > MAX_MANIFEST_BYTES:
        raise ExpertPackageError(
            f"{MANIFEST_NAME} is {manifest.file_size} bytes; the limit is "
            f"{MAX_MANIFEST_BYTES}",
            over_limit=True,
        )
    for slug, members in sorted(skills.items()):
        for path, info in sorted(members.items()):
            if info.file_size > MAX_PACKAGE_FILE_BYTES:
                raise ExpertPackageError(
                    f"skill '{slug[:120]}' file '{path[:120]}' unpacks to "
                    f"{info.file_size} bytes; the limit is {MAX_PACKAGE_FILE_BYTES}",
                    over_limit=True,
                )
    if avatar and avatar[1].file_size > MAX_AVATAR_BYTES:
        raise ExpertPackageError(
            f"avatar is {avatar[1].file_size} bytes; the limit is "
            f"{MAX_AVATAR_BYTES}",
            over_limit=True,
        )


def _manifest(raw: bytes) -> ExpertManifest:
    try:
        return ExpertManifest.model_validate_json(raw.decode("utf-8"))
    except UnicodeDecodeError:
        raise ExpertPackageError(f"{MANIFEST_NAME} is not UTF-8 text")
    except ValidationError as exc:
        raise ExpertPackageError(f"{MANIFEST_NAME} is not a valid expert: {exc}")


def _skill(archive: zipfile.ZipFile, slug: str, members: _Members) -> SkillPackage:
    """One ``skills/<slug>/`` directory as a skill package, held to exactly the
    rules a standalone skill upload is held to."""
    root = members.pop(ROOT_SKILL_MD, None)
    if root is None:
        raise ExpertPackageError(f"skill '{slug[:120]}' has no SKILL.md")
    try:
        skill_md = archive.read(root).decode("utf-8")
    except UnicodeDecodeError:
        raise ExpertPackageError(f"SKILL.md of skill '{slug[:120]}' is not UTF-8 text")
    package = SkillPackage(
        skill_md=skill_md,
        files=[
            SkillFile(
                relative_path=path,
                content=archive.read(info),
                is_executable=bool(mode(info) & 0o111),
            )
            for path, info in sorted(members.items())
        ],
    )
    try:
        validate_package(package)
    except SkillPackageError as exc:
        raise ExpertPackageError(
            f"skill '{slug[:120]}': {exc}", over_limit=exc.over_limit
        )
    return package


def _check_skills_agree(
    manifest: ExpertManifest, skills: dict[str, SkillPackage]
) -> None:
    """A manifest and a tree that disagree would silently drop a skill or
    install one the expert never listed."""
    listed = {skill.slug for skill in manifest.skills}
    if missing := sorted(listed - set(skills)):
        raise ExpertPackageError(
            f"{MANIFEST_NAME} lists {_listed(missing)} but the archive carries no "
            f"{SKILLS_PREFIX} folder for them"
        )
    if extra := sorted(set(skills) - listed):
        raise ExpertPackageError(
            f"archive carries skill folders {_listed(extra)} that "
            f"{MANIFEST_NAME} does not list"
        )


def _check_avatar_agrees(manifest: ExpertManifest, path: str | None) -> None:
    declared = manifest.avatar.path if manifest.avatar else None
    if declared and declared != path:
        raise ExpertPackageError(
            f"{MANIFEST_NAME} names the avatar '{declared}' but the archive does "
            "not carry it"
        )
    if path and not declared:
        raise ExpertPackageError(
            f"archive carries '{path}' but {MANIFEST_NAME} does not name it as "
            "the avatar"
        )
