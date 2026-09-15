"""The ``.zip`` transport for a skill package.

A skill is a directory, so upload and download carry it as an archive. The caps
and the path rules live in :mod:`backend.copilot.tools.skills`; what is here is
only what an archive can carry that a path check cannot see — a symlink, an
encrypted member, a declared size that would unpack to gigabytes.

Both entry points are CPU-bound and synchronous; callers run them off the event
loop.
"""

import io
import zipfile

from backend.api.features.zip_members import add_member, checked_members, mode
from backend.copilot.tools.skills import (
    MAX_PACKAGE_BYTES,
    MAX_PACKAGE_FILE_BYTES,
    MAX_PACKAGE_FILES,
    SkillFile,
    SkillPackage,
    SkillPackageError,
    validate_package,
)

ROOT_SKILL_MD = "SKILL.md"
# A compressed archive already past the uncompressed cap cannot hold a package
# that fits, so one number bounds both the request body and the tree.
MAX_ZIP_BYTES = MAX_PACKAGE_BYTES


def package_from_zip(data: bytes) -> SkillPackage:
    """Unpack an uploaded archive into a validated :class:`SkillPackage`.

    Every refusal is a :class:`SkillPackageError`, ``over_limit`` set where the
    REST edge owes a 413. Sizes are checked from the central directory, so a
    zip bomb is refused without a member being decompressed.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, OSError) as exc:
        raise SkillPackageError(f"upload is not a readable zip archive: {exc}")
    with archive:
        members = _unwrapped(
            _within_the_files_cap(_checked_members(archive.infolist()))
        )
        root = members.pop(ROOT_SKILL_MD)
        # A member's CRC is verified as it decompresses, so a corrupt archive
        # opens cleanly and fails here.
        try:
            skill_md = _decoded(archive.read(root))
            files = [
                SkillFile(
                    relative_path=path,
                    content=archive.read(info),
                    is_executable=bool(mode(info) & 0o111),
                )
                for path, info in members.items()
            ]
        except (zipfile.BadZipFile, OSError) as exc:
            raise SkillPackageError(f"archive member could not be read: {exc}")
    package = SkillPackage(skill_md=skill_md, files=files)
    validate_package(package)
    return package


def zip_from_package(package: SkillPackage) -> bytes:
    """Write a stored package back out: ``SKILL.md`` at the root, siblings at
    their relative paths, executable bits in the mode field so a script that
    was runnable still is after a round trip."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        add_member(archive, ROOT_SKILL_MD, package.skill_md.encode("utf-8"), False)
        for entry in package.files:
            add_member(archive, entry.relative_path, entry.content, entry.is_executable)
    return buffer.getvalue()


def _checked_members(infos: list[zipfile.ZipInfo]) -> dict[str, zipfile.ZipInfo]:
    return checked_members(
        infos,
        max_file_bytes=MAX_PACKAGE_FILE_BYTES,
        max_total_bytes=MAX_PACKAGE_BYTES,
        error=SkillPackageError,
    )


def _within_the_files_cap(
    members: dict[str, zipfile.ZipInfo],
) -> dict[str, zipfile.ZipInfo]:
    """The files cap counts siblings, so the root SKILL.md is the one over it.
    It stays here because only a skill package has a per-package file count."""
    if len(members) > MAX_PACKAGE_FILES + 1:
        raise SkillPackageError(
            f"package has {len(members) - 1} files; the limit is "
            f"{MAX_PACKAGE_FILES}",
            over_limit=True,
        )
    return members


def _unwrapped(members: dict[str, zipfile.ZipInfo]) -> dict[str, zipfile.ZipInfo]:
    """Strip the wrapping directory a "download zip" button adds, so
    ``webapp-testing-main/SKILL.md`` is the package root.

    Only when one directory holds every member and the ``SKILL.md`` is the one
    inside it, so a package whose own root is present is never rewritten.
    """
    tops = {name.split("/", 1)[0] for name in members}
    if len(tops) == 1 and (top := tops.pop()) and f"{top}/{ROOT_SKILL_MD}" in members:
        members = {name[len(top) + 1 :]: info for name, info in members.items()}
    if ROOT_SKILL_MD not in members:
        raise SkillPackageError(
            "archive has no SKILL.md at its root — a skill package is a "
            "SKILL.md and the files beside it"
        )
    return members


def _decoded(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        raise SkillPackageError("SKILL.md is not UTF-8 text")
