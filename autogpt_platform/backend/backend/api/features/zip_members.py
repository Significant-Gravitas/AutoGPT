"""What a zip archive may carry, independent of the package inside it.

A skill package and an expert package disagree about every path rule and share
every archive rule: a symlink points wherever it likes, an encrypted member
cannot be read, and a declared size is the only thing that tells a bomb from a
folder before a member is decompressed. Those live here so one fix reaches both
readers, and so the writers stamp the same epoch and the same mode bits.

The caller passes its own error type because the REST edge maps that type to a
status code; ``over_limit`` is the flag it turns into a 413.
"""

import zipfile
from typing import Protocol

# A live stamp would make two downloads of an unchanged package differ, and
# neither store keeps a per-file mtime to use instead.
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)

_ENCRYPTED_FLAG = 0x1
_FILE_TYPE_MASK = 0o170000
_SYMLINK_TYPE = 0o120000


class MemberError(Protocol):
    """A package error as both packages already raise one."""

    def __call__(self, message: str, *, over_limit: bool = False) -> Exception: ...


def checked_members(
    infos: list[zipfile.ZipInfo],
    *,
    max_file_bytes: int,
    max_total_bytes: int,
    error: MemberError,
) -> dict[str, zipfile.ZipInfo]:
    """Every regular member by its archive path, refusing what an archive can
    carry that a path check never sees, and the sizes the central directory
    declares — all before the first member is read."""
    members: dict[str, zipfile.ZipInfo] = {}
    total = 0
    for info in infos:
        if info.is_dir():
            continue
        named = f"member '{info.filename[:120]}'"
        if info.flag_bits & _ENCRYPTED_FLAG:
            raise error(f"{named} is encrypted")
        if mode(info) & _FILE_TYPE_MASK == _SYMLINK_TYPE:
            raise error(f"{named} is a symlink")
        if info.filename in members:
            raise error(f"{named} appears twice in the archive")
        if info.file_size > max_file_bytes:
            raise error(
                f"{named} unpacks to {info.file_size} bytes; the limit is "
                f"{max_file_bytes}",
                over_limit=True,
            )
        total += info.file_size
        if total > max_total_bytes:
            raise error(
                f"archive unpacks to more than {max_total_bytes} bytes",
                over_limit=True,
            )
        members[info.filename] = info
    if not members:
        raise error("archive holds no files")
    return members


def mode(info: zipfile.ZipInfo) -> int:
    """The unix mode a member was archived with, or 0 when the archive carries
    none — one written on Windows, or by a library that omits it."""
    return info.external_attr >> 16


def add_member(
    archive: zipfile.ZipFile, path: str, content: bytes, is_executable: bool
) -> None:
    """Write one member with a fixed stamp, so two exports of unchanged content
    are byte-identical."""
    info = zipfile.ZipInfo(path, date_time=ZIP_EPOCH)
    info.compress_type = zipfile.ZIP_DEFLATED
    # Neither store has mode bits — this is where the stored flag becomes one.
    info.external_attr = (0o100755 if is_executable else 0o100644) << 16
    archive.writestr(info, content)
