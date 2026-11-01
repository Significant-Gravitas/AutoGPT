"""Bounded reads of zip archive members.

These readers count the bytes a member decompresses to, as they are
produced, and stop once a cap is passed, whatever size the central directory
records for it.

Only stored and deflated members are read. ``zipfile`` inflates a deflate
member in steps no larger than the read asked for, but decompresses each
compressed chunk of the other methods (bzip2, LZMA) whole, which no read size
can bound. Office files and ordinary zip tools use deflate.
"""

import os
import zipfile
from collections.abc import Iterator
from typing import IO

READ_CHUNK_BYTES = 64 * 1024
_BOUNDED_METHODS = frozenset({zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED})


class ZipMemberError(ValueError):
    """A member these readers will not read."""


class ZipMemberTooLarge(ZipMemberError):
    """A member, or the archive as a whole, unpacks past its cap."""


def iter_member(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo, *, max_bytes: int
) -> Iterator[bytes]:
    """Yield a member's bytes in chunks, raising :class:`ZipMemberTooLarge` as
    soon as more than ``max_bytes`` have been produced."""
    if info.compress_type not in _BOUNDED_METHODS:
        raise ZipMemberError(
            f"'{info.filename[:120]}' uses an unsupported compression method"
        )
    produced = 0
    with archive.open(info) as member:
        while chunk := member.read(READ_CHUNK_BYTES):
            produced += len(chunk)
            if produced > max_bytes:
                raise ZipMemberTooLarge(
                    f"'{info.filename[:120]}' unpacks to more than {max_bytes} bytes"
                )
            yield chunk


def read_member(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo, *, max_bytes: int
) -> bytes:
    """A member's bytes, refused once more than ``max_bytes`` are produced."""
    return b"".join(iter_member(archive, info, max_bytes=max_bytes))


def check_unpacked_size(
    file: str | os.PathLike[str] | IO[bytes],
    *,
    max_member_bytes: int,
    max_total_bytes: int,
) -> None:
    """Read every member of an archive and discard the bytes, raising
    :class:`ZipMemberTooLarge` once one member or all of them together unpack
    past their cap."""
    total = 0
    with zipfile.ZipFile(file) as archive:
        for info in archive.infolist():
            for chunk in iter_member(archive, info, max_bytes=max_member_bytes):
                total += len(chunk)
                if total > max_total_bytes:
                    raise ZipMemberTooLarge(
                        f"archive unpacks to more than {max_total_bytes} bytes"
                    )
