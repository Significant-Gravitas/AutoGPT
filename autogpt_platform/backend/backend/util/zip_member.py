"""Zip member reads under a byte cap, for stored and deflate members only."""

import os
import zipfile
from collections.abc import Iterator
from typing import IO, Literal

READ_CHUNK_BYTES = 64 * 1024
_BOUNDED_METHODS = frozenset({zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED})
_ENCRYPTED_FLAG = 0x1


class ZipMemberError(ValueError):
    """A member these readers will not read."""


class ZipMemberTooLarge(ZipMemberError):
    """A member, or the archive as a whole, unpacks past its cap.

    ``cap`` says which: one member's, or the whole archive's. ``counted`` is
    the bytes unpacked when the read stopped, and ``limit`` the cap passed.
    """

    def __init__(
        self,
        message: str,
        *,
        cap: Literal["member", "archive"],
        counted: int,
        limit: int,
    ) -> None:
        super().__init__(message)
        self.cap: Literal["member", "archive"] = cap
        self.counted = counted
        self.limit = limit


def iter_member(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo, *, max_bytes: int
) -> Iterator[bytes]:
    """Yield a member's bytes in chunks, raising :class:`ZipMemberTooLarge` as
    soon as more than ``max_bytes`` have been produced."""
    if info.flag_bits & _ENCRYPTED_FLAG:
        raise ZipMemberError(f"'{info.filename[:120]}' is encrypted")
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
                    f"'{info.filename[:120]}' unpacks to more than {max_bytes} bytes",
                    cap="member",
                    counted=produced,
                    limit=max_bytes,
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
                        f"archive unpacks to more than {max_total_bytes} bytes",
                        cap="archive",
                        counted=total,
                        limit=max_total_bytes,
                    )
