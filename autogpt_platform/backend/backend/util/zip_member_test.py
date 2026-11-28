"""Tests for zip member reads under a byte cap."""

import io
import struct
import warnings
import zipfile
from unittest.mock import MagicMock

import pytest

from backend.util.zip_member import (
    READ_CHUNK_BYTES,
    ZipMemberError,
    ZipMemberTooLarge,
    check_members_disjoint,
    check_unpacked_size,
    iter_member,
    read_member,
)


def _zip(members: dict[str, bytes], method: int = zipfile.ZIP_DEFLATED) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", method) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def _member(
    data: bytes, name: str = "a.bin"
) -> tuple[zipfile.ZipFile, zipfile.ZipInfo]:
    archive = zipfile.ZipFile(io.BytesIO(data))
    return archive, archive.getinfo(name)


@pytest.mark.parametrize("method", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_a_member_within_the_cap_is_read_whole(method: int):
    content = bytes(range(256)) * 400
    archive, info = _member(_zip({"a.bin": content}, method))

    assert read_member(archive, info, max_bytes=len(content)) == content


def test_a_member_past_the_cap_is_refused():
    archive, info = _member(_zip({"a.bin": b"\0" * 10_001}))

    with pytest.raises(ZipMemberTooLarge, match="'a.bin' unpacks to more than 10000"):
        read_member(archive, info, max_bytes=10_000)


def test_the_read_stops_at_the_first_chunk_past_the_cap():
    archive, info = _member(_zip({"a.bin": b"\0" * (READ_CHUNK_BYTES * 50)}))
    chunks = iter_member(archive, info, max_bytes=READ_CHUNK_BYTES)

    assert len(next(chunks)) == READ_CHUNK_BYTES
    with pytest.raises(ZipMemberTooLarge):
        next(chunks)


def test_the_cap_is_counted_on_the_bytes_read():
    archive, info = _member(_zip({"a.bin": b"\0" * 5_000}))
    info.file_size = 10**9

    assert len(read_member(archive, info, max_bytes=5_000)) == 5_000


@pytest.mark.parametrize("method", [zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA])
def test_a_member_of_another_method_is_refused_unread(method: int, monkeypatch):
    archive, info = _member(_zip({"a.bin": b"\0" * 1_000}, method))
    opened = MagicMock(side_effect=AssertionError("the member was opened"))
    monkeypatch.setattr(archive, "open", opened)

    with pytest.raises(ZipMemberError, match="unsupported compression method"):
        read_member(archive, info, max_bytes=10_000_000)
    opened.assert_not_called()


def test_an_encrypted_member_is_refused_unread(monkeypatch):
    archive, info = _member(_zip({"a.bin": b"\0" * 1_000}))
    info.flag_bits |= 0x1
    opened = MagicMock(side_effect=AssertionError("the member was opened"))
    monkeypatch.setattr(archive, "open", opened)

    with pytest.raises(ZipMemberError, match="'a.bin' is encrypted"):
        read_member(archive, info, max_bytes=10_000_000)
    opened.assert_not_called()


def test_an_archive_within_both_caps_passes():
    data = _zip({"a.bin": b"\0" * 1_000, "b.bin": b"\0" * 1_000})

    check_unpacked_size(io.BytesIO(data), max_member_bytes=1_000, max_total_bytes=2_000)


def test_one_member_past_its_cap_fails_the_archive():
    data = _zip({"a.bin": b"\0" * 10, "b.bin": b"\0" * 1_001})

    with pytest.raises(ZipMemberTooLarge, match="'b.bin'"):
        check_unpacked_size(
            io.BytesIO(data), max_member_bytes=1_000, max_total_bytes=10_000
        )


def test_members_together_past_the_total_fail_the_archive():
    data = _zip({f"{i}.bin": b"\0" * 1_000 for i in range(3)})

    with pytest.raises(ZipMemberTooLarge, match="archive unpacks to more than 2500"):
        check_unpacked_size(
            io.BytesIO(data), max_member_bytes=1_000, max_total_bytes=2_500
        )


def _entries_sharing_one_member(count: int) -> bytes:
    """``count`` central-directory entries, all naming and pointing at the
    first member."""
    buffer = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for _ in range(count):
                archive.writestr("a.bin", b"\0" * 100)
    return _with_central_directory_field(buffer.getvalue(), offset=42, value=0)


def _first_entry_claiming_the_whole_archive() -> bytes:
    data = _zip({"a.bin": b"\0" * 100, "b.bin": b"\0" * 100})
    start_dir = zipfile.ZipFile(io.BytesIO(data)).start_dir
    return _with_central_directory_field(
        data, offset=20, value=start_dir, first_only=True
    )


def _with_central_directory_field(
    data: bytes, *, offset: int, value: int, first_only: bool = False
) -> bytes:
    out = bytearray(data)
    start = 0
    while (found := out.find(b"PK\x01\x02", start)) != -1:
        struct.pack_into("<I", out, found + offset, value)
        if first_only:
            break
        start = found + 4
    return bytes(out)


@pytest.mark.parametrize(
    "data",
    [_entries_sharing_one_member(50), _first_entry_claiming_the_whole_archive()],
    ids=["shared member", "overlapping data"],
)
def test_an_archive_whose_entries_overlap_is_refused_unread(data: bytes, monkeypatch):
    opened = MagicMock(side_effect=AssertionError("a member was read"))
    monkeypatch.setattr(zipfile.ZipFile, "open", opened)

    with pytest.raises(ZipMemberError, match="archive entries overlap"):
        check_unpacked_size(
            io.BytesIO(data), max_member_bytes=1_000, max_total_bytes=100_000
        )
    opened.assert_not_called()


def test_an_ordinary_archive_has_disjoint_members():
    data = _zip({f"{i}.bin": bytes([i]) * 1_000 for i in range(5)})

    check_members_disjoint(zipfile.ZipFile(io.BytesIO(data)))
