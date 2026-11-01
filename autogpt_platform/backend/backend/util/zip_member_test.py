"""Tests for zip member reads under a byte cap."""

import io
import zipfile
from unittest.mock import MagicMock

import pytest

from backend.util.zip_member import (
    READ_CHUNK_BYTES,
    ZipMemberError,
    ZipMemberTooLarge,
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
