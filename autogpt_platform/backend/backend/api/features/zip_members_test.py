"""Tests for the archive rules both package readers share: what a zip can
carry that a path check never sees, and the mode bits a round trip must keep."""

import io
import zipfile

import pytest

from backend.api.features.zip_members import (
    ZIP_EPOCH,
    add_member,
    checked_members,
    mode,
)

MAX_FILE = 1024
MAX_TOTAL = 4096


class PackageError(ValueError):
    """Stands in for the caller's own error type, so these tests prove the
    refusals are reported through whatever the caller injects."""

    def __init__(self, message: str, *, over_limit: bool = False):
        super().__init__(message)
        self.over_limit = over_limit


def _infos(
    members: dict[str, bytes],
    *,
    executable: set[str] = set(),
    directories: tuple[str, ...] = (),
) -> list[zipfile.ZipInfo]:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in directories:
            archive.writestr(zipfile.ZipInfo(name), b"")
        for name, content in members.items():
            info = zipfile.ZipInfo(name)
            info.external_attr = (0o100755 if name in executable else 0o100644) << 16
            archive.writestr(info, content)
    return _read(buffer.getvalue())


def _read(data: bytes) -> list[zipfile.ZipInfo]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return archive.infolist()


def _checked(infos: list[zipfile.ZipInfo]) -> dict[str, zipfile.ZipInfo]:
    return checked_members(
        infos,
        max_file_bytes=MAX_FILE,
        max_total_bytes=MAX_TOTAL,
        error=PackageError,
    )


def test_regular_members_come_back_keyed_by_path():
    members = _checked(_infos({"a.txt": b"a", "dir/b.txt": b"b"}))
    assert sorted(members) == ["a.txt", "dir/b.txt"]


def test_directory_entries_are_skipped():
    assert list(_checked(_infos({"a.txt": b"a"}, directories=("dir/",)))) == ["a.txt"]


def test_an_encrypted_member_is_refused():
    with pytest.raises(PackageError, match="encrypted"):
        _checked(_read(_with_encryption_flag(_zip_bytes({"a.txt": b"a"}))))


def test_a_symlink_member_is_refused():
    """A symlink passes every path check and then points wherever it likes, so
    it is refused on its mode rather than its name."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        link = zipfile.ZipInfo("secrets")
        link.external_attr = 0o120777 << 16
        archive.writestr(link, "/etc/passwd")

    with pytest.raises(PackageError, match="symlink"):
        _checked(_read(buffer.getvalue()))


def test_a_member_that_appears_twice_is_refused():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("a.txt", b"first")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("a.txt", b"second")

    with pytest.raises(PackageError, match="twice"):
        _checked(_read(buffer.getvalue()))


def test_a_member_over_the_per_file_cap_is_refused():
    with pytest.raises(PackageError) as exc:
        _checked(_infos({"big.bin": b"\0" * (MAX_FILE + 1)}))
    assert exc.value.over_limit
    assert str(MAX_FILE) in str(exc.value)


def test_an_archive_over_the_total_cap_is_refused():
    members = {f"f{i}.bin": b"\0" * MAX_FILE for i in range(MAX_TOTAL // MAX_FILE + 1)}
    with pytest.raises(PackageError) as exc:
        _checked(_infos(members))
    assert exc.value.over_limit
    assert str(MAX_TOTAL) in str(exc.value)


def test_an_archive_with_no_files_is_refused():
    with pytest.raises(PackageError, match="no files"):
        _checked(_infos({}, directories=("dir/",)))


def test_an_executable_bit_survives_a_write_and_a_read():
    """Real packages ship 100755 scripts; ``./run.py`` stops working without
    the bit, and no store outside the archive keeps it."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        add_member(archive, "run.py", b"print(1)", True)
        add_member(archive, "notes.md", b"# notes", False)

    members = _checked(_read(buffer.getvalue()))
    assert {name: bool(mode(info) & 0o111) for name, info in members.items()} == {
        "run.py": True,
        "notes.md": False,
    }


def test_a_member_with_no_unix_mode_is_never_executable():
    """A zip written on Windows carries no mode field at all, and whatever a
    writer stamps in its place must not read as runnable."""
    assert mode(zipfile.ZipInfo("run.py")) == 0

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("run.py", b"print(1)")

    assert mode(_read(buffer.getvalue())[0]) & 0o111 == 0


def test_every_written_member_carries_the_fixed_epoch():
    """A live stamp would make two downloads of an unchanged package differ."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        add_member(archive, "a.txt", b"a", False)

    assert {i.date_time for i in _read(buffer.getvalue())} == {ZIP_EPOCH}


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def _with_encryption_flag(data: bytes) -> bytes:
    """Set the general-purpose encryption bit in every header. ``writestr``
    rewrites ``flag_bits``, so the standard library cannot build an encrypted
    member any other way."""
    out = bytearray(data)
    for signature, flag_offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        start = 0
        while (found := out.find(signature, start)) != -1:
            out[found + flag_offset] |= 0x1
            start = found + 4
    return bytes(out)
