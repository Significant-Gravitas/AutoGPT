"""Tests for the ``.zip`` transport: what an archive may carry into a skill
folder, and what a stored package looks like on the way back out."""

import io
import struct
import zipfile
from unittest.mock import MagicMock

import pytest

from backend.api.features.skill_zip import (
    MAX_ZIP_BYTES,
    package_from_zip,
    zip_from_package,
)
from backend.copilot.tools.skills import (
    MAX_PACKAGE_BYTES,
    MAX_PACKAGE_FILE_BYTES,
    MAX_PACKAGE_FILES,
    SkillFile,
    SkillPackage,
    SkillPackageError,
)

SKILL_MD = "---\nname: demo\ndescription: A demo package.\n---\n\n# Demo\n"


def _zip(
    members: dict[str, bytes | str],
    *,
    executable: set[str] = set(),
    directories: tuple[str, ...] = (),
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in directories:
            archive.writestr(zipfile.ZipInfo(name), b"")
        for name, content in members.items():
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100755 if name in executable else 0o100644) << 16
            archive.writestr(info, content)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_download_then_upload_is_the_same_tree():
    """A stored package written out and read back is byte-identical, bits
    included — otherwise a download is not a backup."""
    package = SkillPackage(
        skill_md=SKILL_MD,
        files=[
            SkillFile(relative_path="references/API.md", content=b"# API\n"),
            SkillFile(
                relative_path="scripts/run.py",
                content=b"#!/usr/bin/env python3\nprint(1)\n",
                is_executable=True,
            ),
        ],
    )

    restored = package_from_zip(zip_from_package(package))

    assert restored.skill_md == package.skill_md
    assert [(f.relative_path, f.content, f.is_executable) for f in restored.files] == [
        (f.relative_path, f.content, f.is_executable) for f in package.files
    ]


def test_an_uploaded_scripts_executable_bit_is_read():
    """Real packages ship 100755 scripts; ``./run.py`` stops working without
    the bit, and the workspace has nowhere else to keep it."""
    package = package_from_zip(
        _zip(
            {"SKILL.md": SKILL_MD, "scripts/run.py": b"x"},
            executable={"scripts/run.py"},
        )
    )
    assert [(f.relative_path, f.is_executable) for f in package.files] == [
        ("scripts/run.py", True)
    ]


def test_an_archive_with_no_unix_modes_is_read_as_non_executable():
    """A zip written on Windows carries no mode field at all."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("SKILL.md", SKILL_MD)
        archive.writestr("scripts/run.py", b"x")

    package = package_from_zip(buffer.getvalue())
    assert [f.is_executable for f in package.files] == [False]


def test_a_download_carries_a_fixed_timestamp():
    """A per-download stamp would make two downloads of an unchanged skill
    differ; the workspace keeps no per-file mtime to use instead."""
    data = zip_from_package(
        SkillPackage(
            skill_md=SKILL_MD,
            files=[SkillFile(relative_path="a.txt", content=b"a")],
        )
    )
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert {i.date_time for i in archive.infolist()} == {(1980, 1, 1, 0, 0, 0)}


# ---------------------------------------------------------------------------
# What an archive may not carry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    ["../escape.md", "/etc/passwd", "a\\b.md", "nested/../../escape.md"],
)
def test_a_member_that_escapes_the_skill_folder_is_refused(path: str):
    with pytest.raises(SkillPackageError) as exc:
        package_from_zip(_zip({"SKILL.md": SKILL_MD, path: b"x"}))
    assert not exc.value.over_limit


def test_a_symlink_member_is_refused():
    """A symlink passes every path check and then points wherever it likes, so
    it is refused on its mode rather than its name."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("SKILL.md", SKILL_MD)
        link = zipfile.ZipInfo("scripts/secrets")
        link.external_attr = (0o120777) << 16
        archive.writestr(link, "/etc/passwd")

    with pytest.raises(SkillPackageError, match="symlink"):
        package_from_zip(buffer.getvalue())


def test_an_encrypted_member_is_refused():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("SKILL.md", SKILL_MD)
        archive.writestr("secret.bin", b"x")

    with pytest.raises(SkillPackageError, match="encrypted"):
        package_from_zip(_with_encryption_flag(buffer.getvalue()))


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


def test_a_bomb_is_refused_without_a_member_being_read(monkeypatch):
    """The declared sizes are the whole check: 22 MiB of zeros compresses to a
    few KB, so reading first is how a bomb lands in memory."""
    member = b"\0" * MAX_PACKAGE_FILE_BYTES
    data = _zip({"SKILL.md": SKILL_MD, **{f"f{i}.bin": member for i in range(11)}})
    assert len(data) < MAX_PACKAGE_BYTES

    opened = MagicMock(side_effect=AssertionError("a member was read"))
    monkeypatch.setattr(zipfile.ZipFile, "open", opened)

    with pytest.raises(SkillPackageError) as exc:
        package_from_zip(data)
    assert exc.value.over_limit
    opened.assert_not_called()


def test_a_file_over_the_per_file_cap_is_refused_unread(monkeypatch):
    """The package validator would also catch the size — but only after the
    member has been decompressed, which is the cost this check avoids."""
    data = _zip({"SKILL.md": SKILL_MD, "big.bin": b"\0" * (MAX_PACKAGE_FILE_BYTES + 1)})
    opened = MagicMock(side_effect=AssertionError("a member was read"))
    monkeypatch.setattr(zipfile.ZipFile, "open", opened)

    with pytest.raises(SkillPackageError) as exc:
        package_from_zip(data)
    assert exc.value.over_limit
    assert str(MAX_PACKAGE_FILE_BYTES) in str(exc.value)
    opened.assert_not_called()


def test_too_many_files_names_the_field_and_the_limit():
    members = {f"f{i}.txt": b"x" for i in range(MAX_PACKAGE_FILES + 1)}
    with pytest.raises(SkillPackageError) as exc:
        package_from_zip(_zip({"SKILL.md": SKILL_MD, **members}))
    assert exc.value.over_limit
    assert f"{MAX_PACKAGE_FILES + 1} files" in str(exc.value)
    assert str(MAX_PACKAGE_FILES) in str(exc.value)


def test_a_package_at_the_files_cap_is_accepted():
    """The cap counts siblings, so the root SKILL.md is not one of the 100."""
    members = {f"f{i}.txt": b"x" for i in range(MAX_PACKAGE_FILES)}
    package = package_from_zip(_zip({"SKILL.md": SKILL_MD, **members}))
    assert len(package.files) == MAX_PACKAGE_FILES


def test_the_body_cap_is_the_compressed_cap_too():
    """One number bounds the request body and the unpacked tree."""
    assert MAX_ZIP_BYTES == MAX_PACKAGE_BYTES


# ---------------------------------------------------------------------------
# Finding the package root
# ---------------------------------------------------------------------------


def test_a_wrapping_directory_is_unwrapped():
    """What a "download zip" button produces: one directory holding the
    package."""
    package = package_from_zip(
        _zip(
            {
                "webapp-testing-main/SKILL.md": SKILL_MD,
                "webapp-testing-main/scripts/run.py": b"print(1)",
            },
            directories=("webapp-testing-main/", "webapp-testing-main/scripts/"),
        )
    )
    assert package.skill_md == SKILL_MD
    assert [f.relative_path for f in package.files] == ["scripts/run.py"]


def test_a_package_with_its_own_root_is_never_unwrapped():
    package = package_from_zip(
        _zip({"SKILL.md": SKILL_MD, "scripts/run.py": b"print(1)"})
    )
    assert [f.relative_path for f in package.files] == ["scripts/run.py"]


def test_two_packages_in_one_archive_are_refused():
    with pytest.raises(SkillPackageError, match="no SKILL.md"):
        package_from_zip(_zip({"one/SKILL.md": SKILL_MD, "two/SKILL.md": SKILL_MD}))


def test_two_root_skill_mds_are_refused():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("SKILL.md", SKILL_MD)
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("SKILL.md", SKILL_MD.replace("demo", "other"))

    with pytest.raises(SkillPackageError, match="twice"):
        package_from_zip(buffer.getvalue())


def test_a_nested_skill_md_is_an_ordinary_file():
    """Packages ship example SKILL.mds; only the root one names the skill."""
    package = package_from_zip(
        _zip({"SKILL.md": SKILL_MD, "references/SKILL.md": b"# example\n"})
    )
    assert [f.relative_path for f in package.files] == ["references/SKILL.md"]


def test_an_archive_without_a_skill_md_is_refused():
    with pytest.raises(SkillPackageError, match="no SKILL.md"):
        package_from_zip(_zip({"notes.md": b"x"}))


def test_an_empty_archive_is_refused():
    with pytest.raises(SkillPackageError, match="no files"):
        package_from_zip(_zip({}))


def test_a_member_with_a_corrupt_crc_is_refused_as_a_bad_upload():
    """A member's CRC is verified only as it decompresses, so a corrupt archive
    survives the open and fails on the read — where it must still read as a bad
    upload rather than an unhandled error the route answers 500 to."""
    with pytest.raises(SkillPackageError, match="could not be read"):
        package_from_zip(
            _with_a_corrupt_member(_zip({"SKILL.md": SKILL_MD, "a.txt": b"x" * 400}))
        )


def _with_a_corrupt_member(data: bytes) -> bytes:
    """Flip a byte inside the last member's deflate stream, leaving every header
    intact so the central directory still parses."""
    out = bytearray(data)
    header = out.rfind(b"PK\x03\x04")
    name_len, extra_len = struct.unpack_from("<HH", out, header + 26)
    out[header + 30 + name_len + extra_len + 2] ^= 0xFF
    return bytes(out)


def test_something_that_is_not_a_zip_is_refused():
    with pytest.raises(SkillPackageError, match="not a readable zip"):
        package_from_zip(b"---\nname: demo\n---\n")


def test_a_skill_md_that_is_not_utf8_is_refused():
    with pytest.raises(SkillPackageError, match="UTF-8"):
        package_from_zip(_zip({"SKILL.md": b"\xff\xfe binary"}))


def test_directory_entries_are_skipped():
    package = package_from_zip(
        _zip({"SKILL.md": SKILL_MD}, directories=("scripts/", "references/"))
    )
    assert package.files == []
