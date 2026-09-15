"""Tests for the ``.expert.zip`` transport: what an archive may carry into a
team, and what an exported expert looks like on the way back out."""

import io
import json
import zipfile
from unittest.mock import MagicMock

import pytest

from backend.api.features.experts.expert_zip import (
    MAX_ZIP_BYTES,
    package_from_zip,
    zip_from_package,
)
from backend.api.features.experts.package_model import (
    MAX_AVATAR_BYTES,
    ExpertManifest,
    ExpertPackage,
    ExpertPackageError,
    PackagedAvatar,
    PackagedIdentity,
    PackagedSkill,
    PackagedSoul,
)
from backend.copilot.tools.skills import (
    MAX_PACKAGE_BYTES,
    MAX_PACKAGE_FILE_BYTES,
    SkillFile,
    SkillPackage,
)

SKILL_MD = "---\nname: research\ndescription: A demo skill.\n---\n\n# Research\n"


def _package(**overrides) -> ExpertPackage:
    manifest = ExpertManifest(
        identity=PackagedIdentity(name="Maria", role="Ops lead"),
        soul=PackagedSoul(identity="Careful and brief."),
        skills=[PackagedSkill(slug="research", name="research")],
        **overrides,
    )
    return ExpertPackage(
        manifest=manifest,
        skills={
            "research": SkillPackage(
                skill_md=SKILL_MD,
                files=[
                    SkillFile(relative_path="references/API.md", content=b"# API\n"),
                    SkillFile(
                        relative_path="scripts/run.py",
                        content=b"print(1)\n",
                        is_executable=True,
                    ),
                ],
            )
        },
    )


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


def _manifest(**overrides) -> str:
    payload = {
        "format_version": 1,
        "identity": {"name": "Maria"},
        "skills": [{"slug": "research", "name": "research"}],
    }
    payload.update(overrides)
    return json.dumps(payload)


def _minimal(**members: bytes | str) -> bytes:
    return _zip(
        {
            "expert.json": _manifest(),
            "skills/research/SKILL.md": SKILL_MD,
            **members,
        }
    )


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_an_export_read_back_is_the_same_expert():
    package = _package()

    restored = package_from_zip(zip_from_package(package))

    assert restored.manifest == package.manifest
    assert restored.skills["research"].skill_md == SKILL_MD
    assert [
        (f.relative_path, f.content, f.is_executable)
        for f in restored.skills["research"].files
    ] == [
        ("references/API.md", b"# API\n", False),
        ("scripts/run.py", b"print(1)\n", True),
    ]


def test_two_exports_of_the_same_expert_are_byte_identical():
    """A download is a backup only if an unchanged expert keeps producing the
    same file; a live timestamp would break every checksum."""
    assert zip_from_package(_package()) == zip_from_package(_package())


def test_an_avatar_round_trips_with_its_mime_type():
    exported = _package(avatar=PackagedAvatar(kind="file", path="avatar.png"))
    package = exported.model_copy(update={"avatar_bytes": b"\x89PNG\r\n\x1a\n"})

    restored = package_from_zip(zip_from_package(package))

    assert restored.avatar_bytes == b"\x89PNG\r\n\x1a\n"
    assert restored.avatar_mime == "image/png"


def test_a_wrapping_directory_is_unwrapped():
    """What a "download zip" button produces: one directory holding the
    package."""
    package = package_from_zip(
        _zip(
            {
                "maria-export/expert.json": _manifest(),
                "maria-export/skills/research/SKILL.md": SKILL_MD,
            },
            directories=("maria-export/", "maria-export/skills/"),
        )
    )
    assert list(package.skills) == ["research"]


def test_the_body_cap_is_the_unpacked_cap_too():
    assert MAX_ZIP_BYTES == MAX_PACKAGE_BYTES


# ---------------------------------------------------------------------------
# What an expert package may never carry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "memory/facts.json",
        "conversations/2026-09-15.json",
        "workspace/report.xlsx",
        "skills/notes.txt",
        "README.md",
    ],
)
def test_an_entry_outside_the_format_is_refused_by_name(path: str):
    """Memory, chats and workspace files are structurally impossible to carry,
    and the refusal names the paths so the user can see what was dropped."""
    with pytest.raises(ExpertPackageError) as exc:
        package_from_zip(_minimal(**{path: b"x"}))
    assert path in str(exc.value)
    assert "not part of an expert package" in str(exc.value)
    assert not exc.value.over_limit


def test_only_the_first_offending_paths_are_named():
    extra = {f"memory/f{i}.json": b"x" for i in range(14)}
    with pytest.raises(ExpertPackageError) as exc:
        package_from_zip(_minimal(**extra))
    assert str(exc.value).count("memory/f") == 10
    assert "4 more" in str(exc.value)


_AVATAR_MANIFEST = _manifest(avatar={"kind": "file", "path": "avatar.png"})


@pytest.mark.parametrize(
    "archive, expected",
    [
        pytest.param(
            _zip({"skills/research/SKILL.md": SKILL_MD}),
            "no expert.json",
            id="no-manifest",
        ),
        pytest.param(
            _zip({"expert.json": _manifest(), "skills/research/refs/API.md": b"x"}),
            "no SKILL.md",
            id="skill-folder-without-a-skill-md",
        ),
        pytest.param(
            _minimal(**{"skills/research/../../escape.md": b"x"}),
            "unusable segment",
            id="skill-file-escaping-its-folder",
        ),
        pytest.param(
            _zip(
                {
                    "expert.json": _manifest(
                        skills=[
                            {"slug": "research", "name": "research"},
                            {"slug": "ghost", "name": "ghost"},
                        ]
                    ),
                    "skills/research/SKILL.md": SKILL_MD,
                }
            ),
            "carries no",
            id="manifest-skill-without-a-folder",
        ),
        pytest.param(
            _minimal(**{"skills/stowaway/SKILL.md": SKILL_MD}),
            "does not list",
            id="skill-folder-the-manifest-never-mentions",
        ),
        pytest.param(
            _zip(
                {
                    "expert.json": _AVATAR_MANIFEST,
                    "skills/research/SKILL.md": SKILL_MD,
                    "avatar.png": b"\x89PNG",
                    "avatar.gif": b"GIF89a",
                }
            ),
            "more than one avatar",
            id="two-avatars",
        ),
        pytest.param(
            _minimal(**{"avatar.bmp": b"BM"}),
            "avatar.bmp",
            id="avatar-in-an-unsupported-format",
        ),
        pytest.param(
            _minimal(**{"avatar.png": b"\x89PNG"}),
            "does not name",
            id="avatar-the-manifest-does-not-name",
        ),
        pytest.param(
            _zip(
                {"expert.json": _AVATAR_MANIFEST, "skills/research/SKILL.md": SKILL_MD}
            ),
            "does not carry",
            id="named-avatar-the-archive-does-not-hold",
        ),
        pytest.param(
            _zip(
                {
                    "expert.json": b"\xff\xfe binary",
                    "skills/research/SKILL.md": SKILL_MD,
                }
            ),
            "UTF-8",
            id="manifest-that-is-not-utf8",
        ),
        pytest.param(
            _zip({"expert.json": "{", "skills/research/SKILL.md": SKILL_MD}),
            "not a valid expert",
            id="manifest-that-is-not-json",
        ),
        pytest.param(
            b'{"format_version": 1}', "not a readable zip", id="not-a-zip-at-all"
        ),
    ],
)
def test_an_archive_that_is_not_an_expert_package_is_refused(
    archive: bytes, expected: str
):
    """Each of these would otherwise land as half an expert: a skill with no
    instructions, an avatar nobody asked for, a manifest nobody can read."""
    with pytest.raises(ExpertPackageError, match=expected) as exc:
        package_from_zip(archive)
    assert not exc.value.over_limit


def test_an_oversized_avatar_is_refused():
    with pytest.raises(ExpertPackageError) as exc:
        package_from_zip(
            _zip(
                {
                    "expert.json": _AVATAR_MANIFEST,
                    "skills/research/SKILL.md": SKILL_MD,
                    "avatar.png": b"\0" * (MAX_AVATAR_BYTES + 1),
                }
            )
        )
    assert exc.value.over_limit


def test_a_symlink_member_is_refused():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("expert.json", _manifest())
        link = zipfile.ZipInfo("skills/research/secrets")
        link.external_attr = 0o120777 << 16
        archive.writestr(link, "/etc/passwd")

    with pytest.raises(ExpertPackageError, match="symlink"):
        package_from_zip(buffer.getvalue())


def test_an_encrypted_member_is_refused():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("expert.json", _manifest())
        archive.writestr("skills/research/SKILL.md", SKILL_MD)

    with pytest.raises(ExpertPackageError, match="encrypted"):
        package_from_zip(_with_encryption_flag(buffer.getvalue()))


def test_a_bomb_is_refused_without_a_member_being_read(monkeypatch):
    """The declared sizes are the whole check: 22 MiB of zeros compresses to a
    few KB, so reading first is how a bomb lands in memory."""
    member = b"\0" * MAX_PACKAGE_FILE_BYTES
    data = _minimal(**{f"skills/research/f{i}.bin": member for i in range(11)})
    assert len(data) < MAX_PACKAGE_BYTES

    opened = MagicMock(side_effect=AssertionError("a member was read"))
    monkeypatch.setattr(zipfile.ZipFile, "open", opened)

    with pytest.raises(ExpertPackageError) as exc:
        package_from_zip(data)
    assert exc.value.over_limit
    opened.assert_not_called()


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
