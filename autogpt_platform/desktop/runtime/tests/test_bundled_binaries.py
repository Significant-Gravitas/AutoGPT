"""The executables the build takes from somebody else and ships: ffmpeg, from
the imageio-ffmpeg wheel, and the Prisma engines on Linux.

For each, the build has to know exactly what it is shipping: who built the
ffmpeg and under which licence it goes out, and that the Linux engines are
the ones linked against OpenSSL 3, whatever the build machine has installed.
The banners below are what the three real binaries print (imageio-ffmpeg
0.6.0, read on Windows, in WSL and on a Mac).
"""

import hashlib
import importlib
import struct
import sys
import urllib.error
from pathlib import Path

import pytest

DESKTOP = Path(__file__).resolve().parents[2]
# The build's modules are a script directory's, not a package's.
sys.path.insert(0, str(DESKTOP / "build"))
build_runtime = importlib.import_module("build_runtime")
bundled_tools = importlib.import_module("bundled_tools")
elf = importlib.import_module("elf")

GYAN = """ffmpeg version 7.1-essentials_build-www.gyan.dev Copyright (c) 2000-2024 the FFmpeg developers
built with gcc 14.2.0 (Rev1, Built by MSYS2 project)
configuration: --enable-gpl --enable-version3 --enable-static --enable-libx264 --enable-libmp3lame
libavutil      59. 39.100 / 59. 39.100
"""
VANSICKLE = """ffmpeg version 7.0.2-static https://johnvansickle.com/ffmpeg/  Copyright (c) 2000-2024 the FFmpeg developers
built with gcc 8 (Debian 8.3.0-6)
configuration: --enable-gpl --enable-version3 --enable-static --disable-debug --enable-libx264
"""
# No builder's name anywhere in it, and no --enable-version3.
OSXEXPERTS = """ffmpeg version 7.1 Copyright (c) 2000-2024 the FFmpeg developers
built with Apple clang version 13.1.6 (clang-1316.0.21.2.5)
configuration: --prefix=/Volumes/tempdisk/sw --extra-cflags=-fno-stack-check --arch=arm64 --cc=/usr/bin/clang --enable-gpl --enable-libvmaf --enable-libopenjpeg --enable-libopus --enable-libmp3lame --enable-libx264 --enable-libx265 --enable-libvpx --enable-libwebp --enable-libass --enable-libfreetype --enable-fontconfig --enable-libtheora --enable-libvorbis --enable-libsnappy --enable-libaom --enable-libvidstab --enable-libzimg --enable-libsvtav1 --enable-libharfbuzz --enable-libkvazaar --pkg-config-flags=--static --enable-ffplay --enable-postproc --enable-neon --enable-runtime-cpudetect --disable-indev=qtkit --disable-indev=x11grab_xcb
libavutil      59. 39.100 / 59. 39.100
"""
BANNERS = {"win32-x64": GYAN, "linux-x64": VANSICKLE, "darwin-arm64": OSXEXPERTS}


# --- ffmpeg: what it is --------------------------------------------------------


@pytest.mark.parametrize(
    ("configuration", "licence"),
    [
        ("--enable-gpl --enable-version3 --enable-libx264", "GPLv3"),
        ("--enable-gpl --enable-libx264", "GPLv2"),
        ("--enable-version3 --enable-shared", "LGPLv3"),
        ("--enable-shared --disable-static", "LGPLv2.1"),
    ],
)
def test_the_licence_shipped_is_the_one_the_build_is_under(
    configuration: str, licence: str
):
    assert bundled_tools.licence_of(configuration) == licence
    assert licence in bundled_tools.LICENCE_TEXTS


def test_a_build_that_may_not_be_redistributed_stops_the_bundle():
    with pytest.raises(bundled_tools.ToolError, match="may not be redistributed"):
        bundled_tools.licence_of("--enable-gpl --enable-nonfree --enable-libfdk-aac")


@pytest.mark.parametrize(
    ("platform", "builder", "source", "ffmpeg_source"),
    [
        ("win32-x64", "gyan.dev", "github.com/GyanD/codexffmpeg", "ffmpeg-7.1.tar.xz"),
        (
            "linux-x64",
            "johnvansickle.com",
            "johnvansickle.com/ffmpeg/release-source",
            "ffmpeg-7.0.2.tar.xz",
        ),
        ("darwin-arm64", "osxexperts.net", "osxexperts.net", "ffmpeg-7.1.tar.xz"),
    ],
)
def test_every_systems_note_names_its_builder_and_where_the_source_is(
    platform: str, builder: str, source: str, ffmpeg_source: str
):
    build = bundled_tools.describe(BANNERS[platform])
    note = bundled_tools.source_note(build, bundled_tools.ORIGINS[platform])

    built_by = next(line for line in note.splitlines() if line.startswith("Built by:"))
    assert builder in built_by
    assert f"https://ffmpeg.org/releases/{ffmpeg_source}" in note
    assert source in note.split("Corresponding source:")[1]
    assert bundled_tools.ORIGINS[platform].wheel_file in note
    assert "imageio-binaries" in note and "--enable-libx264" in note
    assert "not named" not in note


def test_the_systems_the_app_is_built_for_each_have_a_recorded_builder():
    assert set(bundled_tools.ORIGINS) == {"win32-x64", "linux-x64", "darwin-arm64"}
    with pytest.raises(bundled_tools.ToolError, match="add it to ORIGINS"):
        bundled_tools.origin_of("linux-arm64")


def test_a_binary_that_is_not_the_recorded_builders_stops_the_bundle():
    """The wheel changed where it gets a system's ffmpeg: nothing is written
    that names the old builder for the new binary."""
    for platform, banner in BANNERS.items():
        other = next(key for key in BANNERS if key != platform)
        with pytest.raises(bundled_tools.ToolError, match="find out who built it"):
            bundled_tools.source_note(
                bundled_tools.describe(banner), bundled_tools.ORIGINS[other]
            )
    with pytest.raises(bundled_tools.ToolError, match="nothing recognisable"):
        bundled_tools.describe("not ffmpeg")


def test_a_version_2_or_later_build_goes_out_under_version_3_with_both_texts():
    """GPL version 2 has no way to point at somebody else's server for the
    source; version 3 has (section 6d), and the build may be conveyed under
    it. The macOS build is the one this is about."""
    mac = bundled_tools.describe(OSXEXPERTS)
    assert (mac.licence, mac.conveyed_under) == ("GPLv2", "GPLv3")
    assert mac.licence_texts == ["GPLv2", "GPLv3"]
    note = bundled_tools.source_note(mac, bundled_tools.ORIGINS["darwin-arm64"])
    assert "GPLv2 or any later version; distributed with AutoGPT under GPLv3" in note
    assert "COPYING.GPLv3" in note and "COPYING.GPLv2" in note

    windows = bundled_tools.describe(GYAN)
    assert (windows.licence, windows.conveyed_under) == ("GPLv3", "GPLv3")
    assert windows.licence_texts == ["GPLv3"]


def test_the_encoders_the_backend_uses_must_be_in_the_build():
    listing = (
        "Encoders:\n V..... = Video\n ------\n"
        " V....D libx264              libx264 H.264\n"
        " A....D aac                  AAC (Advanced Audio Coding)\n"
    )
    assert bundled_tools.missing_encoders(listing) == ["libmp3lame"]
    assert bundled_tools.missing_encoders(listing + " A....D libmp3lame  MP3\n") == []


# --- ffmpeg: placing it, and refusing a bundle without its papers -------------


def test_the_wheels_one_binary_is_found_whatever_it_is_called(tmp_path: Path):
    binaries = tmp_path / "imageio_ffmpeg" / "binaries"
    binaries.mkdir(parents=True)
    (binaries / "README.md").write_text("Exes are dropped here by the release script.")
    assert bundled_tools.wheel_binary(tmp_path) is None
    (binaries / "ffmpeg-linux-x86_64-v7.0.2").write_bytes(b"")
    assert (
        bundled_tools.wheel_binary(tmp_path) == binaries / "ffmpeg-linux-x86_64-v7.0.2"
    )
    (binaries / "ffmpeg-macos-aarch64-v7.1").write_bytes(b"")
    with pytest.raises(bundled_tools.ToolError, match="expected one"):
        bundled_tools.wheel_binary(tmp_path)


def test_a_bundle_whose_wheel_has_no_ffmpeg_fails_the_build_with_where_to_look(
    tmp_path: Path,
):
    with pytest.raises(bundled_tools.ToolError, match="imageio-ffmpeg wheel"):
        bundled_tools.install_ffmpeg(tmp_path, tmp_path / "cache", "linux-x64")


def test_a_wheel_file_under_another_name_is_not_taken_for_the_known_build(
    tmp_path: Path,
):
    binaries = tmp_path / "site" / "imageio_ffmpeg" / "binaries"
    binaries.mkdir(parents=True)
    (binaries / "ffmpeg-linux-musl-x86_64-v8.0").write_bytes(b"")
    with pytest.raises(bundled_tools.ToolError, match="now carries ffmpeg-linux-musl"):
        bundled_tools.install_ffmpeg(tmp_path, tmp_path / "cache", "linux-x64")
    assert (binaries / "ffmpeg-linux-musl-x86_64-v8.0").exists()  # not moved


@pytest.fixture
def placed(tmp_path: Path, monkeypatch) -> Path:
    """A bundle as the `tools` step leaves it, for the macOS build; the
    binary and the licence downloads are stand-ins."""
    out = tmp_path / "runtime"
    binary = out / "tools" / "bin" / f"ffmpeg{bundled_tools.EXE}"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"")
    listing = "\n".join(f" A....D {name} x" for name in bundled_tools.REQUIRED_ENCODERS)
    answers = {"-version": OSXEXPERTS, "-encoders": listing}
    monkeypatch.setattr(
        bundled_tools, "_run", lambda ffmpeg, *arguments: answers[arguments[-1]]
    )
    monkeypatch.setattr(bundled_tools, "download", fake_download)
    monkeypatch.setattr(
        bundled_tools,
        "LICENCE_TEXTS",
        {name: fake_artifact(name) for name in bundled_tools.LICENCE_TEXTS},
    )
    bundled_tools.install_ffmpeg(out, tmp_path / "cache", "darwin-arm64")
    return out


def fake_artifact(licence: str):
    text = f"the text of {licence}".encode()
    return bundled_tools.Artifact(
        f"https://example.invalid/COPYING.{licence}", hashlib.sha256(text).hexdigest()
    )


def fake_download(artifact, cache: Path, name: str | None = None) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / (name or artifact.filename)
    target.write_bytes(
        f"the text of {artifact.filename.removeprefix('COPYING.')}".encode()
    )
    return target


def test_the_tools_step_leaves_the_licences_and_the_note_and_the_check_passes(
    placed: Path,
):
    directory = placed / "tools" / "licenses" / "ffmpeg"
    assert sorted(path.name for path in directory.iterdir()) == [
        "COPYING.GPLv2",
        "COPYING.GPLv3",
        "SOURCE.txt",
    ]
    assert "osxexperts.net" in (directory / "SOURCE.txt").read_text(encoding="utf-8")
    bundled_tools.check(placed, "darwin-arm64")


def test_a_bundle_with_ffmpeg_but_without_its_licence_or_note_is_not_sealed(
    placed: Path,
):
    directory = placed / "tools" / "licenses" / "ffmpeg"
    note = (directory / "SOURCE.txt").read_text(encoding="utf-8")

    (directory / "COPYING.GPLv3").unlink()
    with pytest.raises(bundled_tools.ToolError, match="run the tools step"):
        bundled_tools.check(placed, "darwin-arm64")
    (directory / "COPYING.GPLv3").write_bytes(b"the text of GPLv3")
    bundled_tools.check(placed, "darwin-arm64")

    (directory / "COPYING.GPLv3").write_bytes(b"some other text")
    with pytest.raises(bundled_tools.ToolError, match="run the tools step"):
        bundled_tools.check(placed, "darwin-arm64")
    (directory / "COPYING.GPLv3").write_bytes(b"the text of GPLv3")

    (directory / "SOURCE.txt").unlink()
    with pytest.raises(bundled_tools.ToolError, match="run the tools step"):
        bundled_tools.check(placed, "darwin-arm64")
    (directory / "SOURCE.txt").write_text(
        note.replace("7.1", "7.0"), encoding="utf-8", newline="\n"
    )
    with pytest.raises(bundled_tools.ToolError, match="run the tools step"):
        bundled_tools.check(placed, "darwin-arm64")


def test_a_bundle_that_still_has_the_wheels_copy_or_no_ffmpeg_is_not_sealed(
    placed: Path,
):
    binaries = placed / "site" / "imageio_ffmpeg" / "binaries"
    binaries.mkdir(parents=True)
    (binaries / "ffmpeg-macos-aarch64-v7.1").write_bytes(b"")
    with pytest.raises(bundled_tools.ToolError, match="second ffmpeg"):
        bundled_tools.check(placed, "darwin-arm64")
    (binaries / "ffmpeg-macos-aarch64-v7.1").unlink()

    (placed / "tools" / "bin" / f"ffmpeg{bundled_tools.EXE}").unlink()
    with pytest.raises(bundled_tools.ToolError, match="no tools/bin/ffmpeg"):
        bundled_tools.check(placed, "darwin-arm64")


def test_a_licence_download_that_fails_leaves_the_licence_that_was_there(
    placed: Path, monkeypatch
):
    def unreachable(artifact, cache: Path, name: str | None = None) -> Path:
        raise urllib.error.URLError("no network")

    monkeypatch.setattr(bundled_tools, "download", unreachable)
    with pytest.raises(urllib.error.URLError):
        bundled_tools.install_ffmpeg(placed, placed.parent / "cache", "darwin-arm64")

    bundled_tools.check(placed, "darwin-arm64")


# --- the Prisma engines of a Linux bundle --------------------------------------


def executable_needing(*libraries: str) -> bytes:
    """An ELF file with nothing in it but what is being read: a string
    table and a dynamic section that names `libraries`."""
    strings = b"\0" + b"".join(name.encode() + b"\0" for name in libraries)
    offsets, position = [], 1
    for name in libraries:
        offsets.append(position)
        position += len(name) + 1
    dynamic = b"".join(
        struct.pack("<qQ", 1, offset) for offset in offsets
    ) + struct.pack("<qQ", 0, 0)
    header_size, strings_at = 64, 64
    dynamic_at = strings_at + len(strings)
    sections_at = dynamic_at + len(dynamic)

    def section(kind: int, offset: int, size: int, link: int = 0) -> bytes:
        return struct.pack("<IIQQQQIIQQ", 0, kind, 0, 0, offset, size, link, 0, 8, 0)

    sections = (
        section(0, 0, 0)
        + section(3, strings_at, len(strings))
        + section(6, dynamic_at, len(dynamic), link=1)
    )
    header = b"\x7fELF" + bytes([2, 1, 1]) + bytes(9)
    header += struct.pack(
        "<HHIQQQIHHHHHH", 3, 62, 1, 0, 0, sections_at, 0, header_size, 0, 0, 64, 3, 1
    )
    assert len(header) == header_size
    return header + strings + dynamic + sections


def test_the_libraries_an_executable_asks_for_are_read_from_the_file():
    image = executable_needing("libssl.so.3", "libcrypto.so.3", "libc.so.6")
    assert elf.needed_in(image) == ["libssl.so.3", "libcrypto.so.3", "libc.so.6"]
    with pytest.raises(elf.NotElf):
        elf.needed_in(b"MZ" + bytes(100))


def test_only_an_engine_linked_against_openssl_3_passes():
    assert (
        build_runtime.wrong_openssl(["libssl.so.3", "libcrypto.so.3", "libc.so.6"])
        is None
    )
    old = build_runtime.wrong_openssl(
        elf.needed_in(executable_needing("libssl.so.1.1", "libc.so.6"))
    )
    assert "libssl.so.1.1" in str(old) and "libssl.so.3" in str(old)
    assert "no libssl" in str(build_runtime.wrong_openssl(["libc.so.6"]))


def test_a_linux_bundle_takes_the_openssl_3_engine_whatever_else_was_fetched(
    tmp_path: Path,
):
    """On a build machine with OpenSSL 1.1 and 3 the CLI's own choice is 1.1
    (GitHub's ubuntu-22.04); a cache may hold that one from an earlier build."""
    for name in (
        "query-engine-debian-openssl-1.1.x",
        "query-engine-debian-openssl-3.0.x",
        "libquery_engine-debian-openssl-1.1.x.so.node",
        "schema-engine-debian-openssl-1.1.x",
    ):
        (tmp_path / name).write_bytes(b"")

    found = build_runtime.fetched_engine(tmp_path, "query-engine", "linux-x64")
    assert found.name == "query-engine-debian-openssl-3.0.x"
    with pytest.raises(
        RuntimeError, match="did not fetch schema-engine-debian-openssl-3.0.x"
    ):
        build_runtime.fetched_engine(tmp_path, "schema-engine", "linux-x64")


def test_the_other_systems_take_the_one_engine_there_is(tmp_path: Path):
    (tmp_path / "query-engine-darwin-arm64").write_bytes(b"")
    (tmp_path / "libquery_engine-darwin-arm64.dylib.node").write_bytes(b"")
    assert (
        build_runtime.fetched_engine(tmp_path, "query-engine", "darwin-arm64").name
        == "query-engine-darwin-arm64"
    )


def test_the_prisma_cli_is_still_told_which_engines_to_fetch_through_that_variable():
    """Read in the CLI the bundle carries, when there is a bundle."""
    engines = (
        DESKTOP
        / "build"
        / "runtime"
        / "prisma"
        / "node_modules"
        / "@prisma"
        / "engines"
        / "dist"
    )
    if not engines.is_dir():
        pytest.skip("no assembled bundle to read the Prisma CLI from")
    code = "".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in engines.rglob("*.js")
    )
    assert "PRISMA_CLI_BINARY_TARGETS" in code, (
        "the bundled Prisma CLI no longer reads PRISMA_CLI_BINARY_TARGETS, which is how the "
        "build makes a Linux bundle carry the OpenSSL 3 engines (build_runtime.step_prisma)"
    )
