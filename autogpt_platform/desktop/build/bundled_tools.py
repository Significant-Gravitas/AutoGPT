"""`<bundle>/tools/bin`: programs the backend runs by their plain name.

ffmpeg is the first. The backend's locked dependencies already bring one,
inside the imageio-ffmpeg wheel, under a name only that package knows
(`ffmpeg-win-x86_64-v7.1.exe`). The backend also runs `ffmpeg` by name
(blocks/video/_utils.py, the onboarding transcription) and so does yt-dlp for
the video download block, and none of them finds it there. The binary is
moved, not copied (it is 80 MB), to `tools/bin/ffmpeg`; the runtime puts that
directory on the services' PATH and names the file in IMAGEIO_FFMPEG_EXE,
which is where imageio-ffmpeg looks first (runtime settings.py).

The three wheels carry three different builds, by three builders (ORIGINS),
so nothing about the build is assumed: the binary is asked what it is, it
must be the build this file knows the origin of, the licence it is conveyed
under is shipped beside it with a note of where its source is, and a build
that may not be redistributed, or whose origin is not known, stops the bundle
being made.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from artifacts import Artifact, platform_key
from fetch import download

EXE = ".exe" if sys.platform == "win32" else ""
WHEEL = "imageio_ffmpeg"
# What the backend encodes with: H.264 video, and AAC or MP3 audio.
REQUIRED_ENCODERS = ("libx264", "aac", "libmp3lame")

# The licence texts as FFmpeg ships them; they do not change between
# releases, so one tag serves every build.
_COPYING = "https://raw.githubusercontent.com/FFmpeg/FFmpeg/n7.1/COPYING."
LICENCE_TEXTS = {
    "GPLv3": Artifact(
        f"{_COPYING}GPLv3",
        "8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903",
    ),
    "GPLv2": Artifact(
        f"{_COPYING}GPLv2",
        "8177f97513213526df2cf6184d8ff986c675afb514d4e68a404010521b880643",
    ),
    "LGPLv3": Artifact(
        f"{_COPYING}LGPLv3",
        "da7eabb7bafdf7d3ae5e9f223aa5bdc1eece45ac569dc21b3b037520b4464768",
    ),
    "LGPLv2.1": Artifact(
        f"{_COPYING}LGPLv2.1",
        "b634ab5640e258563c536e658cad87080553df6f34f62269a21d554844e58bfe",
    ),
}
# FFmpeg's licences are "version N or later". A build is conveyed under
# version 3, whose section 6(d) allows what the source note does: name the
# place the corresponding source is published. Version 2 has no such option.
CONVEYED_UNDER = {"GPLv2": "GPLv3", "LGPLv2.1": "LGPLv3"}
# Where imageio-ffmpeg's release script takes every binary from
# (its tasks.py), under the name the wheel keeps.
PUBLISHED_AT = "https://github.com/imageio/imageio-binaries/tree/master/ffmpeg"


@dataclass(frozen=True)
class Origin:
    """Who built the ffmpeg in one system's wheel."""

    wheel_file: str  # how the wheel names it, up to the version
    mark: str  # what that builder's `-version` output carries
    builder: str
    source: str


# The Windows and Linux builders sign their version string. The macOS build
# does not; imageio-ffmpeg took it from osxexperts.net (its pull request 114),
# whose builds are the ones configured for /Volumes/tempdisk/sw.
ORIGINS = {
    "win32-x64": Origin(
        "ffmpeg-win-x86_64-",
        "gyan.dev",
        "Gyan Doshi, https://www.gyan.dev/ffmpeg/builds/",
        "https://github.com/GyanD/codexffmpeg (each release lists the libraries and versions)",
    ),
    "linux-x64": Origin(
        "ffmpeg-linux-x86_64-",
        "johnvansickle.com",
        "John Van Sickle, https://johnvansickle.com/ffmpeg/",
        "https://johnvansickle.com/ffmpeg/release-source/ (ffmpeg and every library)",
    ),
    "darwin-arm64": Origin(
        "ffmpeg-macos-aarch64-",
        "--prefix=/Volumes/tempdisk/sw",
        "OSXExperts.NET, https://www.osxexperts.net/",
        "https://www.osxexperts.net/ (the build script and the libraries' versions)",
    ),
}


class ToolError(RuntimeError):
    pass


@dataclass(frozen=True)
class FfmpegBuild:
    version: str
    configuration: str
    banner: str  # what `ffmpeg -version` printed

    @property
    def licence(self) -> str:
        return licence_of(self.configuration)

    @property
    def conveyed_under(self) -> str:
        return CONVEYED_UNDER.get(self.licence, self.licence)

    @property
    def licence_texts(self) -> list[str]:
        """Shipped beside it: the licence it is conveyed under, and the one
        it was built under when that is an earlier version."""
        return sorted({self.licence, self.conveyed_under})


def install_ffmpeg(out: Path, cache: Path, platform: str | None = None) -> Path:
    """Put the wheel's ffmpeg at tools/bin/ffmpeg with its licence. Safe to
    run again on a bundle that already has it there."""
    origin = origin_of(platform or platform_key())
    target = out / "tools" / "bin" / f"ffmpeg{EXE}"
    source = wheel_binary(out / "site")
    if source:
        if not source.name.startswith(origin.wheel_file):
            raise ToolError(
                f"the imageio-ffmpeg wheel now carries {source.name}, not {origin.wheel_file}*. "
                "Find out who built it and update ORIGINS in desktop/build/bundled_tools.py."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.unlink(missing_ok=True)
        shutil.move(source, target)
        target.chmod(0o755)
    elif not target.is_file():
        raise ToolError(
            f"no ffmpeg in {out / 'site' / WHEEL / 'binaries'}. The bundle's ffmpeg is the one "
            "the imageio-ffmpeg wheel carries (a dependency of moviepy, in the backend's "
            "lock); if the backend dropped it or the wheel moved its binary, update "
            "desktop/build/bundled_tools.py. This step runs after `relocate`."
        )
    build = inspect(target)
    missing = missing_encoders(_run(target, "-hide_banner", "-encoders"))
    if missing:
        raise ToolError(
            f"the bundled ffmpeg ({build.version}) cannot encode {', '.join(missing)}"
        )
    _write_licence(licence_directory(out), build, origin, cache)
    return target


def check(out: Path, platform: str | None = None) -> None:
    """What a bundle about to be packaged must hold: the one ffmpeg, the
    licence texts that go with it, and the note of its source, as
    install_ffmpeg leaves them. A `tools` step that was cut short (it
    downloads the licence text) leaves the binary without them."""
    target = out / "tools" / "bin" / f"ffmpeg{EXE}"
    if not target.is_file():
        raise ToolError("the bundle has no tools/bin/ffmpeg; run the tools step")
    if wheel_binary(out / "site"):
        raise ToolError(
            "the bundle has a second ffmpeg, in the wheel; run the tools step"
        )
    build = inspect(target)
    directory = licence_directory(out)
    wanted = {
        f"COPYING.{licence}": LICENCE_TEXTS[licence].sha256
        for licence in build.licence_texts
    }
    found = {path.name: _sha256(path) for path in directory.glob("COPYING.*")}
    note = directory / "SOURCE.txt"
    expected = source_note(build, origin_of(platform or platform_key()))
    if (
        found != wanted
        or not note.is_file()
        or note.read_text(encoding="utf-8") != expected
    ):
        raise ToolError(
            f"{directory} does not hold the licence ({', '.join(wanted)}) and source note of "
            "the ffmpeg in tools/bin; run the tools step"
        )


def licence_directory(out: Path) -> Path:
    return out / "tools" / "licenses" / "ffmpeg"


def origin_of(platform: str) -> Origin:
    if platform not in ORIGINS:
        raise ToolError(
            f"nobody has recorded who builds the ffmpeg in imageio-ffmpeg's {platform} wheel; "
            "add it to ORIGINS in desktop/build/bundled_tools.py"
        )
    return ORIGINS[platform]


def wheel_binary(site: Path) -> Path | None:
    """The one ffmpeg the wheel installed, or None once it has been moved."""
    found = sorted(
        path for path in (site / WHEEL / "binaries").glob("ffmpeg-*") if path.is_file()
    )
    if len(found) > 1:
        raise ToolError(
            f"expected one ffmpeg in the imageio-ffmpeg wheel, found {found}"
        )
    return found[0] if found else None


def inspect(ffmpeg: Path) -> FfmpegBuild:
    return describe(_run(ffmpeg, "-version"))


def describe(banner: str) -> FfmpegBuild:
    version = re.search(r"^ffmpeg version (\S+)", banner, re.MULTILINE)
    configuration = re.search(r"^\s*configuration:(.*)$", banner, re.MULTILINE)
    if not version or not configuration:
        raise ToolError(
            f"`ffmpeg -version` printed nothing recognisable:\n{banner[:500]}"
        )
    return FfmpegBuild(version.group(1), configuration.group(1).strip(), banner)


def licence_of(configuration: str) -> str:
    """FFmpeg's own rule (its LICENSE.md): LGPL 2.1 or later as it comes,
    GPL with --enable-gpl, version 3 of either with --enable-version3, and
    not redistributable at all with --enable-nonfree."""
    flags = set(configuration.split())
    if "--enable-nonfree" in flags:
        raise ToolError(
            "the bundled ffmpeg was built with --enable-nonfree and may not be redistributed"
        )
    family = "GPL" if "--enable-gpl" in flags else "LGPL"
    if "--enable-version3" in flags:
        return f"{family}v3"
    return "GPLv2" if family == "GPL" else "LGPLv2.1"


def missing_encoders(listing: str) -> list[str]:
    """Of REQUIRED_ENCODERS, those `ffmpeg -encoders` does not list."""
    listed = {line.split()[1] for line in listing.splitlines() if len(line.split()) > 2}
    return [name for name in REQUIRED_ENCODERS if name not in listed]


def source_note(build: FfmpegBuild, origin: Origin) -> str:
    if origin.mark not in build.banner:
        raise ToolError(
            f"the bundled ffmpeg ({build.version}) does not say `{origin.mark}`, which the "
            f"builds of {origin.builder} do. The wheel has changed where it gets this "
            "system's ffmpeg: find out who built it and update ORIGINS in "
            "desktop/build/bundled_tools.py."
        )
    release = re.match(r"\d+(\.\d+)*", build.version)
    upstream = (
        f"https://ffmpeg.org/releases/ffmpeg-{release.group(0)}.tar.xz"
        if release
        else "https://ffmpeg.org/download.html (this build is not of a numbered release)"
    )
    if build.licence == build.conveyed_under:
        licence = f"{build.licence} (COPYING.{build.licence} in this directory)"
    else:
        licence = (
            f"{build.licence} or any later version; distributed with AutoGPT under "
            f"{build.conveyed_under}\n          (COPYING.{build.conveyed_under} in this "
            f"directory, beside COPYING.{build.licence})"
        )
    return "\n".join(
        [
            "ffmpeg, as distributed with AutoGPT",
            "",
            f"Version:  {build.version}",
            f"Licence:  {licence}",
            f"Built by: {origin.builder}",
            "",
            "This is the executable the imageio-ffmpeg Python package ships",
            f"({origin.wheel_file}* in {PUBLISHED_AT}),",
            "unchanged. It is statically linked; the libraries compiled into it are",
            "named in its configuration below.",
            "",
            "Corresponding source:",
            f"  ffmpeg itself:          {upstream}",
            f"  the build, its scripts and the libraries: {origin.source}",
            "",
            build.banner.strip(),
            "",
        ]
    )


def _write_licence(
    directory: Path, build: FfmpegBuild, origin: Origin, cache: Path
) -> None:
    """Everything is in hand before the old directory goes: a download that
    fails leaves what was there."""
    note = source_note(build, origin)
    texts = {
        licence: download(
            LICENCE_TEXTS[licence], cache, name=f"ffmpeg-COPYING.{licence}"
        )
        for licence in build.licence_texts
    }
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    for licence, text in texts.items():
        shutil.copyfile(text, directory / f"COPYING.{licence}")
    (directory / "SOURCE.txt").write_text(note, encoding="utf-8", newline="\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(ffmpeg: Path, *arguments: str) -> str:
    result = subprocess.run(
        [str(ffmpeg), *arguments],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise ToolError(
            f"{ffmpeg} {' '.join(arguments)} failed:\n{result.stderr[-1000:]}"
        )
    return result.stdout
