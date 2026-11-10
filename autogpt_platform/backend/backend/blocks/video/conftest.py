import os
import subprocess
from pathlib import Path

import imageio_ffmpeg
import pytest
from moviepy import ColorClip


@pytest.fixture
def ffmpeg_on_path(tmp_path_factory, monkeypatch) -> None:
    """`ffmpeg` on PATH, as the backend's image installs it. The edits strip
    chapters with the `ffmpeg` they find there; moviepy's own copy, which they
    open videos with, isn't on PATH."""
    ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
    bin_dir = tmp_path_factory.mktemp("bin")
    (bin_dir / f"ffmpeg{ffmpeg.suffix}").symlink_to(ffmpeg)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")


@pytest.fixture
def chaptered_video(tmp_path: Path) -> Path:
    """A one-second video with a single chapter, which moviepy can't read
    until its chapters are stripped (https://github.com/Zulko/moviepy/issues/2419)."""
    plain = tmp_path / "plain.mp4"
    ColorClip((16, 16), color=(0, 0, 0), duration=1).write_videofile(
        str(plain), fps=10, logger=None
    )
    chapters = tmp_path / "chapters.txt"
    chapters.write_text(
        ";FFMETADATA1\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=1000\ntitle=Intro\n"
    )
    video = tmp_path / "chaptered.mp4"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-i",
            str(plain),
            "-i",
            str(chapters),
            "-map_metadata",
            "1",
            "-map_chapters",
            "1",
            "-codec",
            "copy",
            str(video),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    plain.unlink()
    return video
