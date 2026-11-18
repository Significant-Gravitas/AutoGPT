"""The edits `run_edit` hands to its child process, called here in-process on
tiny generated media, so what each of them makes is checked where coverage
can see it."""

import logging
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy
import pytest
from moviepy.video.io.ffmpeg_reader import ffmpeg_parse_infos

from backend.blocks.video import _render
from backend.blocks.video._render import EditRefused

ORIGINAL_HZ = 220  # the tone of a video's own sound
NARRATION_HZ = 440  # the tone of the audio added to it


@pytest.fixture(scope="module")
def media(tmp_path_factory) -> Path:
    media = tmp_path_factory.mktemp("media")
    _make_video(media / "silent.mp4", seconds=1)
    _make_video(media / "sound.mp4", seconds=1, sound_hz=ORIGINAL_HZ)
    _make_video(media / "two_seconds.mp4", seconds=2)
    tone = f"sine=frequency={NARRATION_HZ}:duration=1"
    _ffmpeg("-f", "lavfi", "-i", tone, media / "tone.mp3")
    return media


@pytest.fixture(autouse=True)
def _work_in_tmp_path(tmp_path: Path, monkeypatch) -> None:
    # moviepy writes an output's temporary audio track to the working directory
    monkeypatch.chdir(tmp_path)


def test_clip_video_cuts_the_segment_and_returns_its_duration(media, tmp_path):
    out = tmp_path / "out.mp4"

    duration = _render.clip_video(
        _copies(media, tmp_path, "silent.mp4"), str(out), 0.2, 0.7
    )

    assert duration == pytest.approx(0.5)
    assert _duration(out) == pytest.approx(0.5, abs=0.15)


@pytest.mark.parametrize(
    "transition, total", [("none", 4), ("crossfade", 3), ("fade_black", 4)]
)
def test_concat_videos_joins_with_each_transition(media, tmp_path, transition, total):
    out = tmp_path / "out.mp4"

    duration = _render.concat_videos(
        _copies(media, tmp_path, "two_seconds.mp4", "two_seconds.mp4"),
        str(out),
        transition,
        1,
    )

    # A crossfade overlaps the clips by the transition's length
    assert duration == pytest.approx(total)
    assert _duration(out) == pytest.approx(total, abs=0.15)


def test_concat_videos_refuses_a_transition_as_long_as_a_clip(media, tmp_path):
    with pytest.raises(EditRefused, match="must be shorter than the shortest clip"):
        _render.concat_videos(
            _copies(media, tmp_path, "two_seconds.mp4", "silent.mp4"),
            str(tmp_path / "out.mp4"),
            "crossfade",
            1,
        )


@pytest.mark.parametrize(
    "duration, n_loops, total", [(2.5, None, 2.5), (None, 3, 3)], ids=["to", "times"]
)
def test_loop_video_loops_to_a_duration_or_a_number_of_times(
    media, tmp_path, duration, n_loops, total
):
    out = tmp_path / "out.mp4"

    _render.loop_video(
        _copies(media, tmp_path, "silent.mp4"), str(out), duration, n_loops
    )

    assert _duration(out) == pytest.approx(total, abs=0.15)


def test_loop_video_refuses_neither_a_duration_nor_a_count(media, tmp_path):
    inputs = _copies(media, tmp_path, "silent.mp4")

    with pytest.raises(EditRefused, match="Either 'duration' or 'n_loops'"):
        _render.loop_video(inputs, str(tmp_path / "out.mp4"), None, None)


def test_add_audio_to_video_gives_it_the_audio_at_its_volume(media, tmp_path):
    levels = {}
    for volume in (1.0, 0.5):
        out = tmp_path / f"{volume}.mp4"
        _render.add_audio_to_video(
            _copies(media, tmp_path, "silent.mp4", "tone.mp3"), str(out), volume
        )
        levels[volume] = _tone_levels(out)[NARRATION_HZ]

    assert levels[1.0] > 0
    assert levels[0.5] / levels[1.0] == pytest.approx(0.5, rel=0.15)


def test_add_narration_to_video_mixes_by_mode(media, tmp_path):
    original_left = {}
    for mix_mode in ("replace", "mix", "ducking"):
        out = tmp_path / f"{mix_mode}.mp4"
        _render.add_narration_to_video(
            _copies(media, tmp_path, "sound.mp4", "tone.mp3"),
            str(out),
            mix_mode,
            1.0,
            0.3,
        )
        levels = _tone_levels(out)
        original_left[mix_mode] = levels[ORIGINAL_HZ] / levels[NARRATION_HZ]

    # Replaced, kept at original_volume, or ducked to a further 0.3 of that
    assert original_left["replace"] < 0.02
    assert original_left["mix"] == pytest.approx(0.3, rel=0.25)
    assert original_left["ducking"] == pytest.approx(0.09, rel=0.25)


@pytest.mark.parametrize("mix_mode", ["mix", "ducking"])
def test_add_narration_to_video_narrates_a_silent_video(media, tmp_path, mix_mode):
    out = tmp_path / "out.mp4"

    _render.add_narration_to_video(
        _copies(media, tmp_path, "silent.mp4", "tone.mp3"), str(out), mix_mode, 1.0, 0.3
    )

    assert _tone_levels(out)[NARRATION_HZ] > 0


def test_check_input_video_strips_chapters_so_the_video_can_be_read(
    chaptered_video, ffmpeg_on_path
):
    _render.check_input_video([str(chaptered_video)])

    infos = ffmpeg_parse_infos(str(chaptered_video))
    assert infos["video_found"]
    assert not infos["inputs"][0].get("chapters")


def test_strip_chapters_leaves_a_file_ffmpeg_cant_read_as_it_was(
    tmp_path, ffmpeg_on_path, caplog
):
    not_a_video = tmp_path / "not_a_video.mp4"
    not_a_video.write_bytes(b"not a video")

    with caplog.at_level(logging.WARNING, logger=_render.__name__):
        _render.strip_chapters_inplace(str(not_a_video))

    assert not_a_video.read_bytes() == b"not a video"
    assert list(tmp_path.iterdir()) == [not_a_video]
    assert "ffmpeg chapter strip failed" in caplog.text


def test_strip_chapters_without_ffmpeg_leaves_the_file_as_it_was(
    chaptered_video, tmp_path, monkeypatch, caplog
):
    stored = chaptered_video.read_bytes()
    monkeypatch.setenv("PATH", str(tmp_path / "no_ffmpeg_here"))

    with caplog.at_level(logging.WARNING, logger=_render.__name__):
        _render.strip_chapters_inplace(str(chaptered_video))

    assert chaptered_video.read_bytes() == stored
    assert "ffmpeg not found" in caplog.text


def test_check_frame_size_lets_a_file_without_video_through(media):
    _render.check_frame_size(str(media / "tone.mp3"))


def _copies(media: Path, tmp_path: Path, *names: str) -> list[str]:
    """Copies of the inputs, as `run_edit` makes them: the edits strip
    chapters from their inputs in place."""
    copies = []
    for i, name in enumerate(names):
        copies.append(str(tmp_path / f"{i}-{name}"))
        shutil.copyfile(media / name, copies[-1])
    return copies


def _duration(path: Path) -> float:
    return ffmpeg_parse_infos(str(path))["duration"]


def _tone_levels(path: Path) -> dict[int, float]:
    """How loud each test tone is in the first second of a file's sound."""
    pcm = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(path), "-vn", "-ac", "1"]
        + ["-ar", "8000", "-f", "s16le", "-"],
        check=True,
        capture_output=True,
        timeout=120,
    ).stdout
    # A second at 8 kHz, so each tone falls on a 1 Hz bin of its own
    samples = numpy.frombuffer(pcm, dtype=numpy.int16)[:8000] / 32768
    spectrum = numpy.abs(numpy.fft.rfft(samples, n=8000))
    return {
        hz: float(numpy.sqrt((spectrum[hz - 3 : hz + 4] ** 2).sum()))
        for hz in (ORIGINAL_HZ, NARRATION_HZ)
    }


def _make_video(path: Path, seconds: int, sound_hz: int | None = None) -> None:
    """A black 16x16 video at 10 frames a second, with a tone if `sound_hz`."""
    args = ["-f", "lavfi", "-i", f"color=c=black:s=16x16:r=10:d={seconds}"]
    if sound_hz:
        args += ["-f", "lavfi", "-i", f"sine=frequency={sound_hz}:duration={seconds}"]
    args += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if sound_hz:
        args += ["-c:a", "aac", "-shortest"]
    _ffmpeg(*args, path)


def _ffmpeg(*args: str | Path) -> None:
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", *map(str, args)],
        check=True,
        capture_output=True,
        timeout=120,
    )
