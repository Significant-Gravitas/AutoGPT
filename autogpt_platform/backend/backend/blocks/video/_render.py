"""The video blocks' edits, done with moviepy in a child process.

moviepy encodes a video in one call that runs for about as long as the video
plays and never yields. On the executor's event loop that call would hold up
every node sharing the loop, and neither the block's time limit nor a stop
request could end it. So the blocks hand each edit to `run_edit` in `_utils`,
which runs this file in a child interpreter and kills it, with every ffmpeg it
started, when the block's time is up or its run is stopped.

Each edit takes the paths of its input files first, as a list. `run_edit`
hands it copies of its own, made before the child starts, so an edit strips
their chapters in place without touching a file another node reads, and a
node storing the same input meanwhile can't change it under the edit.

The child runs this file as a script, outside the blocks package, so it must
import nothing from `backend`: importing the package loads every block.
"""

import json
import logging
import os
import subprocess
import sys
from collections.abc import Callable
from typing import Any

from moviepy import CompositeAudioClip, concatenate_videoclips
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.fx import CrossFadeIn, CrossFadeOut, FadeIn, FadeOut
from moviepy.video.fx.Loop import Loop
from moviepy.video.io.ffmpeg_reader import ffmpeg_parse_infos
from moviepy.video.io.VideoFileClip import VideoFileClip

logger = logging.getLogger(__name__)

# Largest decoded frame (width * height) the video blocks will open: 4K UHD.
MAX_VIDEO_FRAME_PIXELS = 3840 * 2160


class EditRefused(ValueError):
    """An input an edit refuses, with the reason to show the user."""


def check_input_video(inputs: list[str]) -> None:
    """Strip chapters and check the frame size, without opening the video."""
    (video_path,) = inputs
    strip_chapters_inplace(video_path)
    check_frame_size(video_path)


def clip_video(
    inputs: list[str], output_path: str, start_time: float, end_time: float
) -> float:
    """Extract a segment from the video and return its duration."""
    (video_path,) = inputs
    clip = None
    subclip = None
    try:
        strip_chapters_inplace(video_path)
        clip = open_video(video_path)
        subclip = clip.subclipped(start_time, end_time)
        video_codec, audio_codec = get_video_codecs(output_path)
        subclip.write_videofile(
            output_path, codec=video_codec, audio_codec=audio_codec, logger=None
        )
        return subclip.duration
    finally:
        if subclip:
            subclip.close()
        if clip:
            clip.close()


def concat_videos(
    inputs: list[str],
    output_path: str,
    transition: str,
    transition_duration: int,
) -> float:
    """Concatenate the videos and return the total duration."""
    video_paths = inputs
    clips = []
    faded_clips = []
    final = None
    try:
        # Check every input before opening any of them
        for v in video_paths:
            strip_chapters_inplace(v)
            check_frame_size(v)

        # Load clips
        for v in video_paths:
            clips.append(open_video(v))

        # Validate transition_duration against shortest clip
        if transition in {"crossfade", "fade_black"} and transition_duration > 0:
            min_duration = min(c.duration for c in clips)
            if transition_duration >= min_duration:
                raise EditRefused(
                    f"transition_duration ({transition_duration}s) must be "
                    f"shorter than the shortest clip ({min_duration:.2f}s)"
                )

        if transition == "crossfade":
            for i, clip in enumerate(clips):
                effects = []
                if i > 0:
                    effects.append(CrossFadeIn(transition_duration))
                if i < len(clips) - 1:
                    effects.append(CrossFadeOut(transition_duration))
                if effects:
                    clip = clip.with_effects(effects)
                faded_clips.append(clip)
            final = concatenate_videoclips(
                faded_clips,
                method="compose",
                padding=-transition_duration,
            )
        elif transition == "fade_black":
            for clip in clips:
                faded = clip.with_effects(
                    [FadeIn(transition_duration), FadeOut(transition_duration)]
                )
                faded_clips.append(faded)
            final = concatenate_videoclips(faded_clips)
        else:
            final = concatenate_videoclips(clips)

        video_codec, audio_codec = get_video_codecs(output_path)
        final.write_videofile(
            output_path, codec=video_codec, audio_codec=audio_codec, logger=None
        )

        return final.duration
    finally:
        if final:
            final.close()
        for clip in faded_clips:
            clip.close()
        for clip in clips:
            clip.close()


def loop_video(
    inputs: list[str],
    output_path: str,
    duration: float | None,
    n_loops: int | None,
) -> None:
    """Loop the video until a duration, or a number of times."""
    (video_path,) = inputs
    strip_chapters_inplace(video_path)
    clip = None
    looped_clip = None
    try:
        clip = open_video(video_path)

        if duration:
            # Loop until we reach the specified duration
            looped_clip = clip.with_effects([Loop(duration=duration)])
        elif n_loops:
            looped_clip = clip.with_effects([Loop(n=n_loops)])
        else:
            raise EditRefused("Either 'duration' or 'n_loops' must be provided.")

        assert isinstance(looped_clip, VideoFileClip)
        looped_clip = looped_clip.with_audio(clip.audio)
        looped_clip.write_videofile(
            output_path, codec="libx264", audio_codec="aac", logger=None
        )
    finally:
        if looped_clip:
            looped_clip.close()
        if clip:
            clip.close()


def add_audio_to_video(inputs: list[str], output_path: str, volume: float) -> None:
    """Attach the audio file to the video as its audio track."""
    video_path, audio_path = inputs
    strip_chapters_inplace(video_path)
    strip_chapters_inplace(audio_path)
    video_clip = None
    audio_clip = None
    final_clip = None
    try:
        video_clip = open_video(video_path)
        audio_clip = AudioFileClip(audio_path)
        # Optionally scale volume
        if volume != 1.0:
            audio_clip = audio_clip.with_volume_scaled(volume)

        final_clip = video_clip.with_audio(audio_clip)
        final_clip.write_videofile(
            output_path, codec="libx264", audio_codec="aac", logger=None
        )
    finally:
        if final_clip:
            final_clip.close()
        if audio_clip:
            audio_clip.close()
        if video_clip:
            video_clip.close()


def add_narration_to_video(
    inputs: list[str],
    output_path: str,
    mix_mode: str,
    narration_volume: float,
    original_volume: float,
) -> None:
    """Add the narration audio to the video."""
    video_path, audio_path = inputs
    # Its own copy: the strip `check_input_video` did was on another one
    strip_chapters_inplace(video_path)
    video = None
    final = None
    narration_original = None
    narration_scaled = None
    original = None

    try:
        video = open_video(video_path)
        narration_original = AudioFileClip(audio_path)
        narration_scaled = narration_original.with_volume_scaled(narration_volume)
        narration = narration_scaled

        if mix_mode == "replace":
            final_audio = narration
        elif mix_mode == "mix":
            if video.audio:
                original = video.audio.with_volume_scaled(original_volume)
                final_audio = CompositeAudioClip([original, narration])
            else:
                final_audio = narration
        else:  # ducking - apply stronger attenuation
            if video.audio:
                # Ducking uses a much lower volume for original audio
                ducking_volume = original_volume * 0.3
                original = video.audio.with_volume_scaled(ducking_volume)
                final_audio = CompositeAudioClip([original, narration])
            else:
                final_audio = narration

        final = video.with_audio(final_audio)
        video_codec, audio_codec = get_video_codecs(output_path)
        final.write_videofile(
            output_path, codec=video_codec, audio_codec=audio_codec, logger=None
        )

    finally:
        if original:
            original.close()
        if narration_scaled:
            narration_scaled.close()
        if narration_original:
            narration_original.close()
        if final:
            final.close()
        if video:
            video.close()


def get_video_codecs(output_path: str) -> tuple[str, str]:
    """Get appropriate video and audio codecs based on output file extension.

    Args:
        output_path: Path to the output file (used to determine extension)

    Returns:
        Tuple of (video_codec, audio_codec)

    Codec mappings:
        - .mp4: H.264 + AAC (universal compatibility)
        - .webm: VP8 + Vorbis (web streaming)
        - .mkv: H.264 + AAC (container supports many codecs)
        - .mov: H.264 + AAC (Apple QuickTime, widely compatible)
        - .m4v: H.264 + AAC (Apple iTunes/devices)
        - .avi: MPEG-4 + MP3 (legacy Windows)
    """
    ext = os.path.splitext(output_path)[1].lower()

    codec_map: dict[str, tuple[str, str]] = {
        ".mp4": ("libx264", "aac"),
        ".webm": ("libvpx", "libvorbis"),
        ".mkv": ("libx264", "aac"),
        ".mov": ("libx264", "aac"),
        ".m4v": ("libx264", "aac"),
        ".avi": ("mpeg4", "libmp3lame"),
    }

    return codec_map.get(ext, ("libx264", "aac"))


def strip_chapters_inplace(video_path: str) -> None:
    """Strip chapter metadata from a media file in-place using ffmpeg.

    MoviePy 2.x crashes with IndexError when parsing files with embedded
    chapter metadata (https://github.com/Zulko/moviepy/issues/2419).
    This strips chapters without re-encoding.

    Args:
        video_path: Absolute path to the media file to strip chapters from.
    """
    base, ext = os.path.splitext(video_path)
    tmp_path = base + ".tmp" + ext
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-map_chapters",
                "-1",
                "-codec",
                "copy",
                tmp_path,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            logger.warning(
                "ffmpeg chapter strip failed (rc=%d): %s",
                result.returncode,
                result.stderr,
            )
            return
        os.replace(tmp_path, video_path)
    except FileNotFoundError:
        logger.warning("ffmpeg not found; skipping chapter strip")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def check_frame_size(video_path: str) -> None:
    """Refuse a video with any stream's frame (width x height, from the
    header) over MAX_VIDEO_FRAME_PIXELS."""
    infos = ffmpeg_parse_infos(video_path)
    sizes = [
        stream["size"]
        for input_file in infos.get("inputs", [])
        for stream in input_file.get("streams", [])
        if stream.get("stream_type") == "video" and stream.get("size")
    ]
    if not sizes:
        return
    width, height = max(sizes, key=lambda size: size[0] * size[1])
    if width * height > MAX_VIDEO_FRAME_PIXELS:
        raise EditRefused(
            f"Video frame size {width}x{height} is over the limit of "
            f"{MAX_VIDEO_FRAME_PIXELS:,} pixels per frame (width x height)"
        )


def open_video(video_path: str) -> VideoFileClip:
    """Check the frame size, then open the video. Every video block opens
    videos through this."""
    check_frame_size(video_path)
    return VideoFileClip(video_path)


EDITS: dict[str, Callable[..., Any]] = {
    edit.__name__: edit
    for edit in (
        check_input_video,
        clip_video,
        concat_videos,
        loop_video,
        add_audio_to_video,
        add_narration_to_video,
    )
}


def main() -> None:
    """Do one edit: its name and arguments come as JSON on stdin, and its
    result, refusal or error goes as JSON to the answer file it names."""
    request = json.load(sys.stdin)
    try:
        edit = EDITS[request["edit"]]
        answer = {"value": edit(*request["args"], **request["kwargs"])}
    except EditRefused as e:
        answer = {"refused": str(e)}
    except Exception as e:
        answer = {"error": str(e) or type(e).__name__}
    with open(request["answer_path"], "w", encoding="utf-8") as answer_file:
        # A duration can come back as a numpy number
        json.dump(answer, answer_file, default=float)


if __name__ == "__main__":
    main()
