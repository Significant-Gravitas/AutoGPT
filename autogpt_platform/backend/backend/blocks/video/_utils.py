"""Shared utilities for video blocks."""

from __future__ import annotations

import logging
import os
import subprocess

logger = logging.getLogger(__name__)


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
