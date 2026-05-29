"""Shared utilities for video blocks."""

from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Known operation tags added by video blocks
_VIDEO_OPS = (
    r"(?:clip|overlay|narrated|looped|concat|audio_attached|with_audio|narration)"
)

# Matches: {node_exec_id}_{operation}_ where node_exec_id contains a UUID
_BLOCK_PREFIX_RE = re.compile(
    r"^[a-zA-Z0-9_-]*"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"[a-zA-Z0-9_-]*"
    r"_" + _VIDEO_OPS + r"_"
)

# Matches: a lone {node_exec_id}_ prefix (no operation keyword, e.g. download output)
_UUID_PREFIX_RE = re.compile(
    r"^[a-zA-Z0-9_-]*"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    r"[a-zA-Z0-9_-]*_"
)


def extract_source_name(input_path: str, max_length: int = 50) -> str:
    """Extract the original source filename by stripping block-generated prefixes.

    Iteratively removes {node_exec_id}_{operation}_ prefixes that accumulate
    when chaining video blocks, recovering the original human-readable name.

    Safe for plain filenames (no UUID -> no stripping).
    Falls back to "video" if everything is stripped.
    """
    stem = Path(input_path).stem

    # Pass 1: strip {node_exec_id}_{operation}_ prefixes iteratively
    while _BLOCK_PREFIX_RE.match(stem):
        stem = _BLOCK_PREFIX_RE.sub("", stem, count=1)

    # Pass 2: strip a lone {node_exec_id}_ prefix (e.g. from download block)
    if _UUID_PREFIX_RE.match(stem):
        stem = _UUID_PREFIX_RE.sub("", stem, count=1)

    if not stem:
        return "video"

    return stem[:max_length]


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
