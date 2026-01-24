"""Shared utilities for video blocks."""

import os


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
