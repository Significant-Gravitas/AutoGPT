"""
Backend-agnostic helpers shared across macOS / Windows / Linux backends.
"""

from __future__ import annotations

import base64
import io
from typing import Any

from ...protocol import DisplayMonitor

PASTE_THRESHOLD_CHARS = 200  # Q4 — paste:true only honored when text >= 200 chars.


def capture_fields(
    displays: list[DisplayMonitor],
    monitor: int,
    *,
    stitched: bool,
    virtual_desktop: dict[str, int],
    grabbed_width: int,
    image_width: int,
) -> tuple[int, float, tuple[int, int] | None]:
    """`monitor`, `display_scale` and `logical_size` for a screenshot.

    `display_scale` is the scale the capture actually came back at, image
    pixels per input unit, measured rather than taken from the display:
    whether mss returns a Retina display at its backing resolution depends on
    the mss version, and agents divide image points by this before clicking.
    The stitched desktop (`monitor: -1`) is no single display, so it reports
    its own logical size.
    """
    scale = round(image_width / grabbed_width, 2) if grabbed_width else 1.0
    if not stitched:
        target = displays[monitor]
        return target.index, scale, target.logical_size
    return -1, scale, (int(virtual_desktop["width"]), int(virtual_desktop["height"]))


def image_to_base64(img: Any, *, format: str = "jpeg", quality: int = 75) -> tuple[bytes, str]:
    """Encode a PIL.Image (or anything with .convert/.save) to bytes + mime.

    Centralised so per-OS backends don't duplicate Pillow handling.
    """
    fmt = format.lower()
    if fmt == "jpeg":
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
        return buf.getvalue(), "image/jpeg"
    if fmt == "png":
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue(), "image/png"
    raise ValueError(f"Unsupported image format: {format!r}")


def encode_b64(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def in_any_rect(
    coordinate: tuple[int, int],
    displays: list[dict[str, Any]],
) -> bool:
    """displays = [{"index": ..., "origin": [x,y], "size": [w,h]}, ...].

    A coord is in-bounds if it falls within ANY display's half-open
    [origin, origin+size) rectangle. Half-open because the bottom-right
    pixel of one display is the same coordinate as the top-left of the
    next stitched display — we want to count it exactly once.
    """
    x, y = coordinate
    for d in displays:
        ox, oy = d["origin"]
        w, h = d["size"]
        if ox <= x < ox + w and oy <= y < oy + h:
            return True
    return False


def normalize_displays_for_error(displays: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shape the display list for an INPUT_OUT_OF_BOUNDS error payload."""
    return [
        {
            "index": d.get("index", i),
            "origin": list(d["origin"]),
            "size": list(d["size"]),
        }
        for i, d in enumerate(displays)
    ]


__all__ = [
    "PASTE_THRESHOLD_CHARS",
    "encode_b64",
    "image_to_base64",
    "in_any_rect",
    "normalize_displays_for_error",
]
