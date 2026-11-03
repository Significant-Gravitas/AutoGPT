"""
Backend-agnostic helpers shared across macOS / Windows / Linux backends.
"""

from __future__ import annotations

import base64
import io
from typing import Any

PASTE_THRESHOLD_CHARS = 200  # Q4 — paste:true only honored when text >= 200 chars.


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
