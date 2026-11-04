"""
Backend-agnostic helpers shared across macOS / Windows / Linux backends.
"""

from __future__ import annotations

import base64
import contextlib
import io
import logging
from collections.abc import Iterator, Sequence
from typing import Any

from ...protocol import DisplayMonitor
from ..errors import FeatureNotSupportedError

logger = logging.getLogger(__name__)

PASTE_THRESHOLD_CHARS = 200  # Q4 — paste:true only honored when text >= 200 chars.
# What pyautogui has a key for: printable ASCII, tab and the line breaks.
_TYPEABLE = frozenset(map(chr, range(0x20, 0x7F))) | {"\t", "\n", "\r"}


def _typeable(text: str) -> bool:
    return all(char in _TYPEABLE for char in text)


def chord_keys(key: str, aliases: dict[str, str], known: object = None) -> list[str]:
    """The keys of a `ctrl+s`-style chord, lowercased, with the documented
    modifier names (`super` and its alias `meta`) mapped to the OS's own.

    pyautogui skips a key name it doesn't know without a word, so `ctrl+bogus`
    would press Ctrl alone and still be acknowledged. A chord with a name not
    in `known` (pyautogui's KEY_NAMES) is refused before any key goes down.
    """
    parts = [part.strip().lower() for part in key.split("+")]
    if not all(parts):
        # Dropping the empty part would press another chord (ctrl++s as ctrl+s).
        raise FeatureNotSupportedError(
            "input.key", reason=f"the chord {key!r} has an empty key name"
        )
    keys = [aliases.get(k, k) for k in parts]
    if isinstance(known, (list, tuple, set, frozenset)):
        unknown = [k for k in keys if k not in known]
        if unknown:
            raise FeatureNotSupportedError(
                "input.key",
                reason=f"unknown key name {unknown[0]!r}; COMPUTER_USE.md lists the key names",
            )
    return keys


def display_index_for(displays: Sequence[DisplayMonitor], bounds: tuple[int, int, int, int]) -> int:
    """The index of the display holding the centre of `bounds` (left, top,
    right, bottom, in the displays' coordinates), else 0."""
    cx = (bounds[0] + bounds[2]) / 2
    cy = (bounds[1] + bounds[3]) / 2
    for display in displays:
        ox, oy = display.origin
        width, height = display.logical_size
        if ox <= cx < ox + width and oy <= cy < oy + height:
            return display.index
    return 0


def release_held(pyautogui: Any, *, keys: Sequence[str] = (), button: str | None = None) -> None:
    """Let go of held keys, last pressed first, and a held mouse button,
    however the input action ended.

    pyautogui's fail-safe raises from every call while the cursor is in a
    screen corner, which is where the user moves it to stop the automation.
    A release would raise before it let go, leaving a modifier or the button
    held down. Letting go starts nothing, so the check is off while it runs,
    and each release is tried even when one before it fails.
    """
    failsafe = pyautogui.FAILSAFE
    pyautogui.FAILSAFE = False
    try:
        for key in reversed(keys):
            try:
                pyautogui.keyUp(key)
            except Exception:
                logger.warning("Could not release the %r key", key, exc_info=True)
        if button is not None:
            try:
                pyautogui.mouseUp(button=button)
            except Exception:
                logger.warning("Could not release the %s mouse button", button, exc_info=True)
    finally:
        pyautogui.FAILSAFE = failsafe


@contextlib.contextmanager
def holding(pyautogui: Any, keys: Sequence[str]) -> Iterator[None]:
    """Hold `keys` down, in order, for the body, then let go of every one
    pressed, however the body or a press ended (release_held). A key counts
    as held only once its press went through."""
    held: list[str] = []
    try:
        for key in keys:
            pyautogui.keyDown(key)
            held.append(key)
        yield
    finally:
        release_held(pyautogui, keys=held)


def paste_is_worth_it(text: str) -> bool:
    """Q4: paste only text long enough to need it, or text with characters
    that can't be typed key by key."""
    return len(text) >= PASTE_THRESHOLD_CHARS or not _typeable(text)


def ensure_typeable(text: str) -> None:
    """pyautogui types printable ASCII, tabs and line breaks, and skips
    anything else (other scripts, NUL, ESC, DEL) without a word, so refuse
    rather than type part of the text."""
    if not _typeable(text):
        raise FeatureNotSupportedError(
            "type",
            reason="this text has characters that can't be typed key by key; send it "
            "with paste: true, which needs --enable-clipboard (and without "
            "preserve_clipboard if the clipboard can't be restored)",
        )


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
    "chord_keys",
    "ensure_typeable",
    "paste_is_worth_it",
    "encode_b64",
    "image_to_base64",
    "in_any_rect",
    "normalize_displays_for_error",
]
