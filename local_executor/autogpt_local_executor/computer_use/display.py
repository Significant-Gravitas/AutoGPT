"""
Cross-OS display enumeration.

Backends each ship their own preferred enumerator but all fall back to
`mss` which has decent multi-monitor support. Callers use these helpers
to (a) populate DISPLAY_INFO_RESPONSE and (b) compute the union of
display rects for the INPUT_OUT_OF_BOUNDS check (Q1).

Note: mss gives us **virtual** pixel rects in display-global coordinates
(top-left origin) — exactly what Q1 mandates. Per-OS HiDPI scale is
read from native APIs in the per-OS backends and merged in.
"""

from __future__ import annotations

from typing import Any

from ..protocol import DisplayMonitor


def displays_via_mss() -> list[DisplayMonitor] | None:
    """Return DisplayMonitors via the mss library, or None if unavailable.

    mss.monitors is a list where index 0 is the "stitched" virtual
    desktop and indexes 1..N are individual monitors. We expose 0..N-1
    in our normalized list (so `index` matches `mss` monitor index - 1).
    """
    try:
        import mss  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        with mss.mss() as sct:
            raw = list(sct.monitors[1:])  # skip the stitched virtual desktop
            if not raw:
                return None
            out: list[DisplayMonitor] = []
            for i, m in enumerate(raw):
                width = int(m["width"])
                height = int(m["height"])
                left = int(m["left"])
                top = int(m["top"])
                out.append(
                    DisplayMonitor(
                        index=i,
                        primary=(i == 0),
                        physical_size=(width, height),
                        logical_size=(width, height),
                        scale=1.0,
                        origin=(left, top),
                    )
                )
            return out
    except Exception:
        return None


def displays_to_bounds_list(displays: list[DisplayMonitor]) -> list[dict[str, Any]]:
    """Shape for in_any_rect / normalize_displays_for_error."""
    return [
        {
            "index": d.index,
            "origin": (d.origin[0], d.origin[1]),
            "size": (d.physical_size[0], d.physical_size[1]),
        }
        for d in displays
    ]


__all__ = ["displays_to_bounds_list", "displays_via_mss"]
