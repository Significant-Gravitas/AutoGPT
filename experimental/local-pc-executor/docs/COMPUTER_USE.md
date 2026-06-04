# Local PC Executor — Computer-Use Feature Spec

> **Status**: Draft v0.1 — additive to [PROTOCOL.md](PROTOCOL.md). All ops in this
> document gate on the `computer_use` capability being granted.

## Overview

"Computer use" in the shim means: the platform LLM (Anthropic
`computer_20251124`, OpenAI computer-use tool, or anything that emits the
same family of atomic GUI actions) emits high-level tool calls; the platform
translates them to the wire ops below; the shim runs them on the host with
the user's full OS-level identity (their cookies, their SSH keys, their
locally-installed apps) and returns observable results. Every op writes an
audit record per [AUDIT_LOG.md](AUDIT_LOG.md) so the user can post-hoc
inspect what the agent did. Today's surface — `SCREENSHOT_REQUEST` +
seven `INPUT_ACTION` verbs — is enough for a screen-and-click loop but
falls short of "fill a form across three apps, drag a file from Finder into
a chat window, take a meeting note from a video call." This document
specifies what we add to close that gap.

The v1 surface is deliberately **op-shaped, not macro-shaped**: each wire op
is one observable host action with one audit record and one error mode. The
platform composes them into multi-step agent loops; the shim does not own
the loop.

---

## v1 wire-op surface

These are additive to PROTOCOL.md §Computer Use. The transport, envelope,
correlation, error shape, and `CAPABILITY_NOT_GRANTED` semantics are
unchanged — only the message types and fields below are new or extended.

### Additions to `HELLO.capabilities`

The `computer_use` capability stays a single flag (shim either advertises
it or doesn't, per Wayland / TCC rules in [CROSS_PLATFORM.md](CROSS_PLATFORM.md)).
Inside `HELLO.payload`, add a new field:

```json
{
  "computer_use_features": [
    "screenshot.region",
    "screenshot.window",
    "input.click.modifiers",
    "input.click.button",
    "input.drag.path",
    "input.key.hold",
    "input.mouse.down_up",
    "input.scroll.amount",
    "input.wait",
    "cursor.position",
    "display.info",
    "window.list",
    "window.focus",
    "app.list",
    "app.launch",
    "clipboard.read",
    "clipboard.write",
    "permissions.check"
  ]
}
```

Rationale: cua-driver's PARITY.md shows per-OS gaps even for VERIFIED tools
(e.g. Windows has `drag`, macOS doesn't yet). The platform needs to know
*per-shim* what works without sniffing the OS string. Each feature flag
corresponds to one v1 op or one extension to an existing op; see the per-OS
matrix below for what each shim advertises.

The platform must not call any feature not in `computer_use_features`; the
shim returns `FEATURE_NOT_SUPPORTED` (new error code) if it does.

---

### Extended: `SCREENSHOT_REQUEST`

Backwards-compatible additions to `payload`:

```json
{
  "type": "SCREENSHOT_REQUEST",
  "id": "req-uuid",
  "ts": 1712345678.0,
  "payload": {
    "monitor": 0,
    "quality": 75,
    "region": [100, 200, 900, 700],   // optional [x1, y1, x2, y2] — Anthropic zoom-equivalent
    "window_id": null,                 // optional opaque ID from WINDOW_LIST_RESPONSE
    "format": "jpeg",                  // "jpeg" | "png" — png for OCR/diff workflows
    "include_cursor": true             // overlay cursor in the capture
  }
}
```

`SCREENSHOT_RESPONSE` gains a `region` echo and a `display_scale` field so
the platform can map model-space coordinates back to physical pixels on
HiDPI displays (this is the same trap that bit the cua-driver Linux build —
hardcoded `scale_factor: 1.0`):

```json
{
  "payload": {
    "image_base64": "...",
    "mime_type": "image/jpeg",
    "width": 800,
    "height": 500,
    "monitor": 0,
    "region": [100, 200, 900, 700],
    "display_scale": 2.0,
    "logical_size": [1440, 900]
  }
}
```

Semantics:
- `region` and `window_id` are mutually exclusive.
- `window_id` is opaque — the platform got it from `WINDOW_LIST_RESPONSE`
  and passes it back as-is. The shim is free to compose pid+window-handle
  or AX-ref inside.
- If a window is occluded or off-screen, the shim returns the image of the
  window's bounds anyway (the OS reports cached bitmap on macOS/Windows;
  on Linux X11 we surface `WINDOW_NOT_VISIBLE` because XComposite isn't
  guaranteed).

### Extended: `INPUT_ACTION`

The existing seven actions stay. We expand the action vocabulary and add
modifier-key + button + duration support so the surface matches Anthropic's
`computer_20251124` and OpenAI's computer-use action set without forcing
the platform to chunk a single agent intent across multiple wire ops:

```json
{
  "type": "INPUT_ACTION",
  "id": "req-uuid",
  "ts": 1712345678.0,
  "payload": {
    "action": "left_click",
    "coordinate": [500, 300],

    // New: button selector (default = action-implied)
    "button": "left",        // "left" | "middle" | "right"

    // New: modifiers held during click/scroll (mirrors Anthropic's "text" field)
    "modifiers": ["shift"],  // subset of: "shift" | "ctrl" | "alt" | "super"

    // New: scroll amount instead of clicks (matches model semantics)
    "scroll_amount": 3,
    "scroll_direction": "down",

    // New: hold duration for hold_key
    "duration_ms": 1500,

    // Existing
    "text": null,
    "key": null,
    "direction": null,
    "clicks": null
  }
}
```

New action verbs (additive — old ones keep working):

| Action | Required fields | Notes |
|---|---|---|
| `middle_click` | `coordinate` | Same shape as left/right. |
| `triple_click` | `coordinate` | One wire op, not three (the OS distinguishes triple-click for word/line select). |
| `mouse_down` | `coordinate`, `button` | Press without release; pairs with `mouse_up`. |
| `mouse_up` | `coordinate`, `button` | Release. The pair lets the platform build drags the shim hasn't anticipated (rubber-band select, drawing). |
| `drag` | `path: [[x,y], ...]`, `button` | Single-op drag along a polyline. The shim does press → move-through-each-point → release as one host gesture. Replaces the platform having to thread `mouse_down`/`mouse_move`/`mouse_up`. |
| `hold_key` | `key`, `duration_ms` | Press a key (or chord) and hold for `duration_ms`. Distinct from `key`, which presses-and-releases. |
| `wait` | `duration_ms` | Block the request for `duration_ms` then ACK. Lets the model insert a settling pause (animations, page loads) without round-tripping a no-op screenshot. Capped at 5000 ms server-side. |

The `click`/`scroll` actions get `modifiers` (Shift+click for range-select,
Cmd+click for tab-open, Shift+scroll for horizontal). This is exactly how
Anthropic's tool surfaces it in `computer_20251124`: a `text` field on
click/scroll holding the modifier name. We call ours `modifiers` because
"text" is overloaded with the `type` action's text.

#### New action: `key` extension

The existing `key` action accepts `ctrl+s`-style chords. Two clarifications:
- Modifier vocabulary: `shift`, `ctrl`, `alt`, `super`, `meta` (alias for
  super on POSIX, win on Windows). The shim normalizes per-OS.
- Single keys: `enter`, `tab`, `esc`, `space`, `up`, `down`, `left`,
  `right`, `pageup`, `pagedown`, `home`, `end`, `backspace`, `delete`,
  `f1`-`f24`. Lowercase ASCII letters and digits are accepted as-is.

#### `type` action — input method

Today `_pyautogui.write` is character-by-character. We add an optional
`paste: true` field on the `type` action that the shim implements by
stashing the text on the OS clipboard, sending Cmd/Ctrl+V, and restoring
the previous clipboard. Why: pasting a 4 KB form value char-by-char is
~80 seconds at default 20 ms interval and trips IME/autocomplete on web
forms. Trade-off: clipboard is observable to other apps for ~50 ms;
documented limitation.

```json
{
  "action": "type",
  "text": "long form value...",
  "paste": true
}
```

---

### New: `CURSOR_POSITION_REQUEST` / `CURSOR_POSITION_RESPONSE`

Cheap. Lets the model verify a `mouse_move` landed without paying a
screenshot's bytes.

```json
{
  "type": "CURSOR_POSITION_REQUEST",
  "id": "req-uuid",
  "ts": 1712345678.0,
  "payload": {}
}
```

```json
{
  "type": "CURSOR_POSITION_RESPONSE",
  "id": "req-uuid",
  "ts": 1712345678.0,
  "payload": {
    "x": 512,
    "y": 384,
    "monitor": 0
  }
}
```

---

### New: `DISPLAY_INFO_REQUEST` / `DISPLAY_INFO_RESPONSE`

Replaces the platform having to read `screen_resolution` from `HELLO` and
hope it's still correct (users plug in monitors, rotate iPads as displays,
switch Spaces). Also surfaces per-monitor scale, which `HELLO` doesn't.

```json
{
  "type": "DISPLAY_INFO_REQUEST",
  "id": "req-uuid",
  "ts": 1712345678.0,
  "payload": {}
}
```

```json
{
  "type": "DISPLAY_INFO_RESPONSE",
  "id": "req-uuid",
  "ts": 1712345678.0,
  "payload": {
    "monitors": [
      {
        "index": 0,
        "primary": true,
        "physical_size": [3840, 2160],
        "logical_size": [1920, 1080],
        "scale": 2.0,
        "origin": [0, 0]
      },
      {
        "index": 1,
        "primary": false,
        "physical_size": [2560, 1440],
        "logical_size": [2560, 1440],
        "scale": 1.0,
        "origin": [1920, 0]
      }
    ]
  }
}
```

---

### New: `WINDOW_LIST_REQUEST` / `WINDOW_LIST_RESPONSE`

A multi-app workflow ("copy this row from the spreadsheet into the email
draft") needs to know what windows exist. Per cua-driver's PARITY.md,
`list_windows` is VERIFIED on Windows and "code ready / audit pending" on
the others — same posture for us.

```json
{
  "type": "WINDOW_LIST_REQUEST",
  "id": "req-uuid",
  "ts": 1712345678.0,
  "payload": {
    "app_bundle_id": null,        // optional: filter by app
    "include_minimized": false,
    "include_offscreen": false
  }
}
```

```json
{
  "type": "WINDOW_LIST_RESPONSE",
  "id": "req-uuid",
  "ts": 1712345678.0,
  "payload": {
    "windows": [
      {
        "window_id": "opaque-shim-token",
        "pid": 12345,
        "app_name": "Safari",
        "app_bundle_id": "com.apple.Safari",
        "title": "AutoGPT — Dashboard",
        "bounds": [100, 100, 1820, 1080],
        "monitor": 0,
        "is_focused": true,
        "is_minimized": false,
        "is_fullscreen": false
      }
    ],
    "truncated": false
  }
}
```

`window_id` is an opaque string the shim issues; treat it like a session
handle, not a global identifier. It MAY change across reconnects.

### New: `WINDOW_FOCUS`

```json
{
  "type": "WINDOW_FOCUS",
  "id": "req-uuid",
  "ts": 1712345678.0,
  "payload": {
    "window_id": "opaque-shim-token",
    "raise": true                 // bring to front; false = focus without raising (X11 only)
  }
}
```

Responds with `ACK`. On macOS this is `AXRaise` on the AXUIElementRef; on
Windows, `SetForegroundWindow` (best-effort per Win32's foreground-lock
rules — cua-driver flagged this as a known caveat); on Linux X11,
`XRaiseWindow` + `_NET_ACTIVE_WINDOW`.

### Deliberately omitted from v1

- `get_window_state` — covered by `WINDOW_LIST_RESPONSE` already.
- `move_window` / `resize_window` — wait for v2 once we see real demand.
- `close_window` — too destructive for v1; the agent can Cmd+W via `key`.

---

### New: `APP_LIST_REQUEST` / `APP_LIST_RESPONSE`

Lists what's *running*, not what's installed. Cheaper and more useful for
"is Slack open already?".

```json
{
  "type": "APP_LIST_REQUEST",
  "id": "req-uuid",
  "ts": 1712345678.0,
  "payload": { "include_background": false }
}
```

```json
{
  "type": "APP_LIST_RESPONSE",
  "id": "req-uuid",
  "ts": 1712345678.0,
  "payload": {
    "apps": [
      {
        "pid": 12345,
        "name": "Safari",
        "bundle_id": "com.apple.Safari",
        "executable_path": "/Applications/Safari.app/Contents/MacOS/Safari",
        "is_frontmost": true,
        "window_count": 3
      }
    ]
  }
}
```

### New: `APP_LAUNCH`

```json
{
  "type": "APP_LAUNCH",
  "id": "req-uuid",
  "ts": 1712345678.0,
  "payload": {
    "bundle_id": "com.apple.Safari",         // macOS preferred
    "executable_path": null,                  // Windows/Linux preferred — full path
    "args": [],
    "activate": true                          // bring to front after launch
  }
}
```

Responds with `ACK` containing the launched pid, e.g.
`{ "ok": true, "pid": 67890 }`. The shim uses, in order of preference:
- macOS: `NSWorkspace.openApplication` via PyObjC (no shell, no focus
  steal during background launches — matches cua-driver's design).
- Windows: `ShellExecute` via `pywin32`.
- Linux: `subprocess.Popen` with `start_new_session=True` and `XDG_OPEN`
  fallback if `executable_path` is a `.desktop` file.

We deliberately do **not** fall through to `EXECUTE_COMMAND` — the audit
log distinguishes "agent launched an app" from "agent ran a shell command",
and conflating them costs us reviewability.

### Deliberately omitted from v1

- `list_apps` for installed-but-not-running apps (`/Applications` scan,
  `Start Menu` enumeration). Big per-OS surface for a niche need; defer.
- AppleScript / DDE / D-Bus app control. v2.

---

### New: `CLIPBOARD_READ` / `CLIPBOARD_WRITE`

Real workflows need to move data between apps without re-typing it. A
multi-app workflow is half-blind without clipboard access.

```json
{
  "type": "CLIPBOARD_READ",
  "id": "req-uuid",
  "ts": 1712345678.0,
  "payload": {
    "format": "text"                 // "text" | "image" — v1 only
  }
}
```

```json
{
  "type": "CLIPBOARD_READ_RESPONSE",
  "id": "req-uuid",
  "ts": 1712345678.0,
  "payload": {
    "format": "text",
    "content": "hello",
    "size_bytes": 5
  }
}
```

```json
{
  "type": "CLIPBOARD_WRITE",
  "id": "req-uuid",
  "ts": 1712345678.0,
  "payload": {
    "format": "text",
    "content": "hello"
  }
}
```

Responds with `ACK`.

Security note: clipboard content is **redacted in the audit log** (size +
SHA-256 prefix only, never the value — same rule as `INPUT_ACTION.text`
per [AUDIT_LOG.md](AUDIT_LOG.md)). If the user has a password manager
plugin that writes secrets to the clipboard, the agent could read them;
this is a documented limitation, not a bug, and a `CLIPBOARD_READ` always
emits a user-visible audit event regardless of redaction.

### Deliberately omitted from v1

- Image clipboard (need PNG round-trip; defer with `screenshot.region`
  which covers most use cases).
- File-list clipboard (Finder copy → paste-as-files in another app); maps
  poorly to a string payload.
- Clipboard *sync* with the platform — the platform isn't running on the
  user's machine, so there's no shared clipboard to sync to; defer.

---

### New: `PERMISSIONS_CHECK_REQUEST` / `PERMISSIONS_CHECK_RESPONSE`

Lets the platform pre-flight a workflow ("do I have Screen Recording?
do I have Accessibility?") before issuing a doomed `INPUT_ACTION` that
silently does nothing on macOS.

```json
{
  "type": "PERMISSIONS_CHECK_REQUEST",
  "id": "req-uuid",
  "ts": 1712345678.0,
  "payload": {
    "permissions": ["screen_recording", "accessibility", "input_monitoring"]
  }
}
```

```json
{
  "type": "PERMISSIONS_CHECK_RESPONSE",
  "id": "req-uuid",
  "ts": 1712345678.0,
  "payload": {
    "permissions": {
      "screen_recording": "granted",     // "granted" | "denied" | "unknown" | "not_applicable"
      "accessibility": "denied",
      "input_monitoring": "unknown"
    }
  }
}
```

`not_applicable` is the answer on Linux for "screen_recording" (X11
doesn't have a TCC equivalent — it's allowed-by-default or
blocked-by-compositor). The platform should treat `unknown` as "try and
see"; the shim returns `unknown` rather than guessing.

This op does **not** open System Settings or prompt the user. Triggering
permission prompts is a [PLATFORM_HOOKS.md](PLATFORM_HOOKS.md) install-time
concern; mid-session prompts would interrupt the agent loop in a way the
platform can't recover from.

---

### Wire-op summary

New message types added in v1:

- `CURSOR_POSITION_REQUEST` / `CURSOR_POSITION_RESPONSE`
- `DISPLAY_INFO_REQUEST` / `DISPLAY_INFO_RESPONSE`
- `WINDOW_LIST_REQUEST` / `WINDOW_LIST_RESPONSE`
- `WINDOW_FOCUS` (responds with `ACK`)
- `APP_LIST_REQUEST` / `APP_LIST_RESPONSE`
- `APP_LAUNCH` (responds with `ACK` carrying `pid`)
- `CLIPBOARD_READ` / `CLIPBOARD_READ_RESPONSE`
- `CLIPBOARD_WRITE` (responds with `ACK`)
- `PERMISSIONS_CHECK_REQUEST` / `PERMISSIONS_CHECK_RESPONSE`

Existing message types extended:

- `SCREENSHOT_REQUEST` — adds `region`, `window_id`, `format`, `include_cursor`
- `SCREENSHOT_RESPONSE` — adds `region`, `display_scale`, `logical_size`
- `INPUT_ACTION` — adds `button`, `modifiers`, `scroll_amount`, `duration_ms`,
  `path`, `paste`, and new action verbs: `middle_click`, `triple_click`,
  `mouse_down`, `mouse_up`, `drag`, `hold_key`, `wait`

New error code: `FEATURE_NOT_SUPPORTED` — feature not in
`HELLO.computer_use_features`, or attempted on an unsupported OS.

---

## v2 (deferred)

These are the bigger swings that need a real design pass and probably their
own RFCs. Listed so we don't accidentally reinvent them in v1.

### Accessibility tree (`A11Y_TREE_REQUEST`)

cua-driver exposes `get_accessibility_tree` (Swift-only today, Rust has
parity). Returns the OS-native a11y hierarchy: roles, labels, values,
frames. This is the *right* answer for grounding ("click the Submit
button") without pixel-coordinate guesswork — OSWorld papers consistently
show a11y-augmented agents outperform vision-only.

Why deferred: macOS AX requires `pyobjc-framework-ApplicationServices`
and an `AXObserver` per app, Windows wants `pywinauto`/`uiautomation`
(big trees, slow walks), Linux X11 wants AT-SPI + D-Bus. Each OS has its
own role taxonomy. v2 should normalize to a single cross-OS schema —
roles like `button`, `text_field`, `link`, `image`, with `actions` listing
what's invokable. Don't ship until we've prototyped on all three OSes.

### OCR + Set-of-Mark grounding (`SET_OF_MARK_REQUEST`)

OSWorld's strongest agents annotate the screenshot with numbered bounding
boxes the model refers to by ID ("click element 7"). This needs an OCR
backend (Tesseract, Apple Vision, Windows OCR), an element detector, and
a render step that returns the marked-up screenshot. We don't want to
build that in the shim (heavy deps, model quality matters) — it lives
better on the platform side, with the shim providing raw screenshots +
a11y trees as input.

### Native scripting (`SCRIPT_INVOKE`)

AppleScript on macOS, PowerShell + COM on Windows, `gdbus`/AT-SPI on
Linux. Direct API access into apps (set Safari's URL field, query
Calendar's events, drive Excel) skips the whole screenshot-and-click
dance. Huge power, huge per-app surface — defer to v2 once we know which
3-5 apps are in the critical path.

### Record-and-replay (`RECORDING_START` / `RECORDING_STOP` / `REPLAY`)

cua-driver has `set_recording` + `replay_trajectory`. Lets a user demo a
workflow once, then the agent replays it later. Useful for cron-style
"every Monday morning, do this" automation. The interesting research
question is *how the agent decides what's parameterizable* in a recorded
trace — pure replay is fragile (timestamps, dialog placement). Defer
until we have the demand and a story on robustness.

### File-as-drag-source (`DRAG_FILE`)

"Drag this file from the workspace into the chat window." Native drag
sources need OS-level pasteboard handoff (NSPasteboardItem on macOS,
`DoDragDrop` on Windows, XDND on Linux X11). Different per OS, and the
target app has to accept it. `INPUT_ACTION.drag` covers visual drags
that originate from a screen position; v2's `DRAG_FILE` covers drags
that originate from a *file* and need to look to the target app like a
Finder drag.

### Multi-monitor coordinate normalization

v1 keeps `coordinate` in *primary-monitor logical pixels with origin at
top-left*. Multi-monitor setups need per-monitor coordinates or a global
virtual coordinate space. The right call depends on how the model emits
coordinates after seeing a multi-monitor `screenshot.region` — likely
v2 adds `coordinate_space: "global"|"monitor"|"window"` and a
`monitor_index` / `window_id` qualifier.

### Notifications, system events, IME state

`get_notifications`, `get_active_input_method`, `is_screen_locked`,
`get_focused_text_field_value` — useful, OS-specific, not on the
critical path for "drive a form across three apps." Punt.

---

## Per-OS capability matrix (v1 ops)

Legend: ✅ supported, 🟡 supported with caveat (named), ❌ not supported (reason).

| Wire op / feature | macOS | Linux (X11) | Linux (Wayland) | Windows | WSL2 |
|---|---|---|---|---|---|
| `SCREENSHOT_REQUEST` (full display) | ✅ via `mss` + Quartz | ✅ via `mss` | 🟡 portal+pipewire only; default ❌ | ✅ via `mss` | 🟡 WSLg via X server only |
| `SCREENSHOT_REQUEST.region` | ✅ | ✅ | ❌ (same Wayland block) | ✅ | 🟡 same as full |
| `SCREENSHOT_REQUEST.window_id` | 🟡 via `CGWindowListCreateImage` (needs Screen Recording) | 🟡 X11 XComposite-dependent; off-screen windows may return stale bitmaps | ❌ | ✅ via `PrintWindow` / `BitBlt` | ❌ no Win32 access from inside the Linux VM |
| `SCREENSHOT_REQUEST.include_cursor` | ✅ | ✅ | ❌ | ✅ | 🟡 |
| `INPUT_ACTION` (existing 7 verbs) | ✅ pyautogui → Quartz | ✅ pyautogui → Xlib | ❌ blocked | ✅ pyautogui → SendInput | 🟡 only into X server, not Win32 |
| `INPUT_ACTION.middle_click` | ✅ | ✅ | ❌ | ✅ | 🟡 |
| `INPUT_ACTION.triple_click` | ✅ native triple-click event | ✅ via three click events ~50ms apart (X11 has no native triple) | ❌ | ✅ | 🟡 |
| `INPUT_ACTION.modifiers` on click/scroll | ✅ | ✅ | ❌ | ✅ | 🟡 |
| `INPUT_ACTION.mouse_down`/`mouse_up` | ✅ | ✅ | ❌ | ✅ | 🟡 |
| `INPUT_ACTION.drag` (path) | ✅ | 🟡 X11 may glitch if path is > 100 pts; we sub-sample | ❌ | ✅ | 🟡 |
| `INPUT_ACTION.hold_key` | ✅ | ✅ | ❌ | ✅ | 🟡 |
| `INPUT_ACTION.wait` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `INPUT_ACTION.type.paste` | ✅ Cmd+V via NSPasteboard | ✅ Ctrl+V via xclip/wl-clipboard | 🟡 Wayland: clipboard works, paste keystroke blocked | ✅ Ctrl+V | 🟡 X-server clipboard only |
| `CURSOR_POSITION_REQUEST` | ✅ via Quartz | ✅ via `XQueryPointer` | 🟡 Wayland: returns last-known from `pyautogui`, may lag | ✅ via `GetCursorPos` | 🟡 |
| `DISPLAY_INFO_REQUEST` | ✅ via Quartz `CGGetActiveDisplayList` | ✅ via `xrandr` | 🟡 via `wlr-randr` if present, else single-display fallback | ✅ via `EnumDisplayMonitors` | 🟡 reports the X server's view |
| `WINDOW_LIST_REQUEST` | ✅ via `CGWindowListCopyWindowInfo` | 🟡 via `wmctrl` / EWMH; needs window manager that exposes `_NET_CLIENT_LIST` | ❌ no equivalent in Wayland | ✅ via `EnumWindows` | 🟡 X server windows only |
| `WINDOW_FOCUS` | ✅ via `AXRaise` (needs Accessibility) | ✅ via `_NET_ACTIVE_WINDOW` | ❌ | 🟡 `SetForegroundWindow` honors foreground-lock rules (best-effort) | 🟡 |
| `APP_LIST_REQUEST` | ✅ via `NSWorkspace.runningApplications` | ✅ via `/proc` + .desktop heuristics | ✅ same as X11 (procfs is compositor-agnostic) | ✅ via `EnumProcesses` + window-owner | 🟡 |
| `APP_LAUNCH` | ✅ via `NSWorkspace.openApplication` (PyObjC) | ✅ via Popen + .desktop resolution | ✅ same as X11 | ✅ via `ShellExecute` (pywin32) | 🟡 launches in WSL2, not on Windows host |
| `CLIPBOARD_READ` (text) | ✅ via `NSPasteboard` | ✅ via `xclip` if installed, else dependency-missing | 🟡 via `wl-paste` if installed | ✅ via `OpenClipboard` | 🟡 X server clipboard only |
| `CLIPBOARD_WRITE` (text) | ✅ | ✅ via `xclip` | 🟡 via `wl-copy` | ✅ | 🟡 |
| `PERMISSIONS_CHECK_REQUEST` | ✅ TCC: ScreenCapture / Accessibility / InputMonitoring | ✅ returns `not_applicable` for TCC keys; `granted` if display server is X11 | ✅ returns `denied` for screen_recording unless portal session live | ✅ checks process elevation + UIAccess | ✅ |

Across the board: **Wayland is the largest single capability cliff.** On
a Wayland session the shim should advertise `computer_use_features: ["wait",
"display.info", "app.list", "app.launch", "clipboard.read",
"clipboard.write", "permissions.check"]` and omit everything that needs
input injection or screen capture. The platform should then degrade to
a "talk through bash + files only" loop. This is per
[CROSS_PLATFORM.md → Linux](CROSS_PLATFORM.md) which already gates the
`computer_use` capability on `XDG_SESSION_TYPE=x11`; with the
fine-grained `computer_use_features` list we can additionally allow a
"clipboard + app launch only" Wayland mode that today is wholesale
blocked.

WSL2's column is consistently 🟡 because input/screen capture only reaches
the X server inside the VM, not the Windows host — useful for testing
the shim but rarely what the user actually wants. The shim advertises
`computer_use_features` honestly; the platform decides whether to use it.

---

## What capabilities the shim advertises in `HELLO`

The shim populates `HELLO.payload.computer_use_features` at startup by
running a lightweight probe per feature:

1. **Dependencies present?** — `pyautogui`, `mss`, `Pillow` for v0; per-OS
   adds (`pyobjc-framework-Quartz` on macOS, `pywin32` on Windows,
   `python-xlib` / `wmctrl` on Linux X11, `wl-clipboard` for Wayland
   clipboard).
2. **OS API reachable?** — try `mss.mss().monitors` once for display
   probe; try `NSWorkspace.sharedWorkspace()` once for macOS app probe;
   etc. Failure → drop the feature, don't crash startup.
3. **Compositor / session type?** — `XDG_SESSION_TYPE`, `WAYLAND_DISPLAY`,
   `DISPLAY` env vars on Linux; `sys.platform` everywhere else.
4. **Permissions cached?** — on macOS, run a no-op `CGEventCreate` and
   catch the TCC failure; if it fails, drop input features. Don't trigger
   the prompt mid-session (see PERMISSIONS_CHECK_REQUEST contract above).

The list is sticky for the session — we don't re-probe mid-session.
Reconnect → new `HELLO` → re-probe.

---

## What we deliberately left out (cua-driver has it, we don't)

| cua-driver tool | Why we skipped it |
|---|---|
| `set_value` (direct AX-set on text fields) | Big security surface — agent can stuff text into a focused password field without user-visible keystrokes or audit trail. cua-driver's MCP-stdio model trusts the operator; our threat model has the platform behind a token over WebSocket. Force `type` + `paste`: visible cursor activity, observable in audit. |
| `get_accessibility_tree` | v2. See above — needs per-OS schema normalization we haven't done. |
| `page` tool (CDP / AppleScript page introspection) | Overlaps with platform's existing browser tooling. Don't duplicate. |
| `agent_cursor.*` (5 tools) | This is a visual animated cursor overlay so end-users can *see* the agent acting. Genuinely cool UX but pure presentation; doesn't affect what the agent can do. Belongs in a v2 polish pass. |
| `set_recording` / `replay_trajectory` | v2. Story on parameterization of recorded traces is unsolved. |
| `launch_app` via NSWorkspace as the *only* mechanism | We use NSWorkspace on macOS (great) but fall back to `ShellExecute` / `Popen` elsewhere rather than insisting on the "no shell" purity. `APP_LAUNCH` is a distinct audit event from `EXECUTE_COMMAND`, which gives us the observability cua-driver was chasing without requiring per-OS native API parity. |
| `get_config` / `set_config` | cua-driver's are about runtime config of the driver itself (cursor animation, etc.). Our shim is configured at install via [PLATFORM_HOOKS.md](PLATFORM_HOOKS.md) and we don't want a remote config-write surface. |
| `zoom` as a separate tool | Folded into `SCREENSHOT_REQUEST.region`. Anthropic's `computer_20251124` `zoom` action maps 1:1 to a region screenshot; no need for a separate wire op. |

---

## Open questions

1. **Coordinate space when `region` is set.** When the platform sends
   `INPUT_ACTION.coordinate` after receiving a region screenshot, are
   the coordinates in *region-local* space (0,0 = region top-left) or
   *display-global* space (the shim adds `region.origin`)? Anthropic's
   `zoom` returns a region but expects subsequent clicks in display
   coordinates. Our wire op should pick one and document it — I'm
   leaning display-global because it matches Anthropic.

2. **`window_id` lifetime across reconnects.** If the platform caches
   `window_id` from before a disconnect and the shim reconnects, do we
   invalidate the IDs (forcing a fresh `WINDOW_LIST`) or attempt to
   re-bind them? Invalidation is simpler but means more list calls.

3. **Clipboard sandbox.** The shim's path-jail covers file operations
   inside `allowed_root`. Clipboard is global — anything the user has
   copied is readable, anything we write is visible to all apps. Do we
   add a per-feature toggle (`enable_clipboard` in shim config) so the
   user can install the shim with computer-use but no clipboard access?
   Or fold it into `computer_use_features` advertised at HELLO time?

4. **`paste` action and password managers.** If the user has 1Password
   filling forms via the system clipboard, our `type ... paste: true`
   will clobber it. Do we (a) accept this and document, (b) detect and
   refuse via clipboard-type sniffing, or (c) add a `preserve_clipboard:
   false` opt-out and default-on for restore?

5. **macOS Accessibility permission UX.** v1's `PERMISSIONS_CHECK` is
   read-only by design. But the first time `INPUT_ACTION` runs without
   Accessibility granted, macOS shows the system prompt and the keystroke
   is lost. Do we (a) accept the lost keystroke (model retries on next
   screenshot) or (b) detect-then-retry with a 1s wait? cua-driver does
   (b) via its "TCC auto-relaunch" daemon trick which is more involved
   than we want for v1.
