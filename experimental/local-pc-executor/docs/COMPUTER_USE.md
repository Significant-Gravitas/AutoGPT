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
Inside `HELLO.payload`, add a new field. Two granularities are supported
side by side, both populated by the shim from a live capability probe:

- `computer_use_features` (legacy, fine-grained): the per-op feature flags
  the platform consults before sending a wire op. Each entry corresponds to
  one v1 op or one extension to an existing op.
- `computer_use_features_coarse`: a small fixed set of broad buckets
  (`screenshot`, `input`, `windows`, `apps`, `clipboard`, `permissions`)
  that the platform can use for at-a-glance UX ("clipboard not available"
  vs enumerating eight clipboard sub-features).

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
  ],
  "computer_use_features_coarse": [
    "screenshot",
    "input",
    "windows",
    "apps",
    "clipboard",
    "permissions"
  ]
}
```

The coarse buckets per OS (when computer-use is available at all):

| OS / session | Coarse buckets advertised |
|---|---|
| macOS (AX + Screen Recording granted) | `["screenshot", "input", "windows", "apps", "clipboard", "permissions"]` |
| Windows | `["screenshot", "input", "windows", "apps", "clipboard", "permissions"]` |
| Linux X11 | `["screenshot", "input", "apps", "clipboard", "permissions"]` (windows omitted unless `wmctrl` or AT-SPI is present) |
| Linux Wayland | `["screenshot"]` (input + window listing blocked; clipboard via portal is deferred) |
| WSL2 | `[]` (no display server reachable to the Windows host) |

`clipboard` only appears when the shim was started with
`--enable-clipboard` (per Q3 below); without the flag, clipboard sub-ops
return `FEATURE_NOT_SUPPORTED` even on capable OSes.

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

`SCREENSHOT_RESPONSE` gains a `region` echo, a `display_scale` field so
the platform can map model-space coordinates back to physical pixels on
HiDPI displays (this is the same trap that bit the cua-driver Linux build —
hardcoded `scale_factor: 1.0`), and a `meta` block carrying the crop's
`origin` and `display_id` (see Q1 below):

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
    "logical_size": [1440, 900],
    "meta": {
      "origin": [100, 200],
      "display_id": 0
    }
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

#### Q1 — Coordinate space (locked)

**Coordinates on the wire are always display-global, unscaled,
top-left-origin virtual-display pixels.** Always. There is no
region-local or window-local coordinate mode.

- `region` on a screenshot request only crops the **image returned**;
  it does not shift any subsequent `INPUT_ACTION.coordinate`. The shim
  never owns coordinate state. If the platform asks for a 800×500 crop
  starting at `(100, 200)` and then sends `INPUT_ACTION.coordinate =
  [150, 250]`, the click lands at `(150, 250)` in display-global pixels —
  i.e. 50 px right and 50 px down from the crop's top-left, but expressed
  in the same coordinate space as a full-screen click.
- `SCREENSHOT_RESPONSE.meta.origin` echoes the `[x, y]` top-left of the
  returned crop (= `[region[0], region[1]]` when `region` was set, or
  `[0, 0]` for a full display) so the platform can map model output back
  if it needs to.
- `SCREENSHOT_RESPONSE.meta.display_id` echoes which display the crop
  came from (matches `DISPLAY_INFO_RESPONSE.monitors[i].index`). Same
  display IDs used in `INPUT_ACTION` validation.
- `INPUT_ACTION` rejects any `coordinate` outside the union of connected
  display bounds with `INPUT_OUT_OF_BOUNDS` (new error code). The error
  payload **echoes the valid display rects** so the platform can self-correct
  on the next turn instead of guessing what happened.

Example `INPUT_OUT_OF_BOUNDS` error for a `(10000, 10000)` click on a
single-display 1920×1080 setup:

```json
{
  "type": "ERROR",
  "id": "req-uuid",
  "ts": 1712345678.0,
  "payload": {
    "code": "INPUT_OUT_OF_BOUNDS",
    "message": "coordinate (10000, 10000) is outside the union of connected display bounds",
    "fatal": false,
    "details": {
      "requested_coordinate": [10000, 10000],
      "displays": [
        {"index": 0, "origin": [0, 0], "size": [1920, 1080]}
      ]
    }
  }
}
```

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

#### `type` action — input method (Q4 locked)

Today `_pyautogui.write` is character-by-character. We add an optional
`paste: true` field on the `type` action that the shim implements by
stashing the text on the OS clipboard, sending Cmd/Ctrl+V, and (only when
the platform asks) restoring the previous clipboard. Why: pasting a 4 KB
form value char-by-char is ~80 seconds at default 20 ms interval and
trips IME/autocomplete on web forms.

**Behavior — locked:**

1. **Threshold.** `paste: true` is **only honored when `len(text) >= 200`.**
   Under that threshold the shim falls through to per-key typing, which
   avoids the clipboard race for short strings that don't actually need
   it.
2. **Default = clobber-and-document.** `paste: true` overwrites the
   clipboard, sends the OS paste hotkey, and **does not restore.** This
   is the safe-by-default for password-manager + paste interactions: if
   the agent intentionally pasted something, the user's clipboard now
   contains it, period. No surprise reversal.
3. **Opt-in restore via `preserve_clipboard: true`.** When the platform
   explicitly asks, the shim snapshots the clipboard before pasting and
   restores it afterwards — **but skips the restore if the OS
   `changeCount` / clipboard-sequence-number advanced between snapshot
   and restore.** A bumped sequence number means the user (or another
   app) copied something during the paste, and clobbering that with the
   pre-paste snapshot would lose their action.
4. **Race window.** Snapshot → write → hotkey → wait-for-paste → restore
   is 50–500 ms wall-time depending on OS and target app. During that
   window other apps see the pasted text on the clipboard. This is **not
   papered over**: it's a security-relevant trade-off and the user
   accepts it by enabling `--enable-clipboard`. The `preserve_clipboard`
   path adds a slim "user-action-during-paste cancels restore"
   protection but does not close the snooping window itself.

```json
{
  "action": "type",
  "text": "long form value...",
  "paste": true,
  "preserve_clipboard": false
}
```

`InputActionPayload.paste` and `InputActionPayload.preserve_clipboard`
both default to `false`. With `paste: false` (default), `preserve_clipboard`
is ignored.

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

#### Q2 — `window_id` lifetime (locked)

`window_id` is a shim-minted opaque UUID with a `win_` prefix (e.g.
`win_3f9a8c12-...`), never reused. The mapping table
`{uuid: native_handle}` is wiped on every `HELLO` (process restart **and**
reconnect both invalidate IDs).

The shim maintains the mapping for its process lifetime. At USE time
(`WINDOW_FOCUS`, `SCREENSHOT_REQUEST.window_id`, anything that takes a
`window_id`), the shim re-verifies the native handle by checking the
window's `(pid, class_name, creation_timestamp)` triple against what was
captured at LIST time. If any of those changed, the window was destroyed
and the OS reassigned the handle; the shim returns `WINDOW_STALE`
**never silently re-binds**. The platform must re-issue `WINDOW_LIST_REQUEST`
to get a fresh mapping.

Example `WINDOW_LIST_RESPONSE`:

```json
{
  "payload": {
    "windows": [
      {
        "window_id": "win_3f9a8c12-4b1d-4e8a-9c2a-0f7b8e9d6c5f",
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

Example `WINDOW_STALE` error:

```json
{
  "type": "ERROR",
  "id": "req-uuid",
  "ts": 1712345678.0,
  "payload": {
    "code": "WINDOW_STALE",
    "message": "window_id win_3f9a8c12-... no longer maps to a live window",
    "fatal": false,
    "details": {
      "window_id": "win_3f9a8c12-4b1d-4e8a-9c2a-0f7b8e9d6c5f",
      "hint": "re-list windows"
    }
  }
}
```

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
per [AUDIT_LOG.md](AUDIT_LOG.md)). A `CLIPBOARD_READ` always emits a
user-visible audit event regardless of redaction.

#### Q3 — Clipboard sandbox model (locked)

**Default-deny.** Clipboard ops are off unless the user opts in at install
time. Two CLI flags on `autogpt-shim start`:

| Flag | Default | Effect |
|---|---|---|
| `--enable-clipboard` | off | Enables `CLIPBOARD_WRITE` and a **writeback-only** flavor of `CLIPBOARD_READ`: `CLIPBOARD_READ` returns content **only if the shim itself wrote it within the last 30 seconds.** Anything the user (or any other app) put on the clipboard returns `CLIPBOARD_CONCEALED` with `reason: "writeback_only"`. |
| `--enable-clipboard-read-foreign` | off (requires `--enable-clipboard`) | Lets `CLIPBOARD_READ` return any clipboard content **except** when the current pasteboard carries an `org.nspasteboard.ConcealedType` marker (macOS, the de-facto password-manager convention) or `CF_PRIVATE` clipboard format (Windows). Concealed contents → `CLIPBOARD_CONCEALED` error. |

Without `--enable-clipboard`, all clipboard sub-ops return
`FEATURE_NOT_SUPPORTED` and `clipboard` is omitted from
`computer_use_features_coarse`. Without
`--enable-clipboard-read-foreign`, `CLIPBOARD_READ` is the strict
"only what we just wrote" variant.

The 30-second writeback window is tracked by the shim by stashing the
SHA-256 + `changeCount`/sequence-number at write time; a subsequent
`CLIPBOARD_READ` checks that the stashed value still matches the live
clipboard contents (defeats the "shim wrote X, password manager
overwrote with Y" race) before returning. A mismatch returns
`CLIPBOARD_CONCEALED` with `reason: "writeback_overwritten"`.

Example `CLIPBOARD_CONCEALED` error (concealed-type case):

```json
{
  "type": "ERROR",
  "id": "req-uuid",
  "ts": 1712345678.0,
  "payload": {
    "code": "CLIPBOARD_CONCEALED",
    "message": "clipboard contents marked concealed by source application",
    "fatal": false,
    "details": {
      "reason": "concealed_type",
      "marker": "org.nspasteboard.ConcealedType"
    }
  }
}
```

Example for the writeback-only case (default `--enable-clipboard`,
agent tries to read user-copied text):

```json
{
  "payload": {
    "code": "CLIPBOARD_CONCEALED",
    "message": "clipboard contents not written by the shim within the last 30s",
    "fatal": false,
    "details": {
      "reason": "writeback_only",
      "writeback_age_seconds": null
    }
  }
}
```

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

#### Q5 — macOS TCC first-prompt UX (locked)

Layered, so the user only ever hits the prompt at a moment they're
expecting to ("I just ran the installer") and not in the middle of an
agent loop:

1. **`autogpt-shim doctor` subcommand.** Run at install time, manually,
   or by the install script. On macOS, calls
   `AXIsProcessTrustedWithOptions({"AXTrustedCheckOptionPrompt": True})`
   to **proactively surface** the consent dialog the first time it runs;
   subsequent runs are silent. Also probes
   `CGPreflightScreenCaptureAccess()` and reports both states. On Linux,
   detects X11 vs Wayland. On Windows, reports UAC elevation level.

   Example output (macOS, first run, AX not yet granted):

   ```
   $ autogpt-shim doctor
   ✓ allowed_root: ~/Documents/autogpt-workspace exists, writable
   ✓ keychain: keyring available (macOS Keychain Services)
   ⚠ Accessibility: not granted — System Settings → Privacy & Security → Accessibility, enable autogpt-shim
   ✓ Screen Recording: granted
   ✓ display: 2 displays detected (3024×1964 + 2560×1440)
   ○ computer_use: degraded (Accessibility missing)
   exit 78
   ```

   Exit code is `0` when all requested capabilities are healthy, `78`
   (config error) when computer-use was requested but a prerequisite
   permission is missing.

2. **Daemon refuses to bind WS when consent is missing.** If
   `computer_use` is in the shim config's requested capabilities AND
   `AXIsProcessTrusted()` returns false, the daemon writes a structured
   record to the audit log (`code: PERMISSION_PENDING`, the missing
   permissions, a hint that the user should run `autogpt-shim doctor`),
   logs to stderr, and exits with code `78`. Without `computer_use` in
   the requested set, the daemon binds normally.

3. **launchd `KeepAlive` triggers re-launch on TCC change.** The
   `autogpt-shim install` template includes:

   ```xml
   <key>KeepAlive</key>
   <dict>
     <key>SuccessfulExit</key>
     <false/>
   </dict>
   ```

   This means: any non-zero exit relaunches the daemon. When the user
   grants Accessibility in System Settings, the OS modifies the TCC.db,
   which (a) doesn't notify the running process directly but (b) the
   *next* time `AXIsProcessTrusted()` is called, the new state is
   visible. Combined with `KeepAlive: SuccessfulExit=false`, the chain
   is: daemon exits 78 → launchd re-spawns → new process sees fresh
   TCC state → binds WS. Documented in
   [CROSS_PLATFORM.md](CROSS_PLATFORM.md).

4. **`PERMISSION_PENDING` error on revocation mid-session.** If the user
   revokes Accessibility while the daemon is running, the next
   `INPUT_ACTION` or AX-dependent op catches the OS-level denial and
   returns `PERMISSION_PENDING` (new error code, distinct from
   `CAPABILITY_NOT_GRANTED` because the granting context here is
   the *OS*, not the platform). The platform LLM retries on the next
   turn; the user, after seeing the error in their UI, can re-grant.

   Example `PERMISSION_PENDING` error:

   ```json
   {
     "type": "ERROR",
     "id": "req-uuid",
     "ts": 1712345678.0,
     "payload": {
       "code": "PERMISSION_PENDING",
       "message": "Accessibility permission revoked or not yet granted",
       "fatal": false,
       "details": {
         "permission": "accessibility",
         "platform": "darwin",
         "hint": "Open System Settings → Privacy & Security → Accessibility and enable autogpt-shim"
       }
     }
   }
   ```

5. **Linux / Windows `doctor`.** Returns `0` with platform-specific
   advice: on Linux, distinguishes X11 (OK) vs Wayland (computer-use
   degraded by default), and warns on missing `xclip`/`wl-clipboard`. On
   Windows, reports UAC level and warns that injecting into elevated
   windows from a non-elevated shim is impossible.

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

New error codes introduced by the computer-use surface:

- `FEATURE_NOT_SUPPORTED` — feature not in `HELLO.computer_use_features`,
  or attempted on an unsupported OS / session (e.g. Wayland input).
- `WINDOW_STALE` — `window_id` no longer maps to a live window. Caller
  must re-list (Q2).
- `INPUT_OUT_OF_BOUNDS` — `INPUT_ACTION.coordinate` is outside the union
  of connected display rects. Error carries the valid display rects (Q1).
- `CLIPBOARD_CONCEALED` — clipboard contents are not readable under the
  current sandbox policy (Q3): either the active flag set doesn't permit
  reading foreign contents, the writeback window has expired or been
  overwritten, or the source app marked the contents concealed.
- `PERMISSION_PENDING` — an OS permission required for the op was not
  granted (or was revoked mid-session). Q5; distinct from
  `CAPABILITY_NOT_GRANTED`, which means the *platform* didn't grant the
  shim the capability.

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

## Locked decisions (cross-reference)

The five questions that were open at the v0.1 draft are now decided
inline in the spec above. Quick index:

| Question | Section | Summary |
|---|---|---|
| Q1 — coordinate space with `region` | [Extended: `SCREENSHOT_REQUEST` → Q1](#q1--coordinate-space-locked) | Display-global, unscaled, top-left-origin virtual-display pixels always. `region` only crops the image. `INPUT_OUT_OF_BOUNDS` echoes display rects. |
| Q2 — `window_id` lifetime | [New: `WINDOW_LIST_RESPONSE` → Q2](#q2--window_id-lifetime-locked) | Shim-minted `win_<uuid>`, never reused, wiped on `HELLO`. Re-verify `(pid, class_name, creation_timestamp)` at USE time; mismatch → `WINDOW_STALE`. |
| Q3 — clipboard sandbox | [New: `CLIPBOARD_READ` / `CLIPBOARD_WRITE` → Q3](#q3--clipboard-sandbox-model-locked) | Default-deny. `--enable-clipboard` + optional `--enable-clipboard-read-foreign`. 30 s writeback-only window. `CLIPBOARD_CONCEALED`. |
| Q4 — `paste:true` and password managers | [`type` action — input method (Q4)](#type-action--input-method-q4-locked) | Threshold ≥200 chars. Clobber-and-document by default; opt-in `preserve_clipboard: true` with changeCount-skip-restore. Race window honestly documented. |
| Q5 — macOS TCC first-prompt UX | [`PERMISSIONS_CHECK_REQUEST` → Q5](#q5--macos-tcc-first-prompt-ux-locked) | `autogpt-shim doctor` + daemon refuses to bind when computer-use requested but ungranted + launchd `KeepAlive: SuccessfulExit=false` + `PERMISSION_PENDING` on mid-session revocation. |
