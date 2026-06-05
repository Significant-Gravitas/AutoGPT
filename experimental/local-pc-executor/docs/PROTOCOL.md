# Local PC Executor — WebSocket Protocol Spec

> **Status**: Draft v0.1 — subject to change

## Transport

- **Protocol**: WebSocket over TLS (`wss://`)
- **Endpoint**: `wss://platform.autogpt.net/ws/local-executor/{session_id}`
- **Direction**: Outbound from shim (NAT/firewall friendly — no inbound ports needed)
- **Auth**: Bearer token in `Authorization` header on WebSocket upgrade request

## Message Format

All messages are JSON with this envelope:

```json
{
  "type": "MESSAGE_TYPE",
  "id": "uuid-v4",
  "ts": 1712345678.123,
  "version": "1.0",
  "payload": { ... }
}
```

`id` is used for request/response correlation. Every request gets a response with the same `id`.

`version` is the wire-protocol version this sender speaks, formatted
`"major.minor"`. Always emitted by both sides. Receivers MUST be lenient
on non-HELLO frames — use the negotiated version (see
[Versioning](#versioning)) as the source of truth.

---

## Versioning

The wire protocol carries a `version` field on every envelope, and HELLO /
HELLO_ACK each carry a `protocol_version` field in the payload that
advertises the **maximum** version that side supports.

### Negotiation

On connect, the shim sends `HELLO.payload.protocol_version` (its max). The
platform replies with `HELLO_ACK.payload.protocol_version` (its max). Both
sides compute the **effective negotiated version** as:

| Case | Result |
|---|---|
| Same major, same minor | That version. |
| Same major, different minor | Same major, `min(shim_minor, platform_minor)`. Both sides MUST tolerate forward-compatible additions within a major. |
| Different majors | **Hard error.** Platform closes the WebSocket with close code **4426** and reason `{"error": "PROTOCOL_VERSION_MISMATCH", "shim_max": "...", "platform_max": "...", "hint": "..."}` (JSON-encoded). The shim MUST log the mismatch and MUST NOT auto-reconnect until restarted — this avoids hot-reconnect storms against an incompatible peer. |

### Frame-level `version`

All non-HELLO frames SHOULD set `version` to the negotiated value, but
receivers MUST treat the HELLO-time negotiation as truth. A receiver that
sees a non-HELLO frame with a different *minor* MUST process it normally;
a frame with a different *major* MUST be dropped and logged loudly (and
on the platform side, the session SHOULD be torn down with code 4426).

### Current versions

| Side | Maximum | Notes |
|---|---|---|
| Shim | `1.0` | `autogpt_local_executor.protocol.VERSION` |
| Platform | `1.0` | Mirror this constant in the platform repo. |

---

## Message Types

### Handshake

#### `HELLO` (shim → platform, on connect)
```json
{
  "type": "HELLO",
  "id": "uuid",
  "ts": 1234567890.0,
  "version": "1.0",
  "payload": {
    "shim_version": "0.1.0",
    "protocol_version": "1.0",     // max wire version this shim supports
    "machine_id": "hostname-uuid4",
    "platform": "darwin",          // "darwin" | "linux" | "windows" | "wsl2"
    "arch": "arm64",               // "x86_64" | "arm64" (normalized; see below)
    "screen_resolution": [2560, 1440],   // null if computer_use not available
    "capabilities": [
      "shell",                     // always present
      "files",                     // always present
      "computer_use",              // optional: pyautogui available
      "local_llm",                 // optional: ollama running
      "hardware_serial",           // optional: pyserial available
      "hardware_usb",              // optional: pyusb available
      "hardware_gpio"              // optional: RPi.GPIO available
    ],
    "allowed_root": "/Users/alice/autogpt-workspace",
    "local_llm_models": ["llama3.2:3b", "mistral:7b"],   // empty if no local_llm cap
    "hardware_devices": [
      {"type": "serial", "port": "/dev/ttyUSB0", "desc": "Arduino Uno"},
      {"type": "usb",    "vid": "2341", "pid": "0043", "desc": "Arduino Uno"}
    ]
  }
}
```

#### `HELLO_ACK` (platform → shim)
```json
{
  "type": "HELLO_ACK",
  "id": "same-uuid-as-HELLO",
  "ts": 1234567890.1,
  "version": "1.0",
  "payload": {
    "session_id": "session-uuid",
    "protocol_version": "1.0",     // max wire version this platform supports
    "granted_capabilities": ["shell", "files"],  // subset platform approved
    "max_file_size_bytes": 10485760,
    "command_timeout_seconds": 30,
    "max_concurrent": 4                          // shim caps in-flight requests
  }
}
```

The effective negotiated wire-protocol version is computed per
[Versioning](#versioning). Mismatched majors close the WebSocket with code
4426 and the shim disables auto-reconnect.

`max_concurrent` sizes the shim-side request semaphore. The shim must
refuse (with `SHIM_OVERLOADED`) any request that arrives while the
semaphore is at its cap. Default 4 if platform omits the field.

#### `HELLO.platform` enum

| Value | When | Detection |
|---|---|---|
| `darwin` | macOS | `sys.platform == "darwin"` |
| `linux` | Native Linux (any distro) | `sys.platform == "linux"` and `/proc/version` does NOT contain `microsoft` |
| `windows` | Windows 10/11 (any arch) | `sys.platform == "win32"` |
| `wsl2` | WSL2 distro on Windows host | `sys.platform == "linux"` and `/proc/version` contains `microsoft` |

Rationale: `win32` (Python's `sys.platform`) is misleading on 64-bit Windows
and is a Python-specific quirk. We use Node.js / Go-style names that match what
the rest of the toolchain expects. `wsl2` is broken out because the host-OS
boundary changes hardware, networking, and computer-use semantics — see
[CROSS_PLATFORM.md → WSL2](CROSS_PLATFORM.md#wsl2).

#### `HELLO.arch` enum

| Value | Aliases accepted from the shim (`platform.machine()`) |
|---|---|
| `x86_64` | `x86_64`, `AMD64`, `amd64` |
| `arm64` | `arm64`, `aarch64`, `ARM64` |

Shim is responsible for normalization. Any other value causes the platform to
reject the HELLO with `UNSUPPORTED_ARCH`.

---

### Shell Execution

#### `EXECUTE_COMMAND` (platform → shim)
```json
{
  "type": "EXECUTE_COMMAND",
  "id": "req-uuid",
  "ts": 1234567890.0,
  "payload": {
    "command": "ls -la /tmp",     // mutually exclusive with "argv"
    "argv": null,                  // OR ["ls", "-la", "/tmp"] — no shell parsing
    "shell": "auto",               // "auto" | "bash" | "sh" | "zsh" | "pwsh"
                                   //   | "powershell" | "cmd"
                                   // ignored when "argv" is set
    "cwd": "/Users/alice/autogpt-workspace",
    "timeout_seconds": 30,
    "env": {"MY_VAR": "value"}    // merged with shim's safe env (see below)
  }
}
```

**`command` vs `argv`**: exactly one must be set.
- `command` (string): passed to a shell selected by `shell`. Subject to shell
  parsing — the platform is responsible for quoting.
- `argv` (array of strings): executed directly via `subprocess` with no shell.
  Portable across OSes and immune to shell-quoting bugs. Use this whenever
  shell features (pipes, globs, redirection, env-var expansion) aren't needed.

**`shell` selector** (used only when `command` is set):

| Value | macOS / Linux default binary | Windows default binary |
|---|---|---|
| `auto` | `bash` if on PATH else `sh` | `cmd.exe` |
| `bash` | `bash` | `bash` if installed (Git Bash / WSL); else `SHELL_NOT_AVAILABLE` error |
| `sh` | `sh` | not available; error |
| `zsh` | `zsh` if installed; else error | not available; error |
| `pwsh` | `pwsh` if installed (PowerShell Core); else error | `pwsh` if installed; else error |
| `powershell` | not available; error | `powershell.exe` (Windows PowerShell 5.1) |
| `cmd` | not available; error | `cmd.exe` |

Platform code that today calls `bash -c "..."` must switch to either:
1. Wire `argv` form (preferred), or
2. `shell: "auto"` and rely on the shim to do the right thing.

**`env` field**: the wire payload uses `env` (a `dict[str, str]`).
The Python adapter on the platform side (the `LocalPCShim.commands.run`
method that duck-types E2B's `AsyncSandbox`) accepts E2B's `envs=` kwarg
and translates to the wire `env`. **Don't rename either side**; keep both.

The shim merges wire-supplied `env` into a per-OS safe baseline (see
[CROSS_PLATFORM.md → Environment variables](CROSS_PLATFORM.md#environment-variables)).
On Windows, env-var names are case-folded before merging to mirror Win32
semantics.

#### `COMMAND_RESULT` (shim → platform)
```json
{
  "type": "COMMAND_RESULT",
  "id": "req-uuid",
  "ts": 1234567890.5,
  "payload": {
    "stdout": "total 48\n...",
    "stderr": "",
    "exit_code": 0,
    "timed_out": false,
    "duration_seconds": 0.12
  }
}
```

---

### File Operations

#### `FILE_READ` (platform → shim)
```json
{
  "type": "FILE_READ",
  "id": "req-uuid",
  "ts": 1234567890.0,
  "payload": {
    "path": "/Users/alice/autogpt-workspace/data.csv",
    "encoding": "utf-8",    // "utf-8" | "base64"
    "format": "text",        // "text" | "bytes" — see mapping below
    "offset": 0,
    "length": null          // null = whole file
  }
}
```

**`encoding` ↔ `format` mapping**: the wire keeps `encoding` for backward
compatibility, but the Python adapter on the platform side accepts E2B's
`format=` kwarg (`"text"` or `"bytes"`) and translates as follows:

| E2B Python `format=` | Wire `format` | Wire `encoding` | Returned content |
|---|---|---|---|
| `"text"` (default) | `"text"` | `"utf-8"` | UTF-8 string |
| `"bytes"` | `"bytes"` | `"base64"` | base64-encoded raw bytes |

Adapters MUST send both `format` and `encoding` on the wire even though
they're redundant — the redundancy keeps each side independently parseable
and lets the platform-side adapter speak both E2B and shim vocabularies
without translation.

**CRLF / LF policy for `format: "text"`**: the shim returns the bytes
**as-is from disk**, decoded as UTF-8. It does NOT auto-translate `\r\n` to
`\n`. A Windows-authored text file read with `format: "text"` will contain
`\r\n` in the returned string; a Unix-authored file will contain `\n`. If
bytewise fidelity matters (diffing, hashing, content-addressed storage),
the platform MUST request `format: "bytes"` instead. Rationale: text mode
on Python's `open()` already auto-translates, but doing it at the shim
hides the on-disk truth from the platform and breaks reproducibility.

#### `FILE_CONTENTS` (shim → platform)
```json
{
  "type": "FILE_CONTENTS",
  "id": "req-uuid",
  "ts": 1234567890.1,
  "payload": {
    "content": "col1,col2\n1,2\n",
    "encoding": "utf-8",
    "size_bytes": 16,
    "truncated": false
  }
}
```

#### `FILE_WRITE` (platform → shim)
```json
{
  "type": "FILE_WRITE",
  "id": "req-uuid",
  "ts": 1234567890.0,
  "payload": {
    "path": "/Users/alice/autogpt-workspace/output.txt",
    "content": "hello world\n",
    "encoding": "utf-8",
    "create_parents": true
  }
}
```

**`create_parents` jail semantics**: when `create_parents: true` and the
target's parents don't exist yet, the shim jail-checks the **nearest
existing ancestor** plus the lexical parent. Both must be inside
`allowed_root`. This catches:

- `path="/etc/foo/bar", create_parents=true` → lexical parent `/etc/foo`
  is outside `allowed_root` → reject.
- `path="/workspace/sym-to-etc/bar", create_parents=true` → realpath of
  the nearest existing ancestor (`/workspace/sym-to-etc` → `/etc`)
  resolves outside → reject.

The shim creates parents only after both checks pass.

#### `ACK` (shim → platform, for writes and fire-and-forget ops)
```json
{
  "type": "ACK",
  "id": "req-uuid",
  "ts": 1234567890.1,
  "payload": { "ok": true }
}
```

#### `FILE_STAT` (platform → shim)

Used in place of shell `ls -la` / `stat` / `test -e`. Cross-OS by design.

```json
{
  "type": "FILE_STAT",
  "id": "req-uuid",
  "ts": 1234567890.0,
  "payload": {
    "path": "/Users/alice/autogpt-workspace/data.csv",
    "follow_symlinks": true
  }
}
```

#### `FILE_STAT_RESPONSE` (shim → platform)

```json
{
  "type": "FILE_STAT_RESPONSE",
  "id": "req-uuid",
  "ts": 1234567890.1,
  "payload": {
    "exists": true,
    "is_file": true,
    "is_dir": false,
    "is_symlink": false,
    "size_bytes": 16,
    "mtime": 1712345678.0,        // seconds since epoch, float
    "ctime": 1712345670.0,
    "mode": "0644",                // POSIX octal; on Windows, derived from
                                   // FILE_ATTRIBUTE_READONLY + ACLs
    "owner_uid": 501,              // null on Windows
    "owner_gid": 20,               // null on Windows
    "mime_type": "text/csv",       // best-effort; null if unknown
    "path": "/Users/alice/autogpt-workspace/data.csv"   // canonical path
                                   // (realpath-resolved when follow_symlinks=true)
  }
}
```

`path` is the canonical resolved path — what `readlink -f` would return
on POSIX, or the equivalent on Windows after junction/reparse-point
resolution. Platform-side adapter code uses this in place of the
`readlink -f` shellout (see PLATFORM_HOOKS.md §10.2).

If `exists: false`, the other fields are null EXCEPT `path` may still
be set to the resolved-but-nonexistent target (lets the caller distinguish
"file gone" from "path is in a different tree"). Path-jail violation
returns `ERROR` with `PATH_OUTSIDE_ALLOWED_ROOT` (not `FILE_STAT_RESPONSE`
with `exists: false`) — the two cases must be distinguishable.

#### `FILE_LIST` (platform → shim)

Used in place of shell `ls` / `find`. Optional glob pattern.

```json
{
  "type": "FILE_LIST",
  "id": "req-uuid",
  "ts": 1234567890.0,
  "payload": {
    "path": "/Users/alice/autogpt-workspace",
    "glob": "*.csv",               // null = list everything
    "recursive": false,
    "include_hidden": false,
    "max_entries": 1000            // shim caps at this; truncated flag in response
  }
}
```

#### `FILE_LIST_RESPONSE` (shim → platform)

```json
{
  "type": "FILE_LIST_RESPONSE",
  "id": "req-uuid",
  "ts": 1234567890.1,
  "payload": {
    "entries": [
      {
        "name": "data.csv",
        "path": "/Users/alice/autogpt-workspace/data.csv",
        "is_file": true,
        "is_dir": false,
        "is_symlink": false,
        "size_bytes": 16,
        "mtime": 1712345678.0
      }
    ],
    "truncated": false
  }
}
```

Glob is matched per-OS with `pathlib.PurePath.match`. On case-insensitive
filesystems (macOS APFS default, Windows NTFS default), matching is
case-insensitive; on case-sensitive Linux ext4, it's case-sensitive. The
glob is matched against the **basename**, not the full path; combine with
`recursive: true` to descend.

#### `FILE_DELETE` (platform → shim)

Used in place of shell `rm` / `del`.

```json
{
  "type": "FILE_DELETE",
  "id": "req-uuid",
  "ts": 1234567890.0,
  "payload": {
    "path": "/Users/alice/autogpt-workspace/old.csv",
    "recursive": false,            // required true to delete a non-empty dir
    "missing_ok": false            // if true, no error when path doesn't exist
  }
}
```

Responds with `ACK`. Shim's recycle-bin mode (see SECURITY.md Layer 5 —
optional) moves to `~/.autogpt-trash/` instead of unlinking.

#### `FILE_MOVE` (platform → shim)

Used in place of shell `mv` / `move`. Also covers renames.

```json
{
  "type": "FILE_MOVE",
  "id": "req-uuid",
  "ts": 1234567890.0,
  "payload": {
    "src": "/Users/alice/autogpt-workspace/a.csv",
    "dst": "/Users/alice/autogpt-workspace/b.csv",
    "overwrite": false             // if true, dst is replaced atomically when possible
  }
}
```

**Both `src` and `dst` are path-jail checked.** Cross-device moves
(e.g., source on the home volume, dst on a mounted external drive) are
allowed if both endpoints are inside `allowed_root`; the shim falls back to
copy+delete in that case. Responds with `ACK`.

---

### Computer Use

**For the full v1 computer-use feature spec and per-OS capability matrix, see [COMPUTER_USE.md](COMPUTER_USE.md).**

#### `SCREENSHOT_REQUEST` (platform → shim)
```json
{
  "type": "SCREENSHOT_REQUEST",
  "id": "req-uuid",
  "ts": 1234567890.0,
  "payload": {
    "monitor": 0,                       // 0 = primary, -1 = all monitors stitched
    "quality": 75,                      // JPEG quality 1-100
    "region": null,                     // optional [x1, y1, x2, y2]
    "window_id": null,                  // optional opaque ID from WINDOW_LIST_RESPONSE
    "format": "jpeg",                   // "jpeg" | "png"
    "include_cursor": true              // overlay the OS cursor in the capture
  }
}
```

`region` and `window_id` are mutually exclusive. Coordinates in `region`
are display-global, unscaled, top-left-origin pixels — same as
`INPUT_ACTION.coordinate`. See
[COMPUTER_USE.md](COMPUTER_USE.md) for the full feature spec.

#### `SCREENSHOT_RESPONSE` (shim → platform)
```json
{
  "type": "SCREENSHOT_RESPONSE",
  "id": "req-uuid",
  "ts": 1234567890.2,
  "payload": {
    "image_base64": "...",
    "mime_type": "image/jpeg",
    "width": 2560,
    "height": 1440,
    "monitor": 0,
    "region": null,                     // echoed when request had one
    "display_scale": 1.0,
    "logical_size": [2560, 1440],
    "meta": {                           // see COMPUTER_USE.md Q1
      "origin": [0, 0],
      "display_id": 0
    }
  }
}
```

#### `INPUT_ACTION` (platform → shim)
```json
{
  "type": "INPUT_ACTION",
  "id": "req-uuid",
  "ts": 1234567890.0,
  "payload": {
    "action": "left_click",     // "left_click" | "right_click" | "double_click"
                                // | "middle_click" | "triple_click"
                                // | "mouse_move" | "mouse_down" | "mouse_up"
                                // | "drag" | "type" | "key" | "hold_key"
                                // | "scroll" | "wait"
    "coordinate": [500, 300],   // for click/move/scroll
    "text": null,               // for "type"
    "key": null,                // for "key" e.g. "ctrl+s"
    "direction": null,          // for "scroll": "up" | "down"
    "clicks": null,             // for "scroll": number of clicks
    "button": null,             // "left" | "middle" | "right" — for mouse_down/up/drag
    "modifiers": null,          // subset of "shift"|"ctrl"|"alt"|"super" — for click/scroll
    "scroll_amount": null,      // alternative to clicks for "scroll"
    "scroll_direction": null,   // alternative to direction for "scroll"
    "duration_ms": null,        // for "hold_key" and "wait"
    "path": null,               // [[x,y], ...] for "drag"
    "paste": false,             // for "type" — see Q4
    "preserve_clipboard": false // for "type" with paste — see Q4
  }
}
```

Coordinates are always **display-global, unscaled, top-left-origin
virtual-display pixels** regardless of any prior region screenshot.
Out-of-bounds coordinates return `INPUT_OUT_OF_BOUNDS`. See
[COMPUTER_USE.md §Q1](COMPUTER_USE.md#q1--coordinate-space-locked).

#### Other computer-use ops (xref)

The following message types are specified in full in
[COMPUTER_USE.md](COMPUTER_USE.md). Brief signatures:

| Type | Direction | Returns |
|---|---|---|
| `CURSOR_POSITION_REQUEST` | platform → shim | `CURSOR_POSITION_RESPONSE` with `{x, y, monitor}` |
| `DISPLAY_INFO_REQUEST` | platform → shim | `DISPLAY_INFO_RESPONSE` with per-monitor `index`, `primary`, `physical_size`, `logical_size`, `scale`, `origin` |
| `WINDOW_LIST_REQUEST` | platform → shim | `WINDOW_LIST_RESPONSE` with `windows[].window_id` (shim-minted `win_<uuid>`, see Q2) |
| `WINDOW_FOCUS` | platform → shim | `ACK` (or `WINDOW_STALE`) |
| `APP_LIST_REQUEST` | platform → shim | `APP_LIST_RESPONSE` |
| `APP_LAUNCH` | platform → shim | `ACK` with `pid` |
| `CLIPBOARD_READ` | platform → shim | `CLIPBOARD_READ_RESPONSE` or `CLIPBOARD_CONCEALED` |
| `CLIPBOARD_WRITE` | platform → shim | `ACK` |
| `PERMISSIONS_CHECK_REQUEST` | platform → shim | `PERMISSIONS_CHECK_RESPONSE` |

---

### Error

#### `ERROR` (either direction)
```json
{
  "type": "ERROR",
  "id": "req-uuid",
  "ts": 1234567890.0,
  "payload": {
    "code": "PATH_OUTSIDE_ALLOWED_ROOT",
    "message": "Path /etc/passwd is outside allowed root /Users/alice/autogpt-workspace",
    "fatal": false
  }
}
```

Error codes:
- `PATH_OUTSIDE_ALLOWED_ROOT` — file op tried to escape allowed_root
- `PATH_RESERVED_NAME` — Windows reserved name (CON/PRN/NUL/COM*/LPT*)
- `PATH_INVALID_CHARS` — path contains chars illegal on this OS
- `PATH_NOT_FOUND` — FILE_STAT/READ/DELETE on a path that doesn't exist (and `missing_ok: false`)
- `PATH_NOT_EMPTY` — FILE_DELETE on a non-empty dir with `recursive: false`
- `PATH_EXISTS` — FILE_MOVE with `overwrite: false` and dst exists
- `COMMAND_TIMEOUT` — reserved for future strict-timeout transport; today
  EXECUTE_COMMAND on timeout returns `COMMAND_RESULT` with `timed_out: true`
  and `exit_code = -1` rather than an ERROR. Callers should branch on the
  `timed_out` flag.
- `SHELL_NOT_AVAILABLE` — `shell` selector requested a shell not installed on this OS
- `UNSUPPORTED_ARCH` — HELLO `arch` value not recognized
- `CAPABILITY_NOT_GRANTED` — requested capability not in granted_capabilities
- `AUTH_FAILED` — token invalid or expired
- `SHIM_OVERLOADED` — too many concurrent requests (exceeded `max_concurrent`)
- `INTERNAL_ERROR` — unexpected shim error
- `FILE_TOO_LARGE` — FILE_READ / FILE_WRITE exceeded
  `HELLO_ACK.max_file_size_bytes`
- `DEPENDENCY_MISSING` — a runtime dep needed for the op (pyautogui,
  Pillow, xclip, etc.) isn't installed on the shim host
- `WINDOW_STALE` — computer-use `window_id` no longer maps to a live
  window. Caller must re-issue `WINDOW_LIST_REQUEST`. See
  [COMPUTER_USE.md §Q2](COMPUTER_USE.md#q2--window_id-lifetime-locked).
- `PERMISSION_PENDING` — an OS-level permission required for the op
  (macOS Accessibility / Screen Recording) wasn't granted, or was
  revoked mid-session. Distinct from `CAPABILITY_NOT_GRANTED`. See
  [COMPUTER_USE.md §Q5](COMPUTER_USE.md#q5--macos-tcc-first-prompt-ux-locked).
- `FEATURE_NOT_SUPPORTED` — requested computer-use op is not in
  `HELLO.computer_use_features`, or attempted on an unsupported OS /
  session (e.g. Wayland input). See [COMPUTER_USE.md](COMPUTER_USE.md).
- `CLIPBOARD_CONCEALED` — clipboard contents are not readable under the
  active sandbox policy. See [COMPUTER_USE.md §Q3](COMPUTER_USE.md#q3--clipboard-sandbox-model-locked).
- `INPUT_OUT_OF_BOUNDS` — `INPUT_ACTION.coordinate` is outside the
  union of connected display rects. Error payload includes the valid
  display rects so the caller can correct. See
  [COMPUTER_USE.md §Q1](COMPUTER_USE.md#q1--coordinate-space-locked).

Error `ERROR.payload` MAY carry an optional `details` object whose shape
depends on `code` — see the per-code examples in
[COMPUTER_USE.md](COMPUTER_USE.md).

---

### Keepalive

#### `PING` / `PONG`
```json
{ "type": "PING", "id": "uuid", "ts": 1234567890.0, "payload": {} }
{ "type": "PONG", "id": "same-uuid", "ts": 1234567890.01, "payload": {} }
```

Platform sends `PING` every 30s. Shim must respond with `PONG` within 10s or connection is dropped.

---

### Session ownership

The platform owns at most one active WebSocket per `session_id`. Two shims
trying to claim the same `session_id` (e.g. the same user starting
`autogpt-shim start` on two laptops with the same auth) is a real concern
— the v0 spec was silent on it and the platform did silent last-write-wins,
orphaning the prior shim's pending requests with no clean error.

#### Policy

- **First connecting shim wins ownership** until it explicitly disconnects
  OR the platform sends `SESSION_REVOKED`.
- A second shim from the **SAME `machine_id`** is treated as a re-connect
  (legitimate takeover — laptop sleep/wake, etc.). The platform serves
  `SESSION_REVOKED` to the old shim with reason `another_shim_connected`
  and accepts the new one.
- A second shim from a **DIFFERENT `machine_id`** is REJECTED with WS
  close code **4427** (`SESSION_TAKEN_OVER`) and a structured reason. The
  rejected shim MUST NOT auto-reconnect. Rationale: avoid silent
  split-brain where two machines both try to execute one Claude turn —
  the file-system / window state diverges, audit chains fork, and the
  platform can't meaningfully retry a half-finished command on the "other"
  shim.

#### `SESSION_REVOKED` (platform → shim)
```json
{
  "type": "SESSION_REVOKED",
  "id": "uuid",
  "ts": 1234567890.0,
  "version": "1.0",
  "payload": {
    "reason": "another_shim_connected",   // | "user_revoked" | "platform_shutdown"
    "new_shim_machine_id": "macbook-air-7f3c"   // optional, set when reason is another_shim_connected
  }
}
```

On receipt the shim MUST:

1. Log a `SESSION_REVOKED` audit event with the reason and source.
2. Send no further frames on this connection.
3. Gracefully close its half of the WebSocket (close code 4428 from the
   shim side is acceptable; receivers tolerate any clean close after
   `SESSION_REVOKED`).
4. **NOT auto-reconnect** to the same session_id. Operator restart is
   required to re-establish the session deliberately.

Receivers MUST tolerate unknown future `reason` values (forward-compatible
minor extension); the spec table above is the v1.0 set.

#### WS close-code table

| Code | Label | Meaning | Sender | Shim auto-reconnect? |
|---|---|---|---|---|
| 4426 | `PROTOCOL_VERSION_MISMATCH` | Major-version disagreement in HELLO/HELLO_ACK. | platform | **No** — restart required after upgrade. |
| 4427 | `SESSION_TAKEN_OVER` | A same-`machine_id` shim connected and took over (this connection is the old one being evicted). | platform | **No** — the takeover is intentional. |
| 4428 | `SESSION_REVOKED` | User revoked this session in the platform UI (or related admin action). | platform | **No** — auth gone, would just 401. |
| 4429 | `PLATFORM_SHUTDOWN` | Platform is going down (graceful). | platform | **No** for *this* session — reconnect attempts SHOULD wait for platform health probe. Today the shim just halts; operator restart triggers a fresh connect. |

Codes below 4426 follow IETF/RFC semantics (1000 normal, 1011 server
error, etc.) and the shim DOES auto-reconnect with exponential backoff —
only the application-layer codes in the table above are treated as fatal
"do not retry without operator action".

> The platform-side `ShimConnectionManager` change that actually emits
> `SESSION_REVOKED` and close-code 4427 lives in a separate ticket; this
> section is the contract the shim implements on the receive side.

---

## Concurrency

The platform may send multiple requests before receiving responses (pipelined). The shim
assigns each request its own async task and responds with matching `id` when complete.
Max concurrent requests: `HELLO_ACK.max_concurrent` (default 4).

---

## Backpressure

Pre-#38 the only backpressure signal was the after-the-fact
`SHIM_OVERLOADED` error — the platform issued a request, the shim
rejected it. Better: have the shim proactively advertise free capacity
so the platform can throttle issuance before it hits the wall.

### `pending_capacity` on every response envelope

All shim-→-platform response envelopes carry a top-level
`pending_capacity: int` (placed at envelope level, peer to `id` /
`ts` / `version`). Value:

```
pending_capacity = max_concurrent - in_flight_after_this_response
```

i.e. free slots immediately AFTER this response is sent (we've already
released ours). `0` means "I'm fully saturated — pause issuance". Field
is omitted (or `null`) on platform-→-shim requests and on shim-internal
frames like HELLO; the platform MUST treat absent/null as "no signal,
use prior value".

The platform's `LocalPCShim` adapter SHOULD maintain a per-shim
in-memory `capacity_remaining` counter, decrement on issue, refresh
from `pending_capacity` on each response, and pause new issuance when
the counter hits 0. Wiring that consumer is a separate platform-side
ticket; this section is the contract.

### `STATUS` frame (shim → platform, unsolicited, periodic)

```json
{
  "type": "STATUS",
  "id": "uuid",
  "ts": 1234567890.0,
  "version": "1.0",
  "pending_capacity": 3,
  "payload": {
    "in_flight": 1,
    "max_concurrent": 4,
    "queue_depth": 0,
    "audit_log_bytes": 12345,
    "uptime_seconds": 137.4
  }
}
```

Emission cadence:

- **Every 30s** (`STATUS_INTERVAL_SECONDS` in shim source) while the WS
  is healthy. Cheap unsolicited heartbeat — lets the platform observe
  shim health without spamming an `is_alive` probe.
- **On the full → not-full edge.** When the shim finishes a request
  that took the last free slot, a STATUS frame is emitted IMMEDIATELY
  after the response (in addition to the response carrying
  `pending_capacity=1`), so the platform's throttle releases without
  waiting up to 30s for the next tick.

Receivers MUST treat `STATUS` as advisory — the per-response
`pending_capacity` remains the authoritative number. A receiver that
doesn't understand `STATUS` MAY drop it; the field is forward-compatible
and the spec's discriminated union just gains it within v1.x.

> The platform-side consumer that reads `pending_capacity` / `STATUS`
> and throttles issuance lives in a separate ticket; this section is
> the contract the shim implements on the emit side.

## Reconnection

Shim uses exponential backoff: `min(2^attempt * 1s, 60s) + jitter(0-5s)`.
On reconnect, shim sends a new `HELLO`. Platform re-issues `HELLO_ACK` with same session.
Any in-flight requests at disconnect time are considered failed; platform retries if safe.
