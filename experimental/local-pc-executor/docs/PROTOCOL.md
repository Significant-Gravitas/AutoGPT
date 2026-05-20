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
  "payload": { ... }
}
```

`id` is used for request/response correlation. Every request gets a response with the same `id`.

---

## Message Types

### Handshake

#### `HELLO` (shim → platform, on connect)
```json
{
  "type": "HELLO",
  "id": "uuid",
  "ts": 1234567890.0,
  "payload": {
    "shim_version": "0.1.0",
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
  "payload": {
    "session_id": "session-uuid",
    "granted_capabilities": ["shell", "files"],  // subset platform approved
    "max_file_size_bytes": 10485760,
    "command_timeout_seconds": 30
  }
}
```

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
    "mime_type": "text/csv"        // best-effort; null if unknown
  }
}
```

If `exists: false`, all other fields are null. Path-jail violation returns
`ERROR` with `PATH_OUTSIDE_ALLOWED_ROOT` (not `FILE_STAT_RESPONSE` with
`exists: false`) — the two cases must be distinguishable.

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

#### `SCREENSHOT_REQUEST` (platform → shim)
```json
{
  "type": "SCREENSHOT_REQUEST",
  "id": "req-uuid",
  "ts": 1234567890.0,
  "payload": {
    "monitor": 0,           // 0 = primary, -1 = all monitors stitched
    "quality": 75           // JPEG quality 1-100
  }
}
```

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
    "monitor": 0
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
                                // | "mouse_move" | "type" | "key" | "scroll"
    "coordinate": [500, 300],   // for click/move/scroll
    "text": null,               // for "type"
    "key": null,                // for "key" e.g. "ctrl+s"
    "direction": null,          // for "scroll": "up" | "down"
    "clicks": null              // for "scroll": number of clicks
  }
}
```

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
- `COMMAND_TIMEOUT` — command exceeded timeout
- `SHELL_NOT_AVAILABLE` — `shell` selector requested a shell not installed on this OS
- `UNSUPPORTED_ARCH` — HELLO `arch` value not recognized
- `CAPABILITY_NOT_GRANTED` — requested capability not in granted_capabilities
- `AUTH_FAILED` — token invalid or expired
- `SHIM_OVERLOADED` — too many concurrent requests
- `INTERNAL_ERROR` — unexpected shim error

---

### Keepalive

#### `PING` / `PONG`
```json
{ "type": "PING", "id": "uuid", "ts": 1234567890.0, "payload": {} }
{ "type": "PONG", "id": "same-uuid", "ts": 1234567890.01, "payload": {} }
```

Platform sends `PING` every 30s. Shim must respond with `PONG` within 10s or connection is dropped.

---

## Concurrency

The platform may send multiple requests before receiving responses (pipelined). The shim
assigns each request its own async task and responds with matching `id` when complete.
Max concurrent requests: `HELLO_ACK.max_concurrent` (default 4).

## Reconnection

Shim uses exponential backoff: `min(2^attempt * 1s, 60s) + jitter(0-5s)`.
On reconnect, shim sends a new `HELLO`. Platform re-issues `HELLO_ACK` with same session.
Any in-flight requests at disconnect time are considered failed; platform retries if safe.
