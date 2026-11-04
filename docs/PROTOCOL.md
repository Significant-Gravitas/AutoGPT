# Local PC Executor — WebSocket Protocol Spec

> **Status**: Draft v0.1 — subject to change

## Transport

- **Protocol**: WebSocket over TLS (`wss://`)
- **Control endpoint**: `wss://platform.autogpt.net/ws/local-executor`. Normal
  `autogpt-shim start` keeps this OAuth-authenticated machine connection alive.
- **Data endpoint**: `wss://platform.autogpt.net/ws/local-executor/{session_id}`.
  The control channel activates isolated child runtimes on this existing route.
- **Legacy binding**: explicitly passing `--session-id` skips control mode and
  connects directly to the data endpoint for compatibility.
- **Frame limit**: the shim accepts WebSocket messages up to 16 MiB, matching
  the platform relay/Uvicorn envelope cap. Per-operation payload limits remain
  the lower negotiated `HELLO_ACK.max_file_size_bytes` value.
- **Direction**: Outbound from shim (NAT/firewall friendly — no inbound ports needed)
- **Auth**: Bearer token in `Authorization` header on WebSocket upgrade request

## Message Format

All messages are JSON with this envelope:

```json
{
  "type": "MESSAGE_TYPE",
  "id": "uuid-v4",
  "ts": 1712345678.123,
  "version": "1.1",
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
| Different majors | **Hard error.** The side that notices closes the WebSocket with close code **4426**. A close reason holds at most 123 bytes (RFC 6455 §5.5), so it is short: the shim sends compact JSON, `{"error":"PROTOCOL_VERSION_MISMATCH","shim_max":"...","platform_max":"..."}` (just the `error` field if the versions don't fit), and logs the upgrade hint; the platform sends a short text reason. The shim MUST log the mismatch and MUST NOT auto-reconnect until restarted — this avoids hot-reconnect storms against an incompatible peer. |

### Frame-level `version`

All non-HELLO frames SHOULD set `version` to the negotiated value, but
receivers MUST treat the HELLO-time negotiation as truth. A receiver that
sees a non-HELLO frame with a different *minor* MUST process it normally;
a frame with a different *major* MUST be dropped and logged loudly (and
on the platform side, the session SHOULD be torn down with code 4426).

### Current versions

| Side | Maximum | Notes |
|---|---|---|
| Shim | `1.1` | `autogpt_local_executor.protocol.VERSION` |
| Platform | `1.1` | Mirror this constant in the platform repo. |

---

## Message Types

### Handshake

#### `HELLO` (shim → platform, on connect)
```json
{
  "type": "HELLO",
  "id": "uuid",
  "ts": 1234567890.0,
  "version": "1.1",
  "payload": {
    "shim_version": "0.1.0",
    "protocol_version": "1.1",     // max wire version this shim supports
    "machine_id": "hostname-uuid4",
    "platform": "darwin",          // "darwin" | "linux" | "windows" | "wsl2"
    "arch": "arm64",               // "x86_64" | "arm64" (normalized; see below)
    "screen_resolution": [2560, 1440],   // null if computer_use not available
    "capabilities": [
      "directory_browse",          // control WS only
      "shell",                     // optional: explicit local --enable-shell opt-in
      "files",                     // always present
      "computer_use",              // optional: pyautogui available
      "local_llm",                 // optional: ollama running
      "hardware_serial",           // optional: pyserial available
      "hardware_usb",              // optional: pyusb available
      "hardware_gpio"              // optional: RPi.GPIO available
    ],
    "allowed_root": null,        // control WS; canonical root on a data WS
    "local_llm_models": ["llama3.2:3b", "mistral:7b"],   // empty if no local_llm cap
    "hardware_devices": [
      {"type": "serial", "port": "/dev/ttyUSB0", "desc": "Arduino Uno"},
      {"type": "usb",    "vid": "2341", "pid": "0043", "desc": "Arduino Uno"}
    ]
  }
}
```

`recording` message types are reserved design-preview protocol. The shim never
advertises the `recording` capability, and startup fails closed if its preview
flag is enabled, until capture and interpretation are complete end to end.

#### `HELLO_ACK` (platform → shim)
```json
{
  "type": "HELLO_ACK",
  "id": "same-uuid-as-HELLO",
  "ts": 1234567890.1,
  "version": "1.1",
  "payload": {
    "session_id": null,          // null on control; exact ID on data WS
    "connection_id": null,       // optional platform presence generation
    "protocol_version": "1.1",     // max wire version this platform supports
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
All three limits must be positive. The shim applies the lower of its local
configuration and the value in `HELLO_ACK`, so the platform can reduce a
ceiling but cannot expand access beyond the user's local policy.

The ACK must carry the same message ID as `HELLO` and only capabilities
advertised by the shim. On a data connection it must carry the exact selected
session ID. On the base control connection it must carry a null/empty session
ID, an opaque `connection_id`, protocol 1.1, and the `directory_browse`
capability. A control daemon fails terminally when any of those fields are
missing. The shim enforces the granted subset on every subsequent request.

### Machine control and per-chat roots

The base control connection advertises `directory_browse`. It accepts only the
messages in this section; `FILE_*`, shell, computer-use, local-LLM, and recording
messages are rejected until an isolated per-session data connection is active.
All browse references are invalidated whenever the control WebSocket reconnects.

#### `DIRECTORY_LIST_REQUEST` / `DIRECTORY_LIST_RESPONSE`

Both reference fields null start a five-minute browse and return virtual roots.
Subsequent requests must provide both opaque values. When `next_cursor` is
present, the caller can send it with the same browse and directory references
to fetch another page. Cursors are minted and resolved only by the host. There
is no platform path, recursive flag, glob, offset, or caller-controlled limit.

```json
{
  "type": "DIRECTORY_LIST_REQUEST",
  "id": "request-uuid",
  "ts": 1234567890.0,
  "payload": {"browse_id": null, "directory_ref": null}
}
```

```json
{
  "type": "DIRECTORY_LIST_RESPONSE",
  "id": "request-uuid",
  "ts": 1234567890.1,
  "payload": {
    "browse_id": "host-opaque-id",
    "current": null,
    "parent_ref": null,
    "entries": [{
      "directory_ref": "host-opaque-ref",
      "name": "Home",
      "path": "/Users/alice"
    }],
    "next_cursor": null,
    "truncated": false,
    "expires_at": 1234568190.0
  }
}
```

Each page contains at most 200 immediate real directories and a browse scans at
most 1,000 entries. `truncated` is true only when that hard scan/reference bound
omits entries; `next_cursor` indicates that another safe page is available.
Files, symlinks/junctions, inaccessible entries, and special roots are skipped.
Responses carry only display name, canonical absolute path, and opaque
reference—never content, size, timestamps, permissions, or recursive metadata.

#### `ATTACH_SESSION` / `SESSION_ATTACHED`

The platform allocates a draft/chat session ID and returns the reference chosen
by the user. `expected_connection_id` is carried for platform race detection;
the host intentionally ignores it because reconnect invalidation makes stale
references fail closed.

```json
{
  "type": "ATTACH_SESSION",
  "id": "request-uuid",
  "ts": 1234567890.0,
  "payload": {
    "session_id": "session-uuid",
    "browse_id": "host-opaque-id",
    "directory_ref": "host-opaque-ref",
    "expected_connection_id": "platform-generation-or-null"
  }
}
```

```json
{
  "type": "SESSION_ATTACHED",
  "id": "request-uuid",
  "ts": 1234567890.1,
  "payload": {
    "session_id": "session-uuid",
    "allowed_root": "/Users/alice/Projects/AutoGPT",
    "fingerprint": "64-lowercase-hex-characters",
    "revision": 1,
    "root_grant": "v1.base64url-claims.base64url-hmac"
  }
}
```

The shim strictly re-resolves the directory, fingerprints its canonical path
and filesystem identity, audits the change, consumes the browse, and signs a
grant with a key derived from the machine-only audit key. The global config is
never changed.

#### Activation, restoration, and detachment

`ACTIVATE_SESSION {session_id, revision}` starts a child daemon with a deep copy
of configuration and the attached root. It replies `SESSION_ACTIVATED` with
`session_id`, `allowed_root`, `fingerprint`, and `revision`; readiness of the
child data WebSocket remains the platform presence registry's responsibility.

`RESTORE_SESSION {session_id, root_grant}` verifies signature, machine/session
binding, canonical root, and current filesystem fingerprint, then restores and
starts the child. `SESSION_RESTORED` returns the same binding fields plus the
grant. Grants are durable but revocable through OAuth; they are tamper-evident,
not encrypted.

`DETACH_SESSION {session_id}` stops the child, removes the in-memory binding,
and replies with the standard `ACK`. An active session's root is immutable;
detach before attaching a different folder.

The control daemon also stops a non-recording child after ten minutes without
a data request, while retaining its signed binding for `RESTORE_SESSION`. This
prevents abandoned chats from consuming the eight-child safety cap; the
platform transparently restores the child when the next turn begins.

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

The shim disables this capability by default. It is advertised and accepted
only after the local user starts the daemon with `--enable-shell` (or the
equivalent config setting). Both `command` and direct `argv` execution run with
the shim user's normal OS permissions. `allowed_root` constrains `FILE_*`
operations and the subprocess working directory; it does not sandbox what the
process can access.

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
    "duration_seconds": 0.12,
    "output_truncated": false
  }
}
```

The shim drains stdout and stderr concurrently and retains at most
`HELLO_ACK.max_file_size_bytes` across both fields. If output exceeds that
shared budget, it continues draining to avoid subprocess deadlock and appends
`[command output truncated]` to `stderr` inside the same byte budget, with
`output_truncated: true`. The retained text is also capped below the 16 MiB
transport ceiling to account for worst-case JSON escaping.

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

`offset` and `length` are byte-based. A bounded range may be read from a file
larger than `max_file_size_bytes`; the shim seeks and reads only that range.
Whole-file reads and requested ranges larger than the effective limit return
`FILE_TOO_LARGE`.

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

A symlink or junction is removed itself; its target is untouched. The allowed
root itself cannot be deleted.

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

**Both `src` and `dst` are path-jail checked.** Moves use filesystem rename
or replacement only. Cross-device moves return an error without copying file
contents or pre-deleting the destination. Successful moves respond with `ACK`.
A symlink `src` is moved itself, and with `overwrite` a symlink `dst` is
replaced itself; neither target changes.

---

### Local LLM

**For the full local LLM routing feature spec, gating, and backend matrix, see [LOCAL_LLM.md](LOCAL_LLM.md).**

Local-model routing is disabled scaffolding: the companion never advertises
`local_llm`, leaves `HELLO.local_llm_models` empty, and refuses the enable flag.
The wire shapes below are retained for tests and future integration, not a
currently supported chat path. A missing capability must never be overridden.

#### `LOCAL_LLM_COMPLETION` (platform → shim)

```json
{
  "type": "LOCAL_LLM_COMPLETION",
  "id": "req-uuid",
  "ts": 1234567890.0,
  "version": "1.1",
  "payload": {
    "model": "llama3.2:3b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Summarize X in 3 bullets."}
    ],
    "max_tokens": 1024,
    "temperature": 0.0,
    "top_p": 1.0,
    "stream": true
  }
}
```

- `model` MUST be one of the strings advertised in
  `HELLO.local_llm_models`. Unknown model → `MODEL_NOT_AVAILABLE`.
- `messages` follows the OpenAI / Anthropic-style chat-completions shape:
  a list of `{role, content}` records where `role` is one of `"system"`,
  `"user"`, `"assistant"`. Other roles and attachment payloads are rejected.
- Requests allow 1–256 messages, at most 262,144 characters per message and
  512 KiB total UTF-8 prompt content. Model names contain 1–256 characters.
- `max_tokens` must be an integer from 1 through 8192; default 1024.
- `temperature` is finite in [0, 2] (default 0); `top_p` is finite in (0, 1]
  (default 1). Unknown roles are rejected, not silently ignored.
- `stream` (default `true`) selects between chunked and one-shot
  delivery. When `true`, the shim emits one or more
  `LOCAL_LLM_COMPLETION_CHUNK` frames followed by a single
  `LOCAL_LLM_COMPLETION_RESPONSE`. When `false`, only the
  `LOCAL_LLM_COMPLETION_RESPONSE` is emitted (no chunks).

#### `LOCAL_LLM_COMPLETION_CHUNK` (shim → platform, streaming)

```json
{
  "type": "LOCAL_LLM_COMPLETION_CHUNK",
  "id": "req-uuid",
  "ts": 1234567890.1,
  "version": "1.1",
  "payload": {
    "delta": "The user spent ",
    "finish_reason": null
  }
}
```

Multiple frames per request, sharing the request's `id`. Every chunk
EXCEPT the final one MUST have `finish_reason: null`. The shim emits a
final chunk with `finish_reason` set to one of `"stop"`, `"length"`, or
`"content_filter"`, immediately followed by a
`LOCAL_LLM_COMPLETION_RESPONSE` carrying metadata and empty content.
Output deltas are limited to 8 KiB each and 256 KiB in aggregate. The dormant
handler closes its backend stream on 30-second idle/300-second total timeout,
oversize output, send failure, or task cancellation. End-to-end cancellation
across the platform protocol is still a prerequisite for activation.

#### `LOCAL_LLM_COMPLETION_RESPONSE` (shim → platform, terminal frame)

```json
{
  "type": "LOCAL_LLM_COMPLETION_RESPONSE",
  "id": "req-uuid",
  "ts": 1234567890.5,
  "version": "1.1",
  "payload": {
    "content": "The user spent ...",
    "finish_reason": "stop",
    "tokens": {"prompt": 42, "completion": 384, "total": 426},
    "duration_seconds": 4.2
  }
}
```

`content` is empty after streamed deltas, or the bounded assembled output for
non-streamed requests. The example above represents a non-streamed response.
`tokens` is best-effort — backends that don't report token counts MAY
return `null` for any of the three sub-fields. `duration_seconds` is
wall-clock backend time, not the platform's round-trip.

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

The shim checks raw screenshot bytes against
`HELLO_ACK.max_file_size_bytes` before base64 encoding. Oversized captures
return `FILE_TOO_LARGE` instead of building an envelope that exceeds the relay
limit.

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
virtual-display pixels** regardless of any prior region screenshot. On a
HiDPI display that means logical units (points on macOS). To turn a
point in a screenshot into a coordinate, divide it by the screenshot's
`display_scale`, then add its `meta.origin`.
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
- `FILE_TOO_LARGE` — file, screenshot, clipboard, or command output exceeded
  `HELLO_ACK.max_file_size_bytes`
- `DEPENDENCY_MISSING` — a runtime dep needed for the op (pyautogui,
  Pillow, xclip, etc.) isn't installed on the shim host
- `DIRECTORY_REFERENCE_INVALID` — browse/reference pair is malformed,
  expired, consumed, or belongs to a prior control connection
- `DIRECTORY_UNAVAILABLE` — selected directory disappeared, changed into a
  link/junction, or is no longer accessible
- `ROOT_GRANT_INVALID` — restore/activation grant is malformed, has an invalid
  HMAC or machine/session binding, or its directory fingerprint changed
- `SESSION_NOT_ATTACHED` — activation was requested before folder attachment
- `SESSION_ALREADY_ACTIVE` — root replacement was requested while the child
  data runtime was active; detach first
- `SESSION_REVISION_MISMATCH` — activation/restore supplied a stale or
  conflicting root revision
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
- `MODEL_NOT_AVAILABLE` — `LOCAL_LLM_COMPLETION.model` not present in
  the `HELLO.local_llm_models` advertisement, OR the backend reports
  the model is no longer loaded (e.g. Ollama 404 on `/api/chat`). The
  caller MAY retry with a different model or fall back to cloud-side
  completion. `details.requested_model` and `details.available_models`
  are populated when known. See [LOCAL_LLM.md](LOCAL_LLM.md).
- `LOCAL_LLM_BUSY` — the local LLM backend is already serving another
  request and only supports one in flight at a time (Ollama serializes
  by default). The caller SHOULD retry with backoff or fall back. See
  [LOCAL_LLM.md](LOCAL_LLM.md).
- `LOCAL_LLM_FAILED` — the local LLM backend crashed, the HTTP call
  failed, or the connection to the backend was refused. `details.backend_error`
  carries the raw error string for debugging; the platform-side error
  translator surfaces a clean user-facing message. See
  [LOCAL_LLM.md](LOCAL_LLM.md).

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
  "version": "1.1",
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
"do not retry without operator action". Unexpected clean closes still wait
for the base backoff before reconnecting, preventing a hot loop.

> The platform-side `ShimConnectionManager` change that actually emits
> `SESSION_REVOKED` and close-code 4427 lives in a separate ticket; this
> section is the contract the shim implements on the receive side.

---

## Concurrency

The platform may send multiple requests before receiving responses (pipelined). The shim
assigns each request its own async task and responds with matching `id` when complete.
Max concurrent requests: `HELLO_ACK.max_concurrent` (default 4).

---

## Idempotency

The platform's auto-retry layer (transient WS drops, transport errors,
transient `LOCAL_LLM_BUSY` / `SHIM_OVERLOADED`) re-issues failed
requests. Each op is classified as either **idempotent** (safe to
re-issue with the same payload) or **non-idempotent** (re-issue may
double-effect; the platform must surface to the user instead of silently
retrying).

| Op | Idempotent? | Notes |
|---|---|---|
| `HELLO` / `HELLO_ACK` | yes | Handshake; replayable. |
| `PING` / `PONG` | yes | Keepalive. |
| `EXECUTE_COMMAND` | **no** | A shell command may have side effects (rm, mv, push). Platform MUST NOT auto-retry. |
| `FILE_READ` | yes | Pure read; same path → same bytes (modulo concurrent writers). |
| `FILE_STAT` | yes | Pure read. |
| `FILE_LIST` | yes | Pure read. |
| `FILE_WRITE` | yes | Last-write-wins; replay produces same file state. |
| `FILE_DELETE` | yes | Re-delete with `missing_ok: true` is a no-op; the platform retries with `missing_ok: true`. |
| `FILE_MOVE` | **no** | Source path is gone after success; replay → `PATH_NOT_FOUND`. Platform surfaces, doesn't auto-retry. |
| `SCREENSHOT_REQUEST` | yes | Pure read of current display. |
| `INPUT_ACTION` | **no** | Mouse clicks / key presses have side effects. Never auto-retry. |
| `WINDOW_FOCUS` / `APP_LAUNCH` | **no** | State-mutating UI ops. |
| `CLIPBOARD_READ` | yes | Pure read. |
| `CLIPBOARD_WRITE` | yes | Last-write-wins. |
| `LOCAL_LLM_COMPLETION` | yes | Same prompt almost certainly produces different output (sampling), but for retry semantics the platform CAN safely re-issue — the user-facing turn is non-destructive and the worst case is "you got a slightly different answer". The auto-retry layer treats it as idempotent. |
| `REQUEST_RECORDING_CONSENT` | **no** | Displays a local user prompt. Never auto-retry or duplicate prompts. |
| `START_RECORDING` | **no** | Mints a fresh `recording_id` and begins capture; a re-issue while one is active gets `RECORDING_ALREADY_ACTIVE`. Platform surfaces, doesn't auto-retry. |
| `STOP_RECORDING` | **no** | Finalizes + secure-state-changes the session; re-issue after success → `RECORDING_NOT_FOUND`. |
| `RECORDING_FETCH` | yes | Pure read of a finalized recording (post-redaction). Safe to re-issue. |
| `APPLY_RECORDING_REVIEW` | yes | Reapplying the same step removals/redactions produces the same reviewed recording. |
| `RECORDING_STEP` | n/a | Unsolicited co-pilot-mode stream (shim → platform), modeled like `STATUS`: not acked, exempt from in-flight / `max_concurrent` accounting, and never part of the auto-retry layer. |

Receivers MUST NOT rely on the platform retrying any specific op — this
table just disciplines the platform's auto-retry layer. Application
code that needs explicit retry semantics (e.g. an agent that wants to
re-run a command after it failed) issues a fresh request with a new
`id`.

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
pending_capacity = max_concurrent - admitted_or_in_flight_after_this_response
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
  "version": "1.1",
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
`attempt` counts failed connects in a row and goes back to 0 once a
connection completes its handshake, so a session that drops after
running reconnects promptly. On reconnect, shim sends a new `HELLO`. Platform re-issues `HELLO_ACK` with same session.
Any in-flight requests at disconnect time are considered failed; platform retries if safe.
