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
  "version": "2.0",
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
| Same major, different minor | Same major, `min(shim_minor, platform_minor)`. Each side sends only what the negotiated version defines (see below). |
| Different majors | **Hard error.** The side that notices closes the WebSocket with close code **4426**. A close reason holds at most 123 bytes (RFC 6455 §5.5), so it is short: the shim sends compact JSON, `{"error":"PROTOCOL_VERSION_MISMATCH","shim_max":"...","platform_max":"..."}` (just the `error` field if the versions don't fit), and logs the upgrade hint; the platform sends a short text reason. The shim MUST log the mismatch and MUST NOT auto-reconnect until restarted — this avoids hot-reconnect storms against an incompatible peer. |

### Unknown fields

Within a major, new message types and new values of an open set (a
`reason`, say) are forward-compatible: a receiver drops or tolerates what it
doesn't know, as their sections say. A payload field is not. The shim's
payloads forbid fields they don't define, so a frame carrying one fails
validation and is logged and dropped, never acted on: an older shim that
ignored a newer field could act without a constraint the platform meant it
to apply. The platform learns the shim's maximum from HELLO before it
answers, so it must not send a newer minor's fields to an older shim. A
frame whose objects and arrays nest more than 64 levels deep is logged and
dropped the same way; no message comes close. So is one holding `NaN`,
`Infinity` or a number too large for a double (`1e999`, or a whole number
309 digits long): JSON has none of them, though some encoders write them.
The shim's log says where such a frame failed and how
(`COMPUTER_TOOL_CALL.payload.tool: missing`), never what it held, so the
text a malformed computer tool call was to type stays out of it. A part of
that place the protocol doesn't define, such as the name of a field the
message doesn't have or a key of a map like `env`, is written `*`
(`PING.payload.*: extra_forbidden`), as the frame chose it.

### Frame-level `version`

All non-HELLO frames SHOULD set `version` to the negotiated value, but
receivers MUST treat the HELLO-time negotiation as truth. A receiver that
sees a non-HELLO frame with a different *minor* MUST process it normally;
a frame with a different *major* MUST be dropped and logged loudly (and
on the platform side, the session SHOULD be torn down with code 4426).

### Current versions

| Side | Maximum | Notes |
|---|---|---|
| Shim | `2.0` | `autogpt_local_executor.protocol.VERSION` |
| Platform | `2.0` | Mirror this constant in the platform repo. |

`2.0` is the first released version; nothing older is spoken.

---

## Message Types

### Handshake

#### `HELLO` (shim → platform, on connect)
```json
{
  "type": "HELLO",
  "id": "uuid",
  "ts": 1234567890.0,
  "version": "2.0",
  "payload": {
    "shim_version": "0.1.0",
    "protocol_version": "2.0",     // max wire version this shim supports
    "machine_id": "hostname-uuid4",
    "platform": "darwin",          // "darwin" | "linux" | "windows" | "wsl2"
    "arch": "arm64",               // "x86_64" | "arm64" (normalized; see below)
    "screen_resolution": [2560, 1440],   // main display's logical size; null without computer_use
    "capabilities": [
      "directory_browse",          // control WS only
      "shell",                     // optional: explicit local --enable-shell opt-in
      "files",                     // always present
      "computer_use",              // optional: cua-driver installed and started
      "local_llm",                 // optional: ollama running
      "hardware_serial",           // optional: pyserial available
      "hardware_usb",              // optional: pyusb available
      "hardware_gpio"              // optional: RPi.GPIO available
    ],
    "allowed_root": null,        // control WS; canonical root on a data WS
    "computer_use_driver": {"name": "cua-driver", "version": "0.38.0", "runtime": "direct"},
    "computer_use_tools": ["click", "get_window_state", "list_windows", "..."],
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
  "version": "2.0",
  "payload": {
    "session_id": null,          // null on control; exact ID on data WS
    "connection_id": "conn-7f3a", // opaque presence generation; required on control
    "protocol_version": "2.0",     // max wire version this platform supports
    "granted_capabilities": ["directory_browse"],  // control; a data WS gets its subset, e.g. ["shell", "files"]
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
session ID. On the base control connection it must carry a `session_id`
that is null or empty (named, not left out), an opaque `connection_id`, a
2.x protocol version (named in `protocol_version`, not left to its
default), and the `directory_browse` capability. A control daemon fails
terminally on an ACK it can't use. One that doesn't parse closes with 4400,
whatever else it lacks; that includes an ACK without `granted_capabilities`,
which every ACK must have. One that parses but lacks a control field above
closes with 4426. The shim enforces the granted subset on every subsequent
request.

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
  "version": "2.0",
  "payload": {"browse_id": null, "directory_ref": null}
}
```

```json
{
  "type": "DIRECTORY_LIST_RESPONSE",
  "id": "request-uuid",
  "ts": 1234567890.1,
  "version": "2.0",
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
  "version": "2.0",
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
  "version": "2.0",
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
platform transparently restores the child when the next turn begins. A child
with a request still queued or running is not idle, however long that request
takes. The stop is audited (`IDLE_SESSION_DETACH`) before it happens, and from
the moment the daemon decides on it the child refuses requests with
`SHIM_OVERLOADED` (`details.reason: "detaching"`), so none starts only to be cut
off. Nothing ran, so a refused request can be sent again once the child is
restored: the auto-retry layer re-sends only an idempotent one (see
[Idempotency](#idempotency)), and a `COMPUTER_TOOL_CALL`, like every other
non-idempotent request, goes back to the model, which decides. If the audit
record can't be written, the child keeps running.

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
  "version": "2.0",
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
  "version": "2.0",
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
transport ceiling to account for worst-case JSON escaping. A command is done
when it exits: a program it started that still holds its stdout or stderr
(a server started in the background, say) keeps running, and what it writes
there is read for 2 more seconds, with `timed_out: false`.

---

### File Operations

A file operation's `path` (`src` and `dst` for a move), like
`EXECUTE_COMMAND`'s `cwd`, is normally absolute. A relative one is taken
from `allowed_root`, as the platform's own tools take it, never from the
directory the shim happens to run in; `~` is the user's home. Either way it
must resolve inside `allowed_root`.

#### `FILE_READ` (platform → shim)
```json
{
  "type": "FILE_READ",
  "id": "req-uuid",
  "ts": 1234567890.0,
  "version": "2.0",
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
  "version": "2.0",
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
  "version": "2.0",
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

**Replacing, not truncating**: the content goes to a new file beside the
target, which is then renamed over it, so a write that fails part way (a
full disk, say) leaves the old contents whole (on the in-place fallback
below, it leaves a partial file instead). An existing file keeps its
owner and group, and its mode, except set-user-ID and set-group-ID, which a
write drops so that new contents never run with the file owner's privileges
(Linux drops them for an unprivileged writer; the shim drops them whatever
it runs as, and refuses the write when it can't, as for someone else's file
written in place); a new one gets the usual default (0o666 less the umask).
The new file beside an existing one is owner-only until it has that mode,
so no other account can open it, and read the new contents, before then. An existing file with more than
one hard link is refused with `PATH_OUTSIDE_ALLOWED_ROOT`, as `FILE_READ`
refuses one: its other names may be outside the root, and the shim can't
tell, so the write neither succeeds nor replaces it. Because a target the
write does replace gets a new inode, extended attributes on the old file
don't carry over, and the new file has the permissions the folder gives a
new file. So a file whose ACL differs from that (entries of its own, or a
folder whose default or inherited entries came after it) is written in
place, keeping its own: a POSIX ACL on Linux, where the mode's group bits
are the ACL's mask, so a file that let another account in and kept its own
group out would come back open to the group; the ACL an NFSv4 or SMB mount
reads from its server; a macOS ACL; and on Windows the owner and DACL
(inheritance turned off, say). The shim compares the two before any
content is written, and writes in place when it can't read them. Written in
place, a file's mode is changed only to drop set-user-ID or set-group-ID:
on an NFS mount even a chmod that changes nothing reaches the server, and
a server that drops ACL entries on a chmod would drop the file's. A Windows
folder WSL2 reaches through `/mnt` shows no ACL to compare, so a file
there that is replaced takes the folder's. Where the directory takes no new entries, (on
Windows) another program has the file open, or the shim may not give the new
file the old one's owner and group (a file another account owns, a group the
shim isn't in), the shim overwrites the file in place instead:
it truncates the file and writes the content, so a write that fails part way
there leaves a partial file, and the old contents are gone.

#### `ACK` (shim → platform, for writes and fire-and-forget ops)
```json
{
  "type": "ACK",
  "id": "req-uuid",
  "ts": 1234567890.1,
  "version": "2.0",
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
  "version": "2.0",
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
  "version": "2.0",
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
  "version": "2.0",
  "payload": {
    "path": "/Users/alice/autogpt-workspace",
    "glob": "*.csv",               // null = list everything
    "recursive": false,
    "include_hidden": false,
    "max_entries": 1000            // at most 10,000; truncated flag in response
  }
}
```

#### `FILE_LIST_RESPONSE` (shim → platform)

```json
{
  "type": "FILE_LIST_RESPONSE",
  "id": "req-uuid",
  "ts": 1234567890.1,
  "version": "2.0",
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

An entry's `size_bytes` and `mtime` are its own (an `lstat`). For a symlink,
`is_file` and `is_dir` describe its target only when the target is inside
the allowed root; for one outside, missing or unreadable, both are false,
so a listing never says whether a path outside the root exists. A recursive
listing never enters a symlink or a Windows junction. An entry whose name
isn't valid UTF-8 (possible on Linux) is left out: JSON can't carry it, and
no request could name it.

`truncated` is true only when something was left out: a match past
`max_entries` (which is capped at 10,000), a match that would take the
entries past the reply's size budget (16 MiB less 64 KiB, so the reply
stays under the frame limit), a directory entry past the walk's scan
budget of 100,000 entries, or the entries of a subfolder the walk couldn't
read. A folder asked for that can't be read is an error (`INTERNAL_ERROR`,
or `PATH_NOT_FOUND` once it's gone), not an empty listing. Exactly
`max_entries` matches come back with `truncated: false`. To see the rest,
list a narrower path or glob. The scan
budget counts entries as each directory is read, so a directory with
millions of entries is never read whole.

Glob is matched with `fnmatch.fnmatchcase`. On case-insensitive
filesystems (macOS APFS default, Windows NTFS default), matching is
case-insensitive; on case-sensitive Linux ext4, it's case-sensitive. The
shim tells which by looking one of the listed folder's entries up again
with its ASCII letters' case swapped: a case-insensitive filesystem
resolves that name to the same file, and a case-sensitive one to none or
another. This only reads, so a read-only mount is told apart too. The
shim looks at no more than the folder's first 100 entries, in the order
the OS lists them, and passes over an entry whose name has no ASCII
letter, a file with another hard link (which could hold both names), an
entry with no inode number or whose own stat fails, and one whose swapped
name can't be looked up for a reason other than it not being there (a
permission error, say). When none of them serves (an empty folder, say),
the OS's default decides: case-insensitive on macOS and Windows,
case-sensitive elsewhere, even where an entry past the first 100 would
have shown that the folder folds case. Nothing is written to the folder to
tell, and each listing looks again. The listed folder's rule holds for the
whole walk. The glob is matched against the **basename**, not the full
path; combine with `recursive: true` to descend.

#### `FILE_DELETE` (platform → shim)

Used in place of shell `rm` / `del`.

```json
{
  "type": "FILE_DELETE",
  "id": "req-uuid",
  "ts": 1234567890.0,
  "version": "2.0",
  "payload": {
    "path": "/Users/alice/autogpt-workspace/old.csv",
    "recursive": false,            // required true to delete a non-empty dir
    "missing_ok": false            // if true, no error when path doesn't exist
  }
}
```

Responds with `ACK`. A symlink is removed itself, never its target, but
the path must still resolve inside the root: a link pointing outside it is
refused with `PATH_OUTSIDE_ALLOWED_ROOT` and left in place (CROSS_PLATFORM.md).
A delete is permanent: the shim has no recycle bin, backup or rollback
(SECURITY.md).

A symlink or junction is removed itself; its target is untouched. The allowed
root itself cannot be deleted.

#### `FILE_MOVE` (platform → shim)

Used in place of shell `mv` / `move`. Also covers renames.

```json
{
  "type": "FILE_MOVE",
  "id": "req-uuid",
  "ts": 1234567890.0,
  "version": "2.0",
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
replaced itself; neither target changes. Without `overwrite`, the rename
itself refuses an existing `dst` (`renameat2` with `RENAME_NOREPLACE` on
Linux, `renamex_np` with `RENAME_EXCL` on macOS, a plain rename on
Windows), so there a file created after the shim's check is never
replaced. A filesystem without that flag gets a check followed by the
rename, and there a file another program creates at `dst` between the two
is replaced.

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
  "version": "2.0",
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
  "version": "2.0",
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
handler closes its backend stream on a 30-second idle timeout between chunks
(not before the first, which loading the model and reading the prompt can
delay), the 300-second total timeout, oversize output, send failure, or task
cancellation. End-to-end cancellation
across the platform protocol is still a prerequisite for activation.

#### `LOCAL_LLM_COMPLETION_RESPONSE` (shim → platform, terminal frame)

```json
{
  "type": "LOCAL_LLM_COMPLETION_RESPONSE",
  "id": "req-uuid",
  "ts": 1234567890.5,
  "version": "2.0",
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

Computer use relays [cua-driver](https://github.com/trycua/cua)'s own tools.
The shim runs `cua-driver mcp` and offers a reviewed subset of its tools
under its own policy; the platform lists them, then calls them by name with
the driver's arguments. Coordinates, window IDs and element tokens are the
driver's, scoped to a window and a state snapshot, not a shim-wide screen.
[CUA_DRIVER.md](CUA_DRIVER.md) has the design, the catalog and its policy;
[COMPUTER_USE.md](COMPUTER_USE.md) has setup and behavior per OS.

`HELLO.computer_use_driver` names the driver once it has started:
`{"name": "cua-driver", "version": "0.38.0", "runtime": "direct"}`. The
runtime is always `direct`, a driver process the shim owns; `service`, a
driver relaying to a shared service such as CuaDriver.app, is reserved.
`HELLO.computer_use_tools` lists, sorted, the names of the tools
`COMPUTER_TOOLS_RESPONSE` would return, so the platform can ask for consent
to exactly those without a round trip. `computer_use` is advertised only
once the driver has started.

#### `COMPUTER_TOOLS_REQUEST` (platform → shim)
```json
{"type": "COMPUTER_TOOLS_REQUEST", "id": "req-uuid", "ts": 1234567890.0, "version": "2.0", "payload": {}}
```

#### `COMPUTER_TOOLS_RESPONSE` (shim → platform)
```json
{
  "type": "COMPUTER_TOOLS_RESPONSE",
  "id": "req-uuid",
  "ts": 1234567890.2,
  "version": "2.0",
  "payload": {
    "driver": {"name": "cua-driver", "version": "0.38.0", "runtime": "direct"},
    "instructions": "cua-driver: background GUI automation. ...",
    "tools": [
      {
        "name": "get_window_state",
        "description": "...",
        "input_schema": {"type": "object", "properties": {"pid": {"type": "integer"}, "screenshot_out_file": {"type": "string"}}},
        "annotations": {"readOnlyHint": true},
        "approval_arguments": {"screenshot_out_file": "writes the screenshot to this file"}
      }
    ]
  }
}
```

`instructions` is the driver's own operating guidance with the shim's notes
for this machine appended (clipboard mode, screenshot budget, what is not
offered). The model should see it before it acts. `tools` holds only what
this machine offers: the clipboard tools appear only with
`--enable-clipboard`, and a tool the driver lacks on this OS is absent.
Schemas never show `session`: the shim sets it on every call, one label per
shim session, so each session gets its own agent cursor and cleanup.
cua-driver ends a label left idle for five minutes and refuses later calls
naming it; the shim then moves the session to a new label and sends the
refused call again, once (it never ran). Element tokens and other state
from the ended label are gone, so the model takes a fresh snapshot, as
after any stale state. The
list doesn't change while the shim runs; the platform may cache it per
connection. Listing tools doesn't wait for a running action: it never takes
the turn computer tool calls take. Like any request, though, it needs one of
the session's and the machine's request slots (see
[Concurrency](#concurrency)), and waits up to ten seconds for a machine slot
other chats' work holds, then gets `SHIM_OVERLOADED`.

#### `COMPUTER_TOOL_CALL` (platform → shim)
```json
{
  "type": "COMPUTER_TOOL_CALL",
  "id": "req-uuid",
  "ts": 1234567890.0,
  "version": "2.0",
  "payload": {
    "tool": "run_steps",
    "arguments": {"pid": 844, "window_id": 10725, "steps": [{"tool": "click", "args": {"element_token": "s0000002a:14"}}], "observe": {}},
    "timeout_seconds": 120
  }
}
```

`tool` matches `^[a-z][a-z0-9_]{0,63}$`. `arguments` follows the tool's
`input_schema`. Calls from every session on a machine take turns: there is
one screen. `timeout_seconds` (1-600, default 120) bounds the whole call,
from when the shim receives it: the wait for its turn counts, and so does
the wait for cua-driver to start, on the first call or after it stopped. A
call whose deadline passes while it waits never reaches the driver and gets
`COMMAND_TIMEOUT` with `details.started: false`; nothing ran, so it can be
sent again. A start that outlasts it goes on, for the calls after it. At the deadline of a call the driver is running, the shim stops
the driver, which can't cancel a call it is running, and answers
`COMMAND_TIMEOUT` with `details.started: true`, though an action the driver
already made stays made. The next call starts the driver afresh, without the
element tokens and sessions of the old one. Once the driver is done, the call
lets the next one have the computer and writes its audit record, and its
answer goes out when the record is in: an audit log held up delays that
answer, not other calls.

`approved_arguments` (at most 8, default empty) names the tool's approval
arguments the platform approved for this call: usually the person, on an
approval card, but a chat that runs Unsupervised approves without asking,
and a standing rule the person set on an earlier card ("from now on")
approves later calls of that kind whatever their arguments. The shim can't
tell which and trusts the field. An argument in a tool's
`approval_arguments` can reach the disk (`screenshot_out_file`, macOS
`click.debug_image_out`, clipboard `file_path` / `image_path`,
`browser_set_input_files.files`, and
`run_script.script`, whose calls inside the driver can save a screenshot to
a file or start a program where the shim can't check them) or runs a
program out of sight
(`launch_app`'s `launch_path`, `path`, `additional_arguments`, `urls` and
`webkit_inspector_port`); a call that
uses one without naming it there gets `APPROVAL_REQUIRED` with
`details.tool` and `details.argument`, and nothing reaches the driver. Only
an approval argument of the called tool may be named.

The shim refuses, with `FEATURE_NOT_SUPPORTED` and `details.tool`, a tool it
doesn't offer, and, with `details.argument` too, an argument it never
accepts: `check_permissions` with `prompt: true`; any tool's approval
argument inside a `run_steps` step, whatever tool the step names, or its
`observe`; a `run_steps` step that is a `launch_app` (`details.argument`
`steps[i]`), since only a launch called on its own has its name checked
against the installed apps; and a `launch_app` `name`, `bundle_id` or
`aumid` that isn't an app's name or id (a path, a URL scheme, a leading
`-`, a control character, on Linux a name whose first word is a program,
a program's own name included, or, unless `urls` or `launch_path`
is given, one that isn't an installed app's name or desktop-file id in
`list_apps`, and on Windows a `bundle_id` that isn't an AUMID or a `name` no
installed app in `list_apps` would match).

#### `COMPUTER_TOOL_RESULT` (shim → platform)
```json
{
  "type": "COMPUTER_TOOL_RESULT",
  "id": "req-uuid",
  "ts": 1234567890.4,
  "version": "2.0",
  "payload": {
    "tool": "run_steps",
    "is_error": false,
    "content": [
      {"type": "text", "text": "1/1 steps ok ..."},
      {"type": "image", "data": "iVBORw0...", "mime_type": "image/png"}
    ],
    "structured_content": {"ok": true, "executed": 1}
  }
}
```

The driver's own answer, including its refusals and failed actions:
`is_error: true` is a result the model reads and acts on (a stale element
token, a refused background delivery), not a shim failure. Shim failures
come back as `ERROR`: `DEPENDENCY_MISSING` (no usable cua-driver),
`INTERNAL_ERROR` (the driver stopped), `COMMAND_TIMEOUT`,
`CLIPBOARD_CONCEALED`, `FEATURE_NOT_SUPPORTED`, `APPROVAL_REQUIRED`.

Content is text and image blocks; another kind of block becomes a text
note. Images are base64 as the driver produced them, at most
`max_screenshots_per_minute` a minute (default 10) per machine; past that,
or when the result would exceed `HELLO_ACK.max_file_size_bytes` or the
WebSocket limit, images are left out and a text block says why. `structured_content` is the driver's
`structuredContent`, dropped only when the result can't otherwise fit.
The limit holds for the whole `payload` as it is sent, its notes
included: what is left of the first text block is cut to fit beside a
note that says so, and a limit too small for even that note gets the
smallest result there is, `tool` and `is_error` with an empty `content`
and `structured_content: null`.

`clipboard_read` returns text under the shim's rules: by default only text
this shim wrote in the last 30 seconds (else `CLIPBOARD_CONCEALED` with
`details.reason` `writeback_only` or `writeback_overwritten`); with
`--enable-clipboard-read-foreign` anything except content a password
manager marked private (`concealed_type`, with `details.marker`), and never
a copy made while it was read (`changed`).

---

### Error

#### `ERROR` (either direction)
```json
{
  "type": "ERROR",
  "id": "req-uuid",
  "ts": 1234567890.0,
  "version": "2.0",
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
- `COMMAND_TIMEOUT` — a `COMPUTER_TOOL_CALL` ran out of `timeout_seconds`
  (`details.tool`, `details.timeout_seconds`). `details.started` says
  whether anything ran: `false` when the call was still waiting for its
  turn or for cua-driver to start, so nothing reached the driver; `true`
  when the driver was running it, so the shim stopped the driver, and what
  the call already did stays done. Neither is retried automatically (see
  [Idempotency](#idempotency)); the model decides, and may send a call that
  didn't start again. `EXECUTE_COMMAND` doesn't use this code: on timeout it
  returns `COMMAND_RESULT` with `timed_out: true`, and `exit_code` is the
  stopped process's own status (on POSIX the negated signal that ended it,
  such as -15 or -9), or -1 when there is none to read, so callers branch on
  the `timed_out` flag, not the code.
- `SHELL_NOT_AVAILABLE` — `shell` selector requested a shell not installed on this OS
- `UNSUPPORTED_ARCH` — HELLO `arch` value not recognized
- `CAPABILITY_NOT_GRANTED` — requested capability not in granted_capabilities
- `AUTH_FAILED` — token invalid or expired
- `SHIM_OVERLOADED` — too many concurrent requests (exceeded `max_concurrent`)
- `INTERNAL_ERROR` — unexpected shim error
- `FILE_TOO_LARGE` — file or command output exceeded
  `HELLO_ACK.max_file_size_bytes`
- `DEPENDENCY_MISSING` — a runtime dependency the op needs isn't installed
  on the shim host: no cua-driver, or one of a version the shim wasn't
  reviewed against
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
- `PERMISSION_PENDING` — an OS-level permission computer use needs (macOS
  Accessibility / Screen Recording) isn't granted. Distinct from
  `CAPABILITY_NOT_GRANTED`. See [COMPUTER_USE.md](COMPUTER_USE.md).
- `FEATURE_NOT_SUPPORTED` — a computer tool this machine doesn't offer, or
  an argument it never accepts (`details.tool`, `details.argument`). See
  [CUA_DRIVER.md](CUA_DRIVER.md).
- `APPROVAL_REQUIRED` — a computer tool call used one of the tool's
  approval arguments without naming it in `approved_arguments`
  (`details.tool`, `details.argument`). Those arguments reach the disk or
  start a program out of sight (see `COMPUTER_TOOL_CALL` above); nothing
  reached the driver.
- `CLIPBOARD_CONCEALED` — `clipboard_read` found content this session may
  not read; `details.reason` says why. See [COMPUTER_USE.md](COMPUTER_USE.md).
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
depends on `code`.

---

### Keepalive

#### `PING` / `PONG`
```json
{ "type": "PING", "id": "uuid", "ts": 1234567890.0, "version": "2.0", "payload": {} }
{ "type": "PONG", "id": "same-uuid", "ts": 1234567890.01, "version": "2.0", "payload": {} }
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
  "version": "2.0",
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

The shim also cancels the session's requests and, on the machine's
connection, stops its chats' sessions, at once rather than after the audit
record, which a held-up audit writer can delay. A side effect already running
in a thread can't be stopped and may never finish, so the audit record and
the close come first; the shim then waits up to 10 seconds for such work,
lets go of any still running, and sends none of its replies. A fatal close
code (4426, 4427 or 4428, below) stops the session's work the same way.

Receivers MUST tolerate unknown future `reason` values (forward-compatible
minor extension); the table above is the 2.0 set.

#### WS close-code table

| Code | Label | Meaning | Sender | Shim auto-reconnect? |
|---|---|---|---|---|
| 4426 | `PROTOCOL_VERSION_MISMATCH` | Major-version disagreement in HELLO/HELLO_ACK. | platform | **No** — restart required after upgrade. |
| 4427 | `SESSION_TAKEN_OVER` | Another shim connected for this session. Sent to the connection it replaces, after `SESSION_REVOKED` (`another_shim_connected`); under the ownership policy above, also to a newcomer from a different `machine_id`, which is rejected. Today's platform replaces the older connection whichever machine the newcomer is on. | platform | **No** — the takeover, or the rejection, is intentional. |
| 4428 | `SESSION_REVOKED` | User revoked this session in the platform UI (or related admin action). | platform | **No** — auth gone, would just 401. |
| 4429 | `PLATFORM_SHUTDOWN` | Platform is going down (graceful). | platform | **Yes**, after the reconnect backoff: the platform comes back, and halting would leave the machine offline until someone restarted the shim. The backoff grows while the platform is still down; the shim doesn't probe the platform's health first. |

Codes below 4426 follow IETF/RFC semantics (1000 normal, 1011 server
error, etc.) and the shim DOES auto-reconnect with exponential backoff —
only 4426, 4427 and 4428 in the table above are treated as fatal "do not
retry without operator action"; after 4429 the shim reconnects too, as the
table says. A clean close (1000, 1001) counts as any other drop (see
[Reconnection](#reconnection)): one right after `HELLO_ACK` keeps the
backoff growing, preventing a hot loop.

> The platform sends `SESSION_REVOKED` and close code 4427 to the connection
> a newer one replaces. It doesn't yet reject a newcomer from a different
> `machine_id`; for that part of the policy, this section is the contract
> the shim implements on the receive side.

---

## Concurrency

The platform may send multiple requests before receiving responses (pipelined). The shim
assigns each request its own async task and responds with matching `id` when complete.
Max concurrent requests: `HELLO_ACK.max_concurrent` (default 4).
The chats on a machine also share the control connection's `max_concurrent`
slots. A request waits up to ten seconds for one, then gets
`SHIM_OVERLOADED` with `details.reason: "machine_busy"`. A slot stays taken
until its work ends, including work still running in a thread after its
session ended.
`COMPUTER_TOOL_CALL`s take no machine slot: those from every session on a
machine take turns, since they share one screen, so calls waiting their turn
would otherwise hold every slot while a long one ran. They still count
against their own session's `max_concurrent`, while they wait for their turn
too: a call that arrives with its session at that limit gets
`SHIM_OVERLOADED` (`details.max_concurrent`) at once, before it reaches the
computer's queue. Each waits for its turn no longer than its own deadline;
one whose deadline comes first gets `COMMAND_TIMEOUT` with
`details.started: false`.

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
| `COMPUTER_TOOLS_REQUEST` | yes | Pure read of the tool list. |
| `COMPUTER_TOOL_CALL` | **no** | Most tools click, type or launch. Never auto-retry, even a read-only one: the model decides what to do after a failure. |
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
  "version": "2.0",
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
and the spec's discriminated union just gains it within a major.

> The platform-side consumer that reads `pending_capacity` / `STATUS`
> and throttles issuance lives in a separate ticket; this section is
> the contract the shim implements on the emit side.

## Reconnection

Shim uses exponential backoff: `min(2^attempt * 1s, 60s) + jitter(0-5s)`.
`attempt` counts failed connects in a row and goes back to 0 once a
session that completed its handshake has stayed up for a minute, so a
session that drops after running reconnects promptly, while one the
platform ends right after `HELLO_ACK` (closing with 1011 while
overloaded, say, or cleanly) counts as a failed connect and the backoff keeps growing. On reconnect, shim sends a new `HELLO`. Platform re-issues `HELLO_ACK` with same session.
The shim also closes a connection itself (1000) to send a new `HELLO`
when one it sent left `computer_use` out only for want of the desktop's
display (a Linux systemd service started at login, before the desktop
exported it) and the display is up now. It waits until nothing is under
way on that connection, and on the machine connection, whose next session
stops every chat's, until no chat's work is either. From the moment it
decides, that session, and on the machine connection every chat's, refuses
requests with `SHIM_OVERLOADED` (`details.reason: "reconnecting"`) until the
next connection (a chat's until the machine's next session stops it, even if
the chat reconnects first), so none starts during the close and the backoff
only to be cut off; nothing ran, so each can be sent again, as after an idle
detach.
Any in-flight requests at disconnect time are considered failed; platform retries if safe.
