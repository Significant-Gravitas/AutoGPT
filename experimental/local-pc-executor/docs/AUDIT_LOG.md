# Audit Log — Format & Tamper-Evidence

> **Status**: Draft v0.1 — subject to change
>
> Every operation the shim executes is appended to a local audit log.
> The log is the user's primary forensic tool: "what did the platform
> ask my machine to do, and when?" This doc pins the format, the
> tamper-evidence scheme, and the rotation policy so consumers (the
> user, the platform-side viewer, third-party tools) can rely on a
> stable contract.

## Location

Per [CROSS_PLATFORM.md → Audit log location](CROSS_PLATFORM.md#audit-log-location):

| OS | Path |
|---|---|
| macOS | `~/Library/Logs/autogpt-local-executor/audit.log` |
| Linux | `$XDG_STATE_HOME/autogpt-local-executor/audit.log` (fallback `~/.local/state/...`) |
| Windows | `%LOCALAPPDATA%\autogpt-local-executor\logs\audit.log` |
| WSL2 | Linux path |

Permissions: created `0600` on POSIX (owner-only); ACL'd to the running
user on Windows. The shim refuses to start if the audit log can't be
opened for append.

## Format

Newline-delimited JSON (JSONL). One record per line. Each record is a
single JSON object with this shape:

```json
{
  "ts": 1712345678.123,
  "session_id": "session-uuid",
  "machine_id": "machine-uuid",
  "shim_version": "0.1.0",
  "request_id": "msg-uuid",
  "op": "EXECUTE_COMMAND",
  "details": { ... },
  "result": {
    "ok": true,
    "exit_code": 0,
    "duration_ms": 124,
    "error_code": null
  },
  "seq": 4218,
  "prev_hmac": "f4a2…",
  "hmac": "9c81…"
}
```

| Field | Type | Notes |
|---|---|---|
| `ts` | float (unix epoch seconds) | Record timestamp, monotonic-rounded |
| `session_id` | string | The platform session that issued the op |
| `machine_id` | string | This shim's stable machine identifier |
| `shim_version` | string | semver of the shim writing the line |
| `request_id` | string \| null | Wire envelope `id` for traceability (null for shim-internal events) |
| `op` | string | One of the message types from PROTOCOL.md plus a few shim-internal events (see below) |
| `details` | object | Op-specific payload (see "Per-op fields") |
| `result.ok` | bool | True if the op completed without error |
| `result.exit_code` | int \| null | For EXECUTE_COMMAND only |
| `result.duration_ms` | int | Op elapsed time |
| `result.error_code` | string \| null | ErrorCode enum value on failure (see PROTOCOL.md error codes) |
| `seq` | int | Monotonic per-log-file sequence number, starts at 1, resets on rotation |
| `prev_hmac` | string \| null | HMAC of the previous record (null for the first record in a file) |
| `hmac` | string | HMAC of THIS record (see "Tamper-evidence chain") |

### Per-op `details`

| `op` | `details` shape |
|---|---|
| `EXECUTE_COMMAND` | `{command, argv, shell, cwd, env_keys[], timeout_seconds}` — env *keys* only, never values |
| `FILE_READ` | `{path, encoding, offset, length, size_bytes_returned}` |
| `FILE_WRITE` | `{path, encoding, size_bytes_written, create_parents}` |
| `FILE_STAT` | `{path, follow_symlinks}` |
| `FILE_LIST` | `{path, glob, recursive, include_hidden, max_entries, entries_returned}` |
| `FILE_DELETE` | `{path, recursive, missing_ok}` |
| `FILE_MOVE` | `{src, dst, overwrite}` |
| `SCREENSHOT_REQUEST` | `{monitor, quality, image_bytes_returned}` — no image content |
| `INPUT_ACTION` | `{action, coordinate, key, direction, clicks}` — `text` payload is redacted to length |
| `HELLO` | `{platform, arch, capabilities, allowed_root}` |
| `HELLO_ACK` | `{granted_capabilities, max_concurrent, max_file_size_bytes, command_timeout_seconds}` |
| Shim-internal | See below |

### Shim-internal events

These have `request_id: null` and document state changes the platform
didn't trigger:

| `op` | When |
|---|---|
| `SHIM_START` | Daemon initialised, audit log opened |
| `SHIM_STOP` | Daemon exit (graceful or via signal) |
| `WS_CONNECTED` | WebSocket upgrade succeeded |
| `WS_DISCONNECTED` | WebSocket loop ended (carries `details.reason`) |
| `TOKEN_REFRESHED` | OAuth refresh token used |
| `CONFIG_RELOADED` | `HELLO_ACK` rewrote runtime config |
| `JAIL_VIOLATION` | Path-jail rejected an op before exec (carries `details.code`) |

### What's NEVER logged

- Wire payload content for FILE_READ / FILE_WRITE (only sizes).
- Screenshot image bytes (only dimensions + byte count).
- `INPUT_ACTION.text` payload (only its length, to avoid leaking what
  the LLM is typing into password fields).
- Environment variable VALUES (only keys, so the user can see "GH_TOKEN
  was set" without seeing the token itself).
- Refresh tokens or access tokens.

## Tamper-evidence chain

Each record carries an HMAC of its own JSON form chained to the
previous record's HMAC. Truncating the log, deleting a record, or
modifying any field breaks the chain at the modification point.

### Keying

A per-machine **audit key** is generated on first run and stored in the
OS keychain (same backend as the OAuth tokens — see CROSS_PLATFORM.md
"Keychain / credential storage"). The key is 32 bytes from
`secrets.token_bytes(32)`. It NEVER leaves the machine.

If the keychain is unavailable, the encrypted-file fallback
(`AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE`) holds the audit key alongside the
OAuth tokens.

### Algorithm

```
record_without_hmac = {ts, session_id, machine_id, shim_version,
                       request_id, op, details, result, seq, prev_hmac}
canonical_bytes     = json_canonical(record_without_hmac)
hmac                = HMAC-SHA256(audit_key, canonical_bytes)
record              = {**record_without_hmac, "hmac": hmac.hex()}
```

`json_canonical` is **RFC 8785 JSON Canonicalization Scheme (JCS)**:
sorted keys, no whitespace, normalized numbers. JCS guarantees byte-
identical output across Python / Go / Rust verifiers.

### Verification

A consumer (the user via `autogpt-shim audit verify`, or the platform-
side viewer) reads the log line-by-line:

```
for each record in file:
    actual_hmac = HMAC-SHA256(audit_key, canonical_bytes_without_hmac(record))
    if actual_hmac != record.hmac:
        FAIL "record {seq} tampered"
    if record.prev_hmac != previous_record.hmac:
        FAIL "chain break between {seq-1} and {seq}"
    if record.seq != previous_record.seq + 1:
        FAIL "sequence gap"
    previous_record = record
```

Verification needs the audit key. The user has it on their machine.
The platform does NOT — the platform receives the audit log only when
the user uploads it for support, and the user provides the key
out-of-band.

## Rotation

| Trigger | Action |
|---|---|
| File reaches **64 MiB** | Rotate to `audit.log.{YYYYMMDD-HHMMSS}` and start fresh |
| 30 days since first record | Rotate (caps untouched logs in the working set) |
| Shim restart | No rotation; appends to existing file |
| User runs `autogpt-shim audit rotate` | Explicit rotation |

Each new file starts with `seq=1` and `prev_hmac=null`. The first
record (`SHIM_START` for restarts, or just the first op) carries
`prev_hmac=null` to mark the chain origin.

Rotated files are kept indefinitely by default. `autogpt-shim audit
prune --older-than 90d` deletes files where the newest record is older
than the threshold. The shim never auto-prunes — log loss is the
user's call.

## Reading and exporting

```bash
autogpt-shim audit tail        # tail -f the current file
autogpt-shim audit show {seq}  # pretty-print one record
autogpt-shim audit verify      # chain-verify the current file
autogpt-shim audit verify-all  # verify every rotated file
autogpt-shim audit export      # zip + sign for support upload
```

Export bundles all current+rotated files into a `.zip` with a
detached signature (the user's audit key signs the manifest). The
platform-side viewer accepts these uploads, verifies the signature
against the user's machine_id, and shows the operator a parsed feed.

## What this catches

- **Log truncation** — chain breaks at the first record after the cut.
- **Record deletion** — `seq` jumps, prev_hmac mismatches.
- **Field tampering** — HMAC mismatch on the tampered record.
- **Wholesale log replacement** — verifier needs the audit key; an
  attacker who replaces both the log and the key in the keychain has
  also locked themselves out of any future verification by the user.
- **Cross-machine impersonation** — `machine_id` is in every record
  and signed; can't lift a chain from one machine and replay on another.

## What this does NOT catch

- Real-time tampering by a process running AS the shim user with
  keychain access. (Mitigation: keychain prompts are user-visible on
  macOS/Windows; an attacker writing to the audit key triggers a TCC
  / consent dialog.)
- The platform asking for something legitimate but harmful. (Mitigation:
  Layer 5 in SECURITY.md — local confirmation prompts for dangerous ops.)
- An attacker who compromises the platform's OAuth issuer and forges a
  token. (Mitigation: token revocation per OAUTH_FLOW.md; user reviews
  audit log for unexpected sessions.)

## Future

- **Platform-side audit viewer.** Web UI under settings that ingests
  exported zips and presents a per-session timeline. Tracked in the
  PR description, not built yet.
- **Append-only filesystem flags.** macOS `chflags uchg`, Linux
  `chattr +a`, Windows ACL deny-rewrite. Hardens against in-band
  tampering even by the shim's own user.
- **Remote witness.** Optionally, shim can stream HMAC checkpoints to
  the platform every N records so the platform can detect log
  truncation after-the-fact even without a user-initiated upload.
  Trade-off: leaks operation metadata (timing, counts) to the platform.
