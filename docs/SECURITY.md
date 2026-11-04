# Security Model — Local PC Executor

The executor is experimental software running as your OS user. Its safeguards
limit accepted requests; they do not isolate agent work from that user account.
Use a dedicated workspace and enable only the capabilities you need.

---

## Threat Model

### What We're Protecting Against

| Threat | Severity | Mitigation |
|--------|----------|------------|
| Prompt injection causing arbitrary command execution | Critical | Shell disabled by default; explicit local opt-in; command audit log |
| Shim token stolen → attacker controls machine | Critical | OS keychain storage; platform ownership/client checks; OAuth uses the dedicated `CONNECT_LOCAL_EXECUTOR` scope |
| Path traversal outside allowed_root | High | Shim enforces path jail; platform validates paths before sending |
| Platform compromise → misuse of paired machines | High | Locally disabled capabilities remain disabled; no per-operation local confirmation is implemented |
| Man-in-the-middle on WebSocket | High | TLS required except explicit loopback development URLs; certificate verification, no pinning |
| Runaway agent loops (infinite commands) | Medium | Machine-wide rate and concurrency limits |
| Accidental file deletion/overwrite | Medium | Selected-root restriction only; no recycle bin, backup, or rollback |
| Screen capture leaking sensitive info | High | Local computer-use opt-in plus platform grant; no sensitive-region masking |

### What We Are NOT Protecting Against

- A compromised AutoGPT platform account — if someone has your OAuth token, they can use your shim. Protect your account with 2FA.
- Physical access to the machine running the shim.
- A malicious Claude response that the user explicitly approved.
- Concurrent local processes replacing files or ancestors after path validation.
- Same-user code accessing the account's credentials or executor configuration.
- Elevated operations — the shim does not drop privileges. Do not run it as root
  or an administrator; enabled shell/computer access can request elevation.

---

## Defense Layers

### Layer 1: OAuth and Local Capability Gates
The public shim client receives only the `CONNECT_LOCAL_EXECUTOR` OAuth permission,
which grants this connection and nothing else.
Individual local capabilities are a separate, shim-enforced boundary: shell,
computer use, clipboard access, local models, and hardware are only advertised
when their local configuration gates are enabled and their runtime dependencies
are available. Recording is stricter: it remains a design preview, is never
advertised, and causes startup to fail closed if enabled. The platform's grant
cannot turn on a locally disabled capability.

There are no separate OAuth scopes for shell, computer use, hardware, or
background work. Hardware probes exist, but hardware operation dispatch is not
implemented. The daemon checks negotiated capability grants outside the model
before dispatching operations.

All platform endpoints and explicit OAuth/WebSocket overrides require HTTPS
or WSS. HTTP and WS are accepted only for exact `localhost` or literal loopback
IP addresses for local development, not private LAN addresses or hostnames
that merely resolve to loopback. URLs containing credentials, queries,
fragments, invalid ports, or ambiguous characters are rejected. Validation
runs during configuration and again when deriving URLs for use.

### Layer 2: Allowed Root Path Jail
All `FILE_*` operations are jailed to the per-chat `allowed_root`. In normal
control mode the user selects it through one-level remote directory browsing;
legacy `--session-id` mode still reads it from startup configuration.
The full algorithm is in [CROSS_PLATFORM.md → Path Jail Strategy](CROSS_PLATFORM.md#path-jail-strategy);
a naive `path.startswith(allowed_root)` check is **not enough** and the shim
must use the prescribed algorithm. Violation → `PATH_OUTSIDE_ALLOWED_ROOT`
error, no execution. Complete destinations are checked even when the file or
its parent directories do not exist yet.

**This jail is not a shell sandbox.** When the user explicitly enables shell
execution, both shell strings and direct `argv` processes run with the shim
user's normal OS permissions. Validating the subprocess working directory does
not stop a command from reading, writing, executing, or making network requests
outside `allowed_root`. Treat `--enable-shell` as user-account-level access.
Filtering inherited environment variables and storing tokens in keyring do
not isolate credentials from arbitrary code running as that same user.
Computer use can operate applications, browsers, and terminals outside the
selected folder with the user's existing access.

Recommended: create a dedicated workspace directory, not your home dir.
```
~/.autogpt/workspace/   ← good
~/                      ← bad, don't do this
/                       ← extremely bad
```

The platform cannot submit a path to the host browser. Navigation uses random,
five-minute opaque references bound to the current control connection. A
selection consumes the browse and produces an HMAC-signed root grant bound to
machine, session, revision, canonical path, and filesystem fingerprint. Every
control reconnect invalidates all outstanding references. Restoring a session
requires the grant and revalidates the live directory identity.
The paired user can select a folder remotely without a native confirmation on
the executor computer. Detaching a session does not permanently revoke its
signed root grant; restoration still requires platform session authorization.

Directory browsing exposes immediate directory names and canonical paths to the
paired platform. It never returns files, content, sizes, timestamps,
permissions, symlinks/junctions, inaccessible directories, or recursive data.
Control connections cannot execute data-plane operations; activated children
use copied configuration and immutable session-bound audit contexts.

#### Per-OS path attacks the shim must catch

| OS | Attack vector | Defense |
|---|---|---|
| Windows | NTFS Alternate Data Streams: `workspace\file.txt:hidden` writes to a hidden stream that doesn't show in Explorer | Reject any path containing `:` after the drive component (`C:`). |
| Windows | Reserved device names (`CON`, `PRN`, `AUX`, `NUL`, `COM1`–`COM9`, `LPT1`–`LPT9`) — case-insensitive, with or without extension. `workspace\CON.txt` opens the console device. | Reject by case-folded basename match against the reserved list, BEFORE checking extension. |
| Windows | `\\?\` path-length prefix bypass — `\\?\C:\workspace\..\..\etc` lets a long path skip normalization in some APIs | Always pass through `os.path.realpath` which resolves these. |
| Windows | NTFS junctions and reparse points pointing outside the jail | `os.path.realpath` (Python 3.8+) follows them; compare resolved paths. |
| macOS | APFS firmlinks (`/Users`, `/Applications`, `/Library`) — invisible to most APIs but `realpath` reveals them | Always `realpath` both the requested path and `allowed_root`. |
| macOS | Case-insensitive default FS: `/Workspace/foo` references the same inode as `/workspace/foo` — naive string `startswith` lets `/workspaceother/foo` through if `allowed_root = /workspace` | Always compare with `os.path.normcase` after `realpath`, AND with a `os.sep` boundary appended. |
| Linux | Bind mounts pointing outside the jail | `realpath` does NOT resolve bind mounts. Document that the user is responsible for not bind-mounting external dirs into `allowed_root`. |
| Linux | Case-folded ext4 dirs (kernel 5.2+) inside a case-sensitive FS | The jail probe must check case-sensitivity *per directory*, not globally. |
| All | Path-jail TOCTOU: attacker swaps a file or ancestor between check and open | **Not contained.** Descriptor-based traversal and Windows reparse-safe handles are not implemented. Do not share the workspace with untrusted concurrent local processes. |
| All | Long-path attacks (`a/../a/../a/...` repeated) to exhaust normalization | Cap path length at OS-appropriate max (4096 Linux, 1024 macOS, 32767 Windows with `\\?\`) before normalization. |

The path jail is not a filesystem sandbox. Bind mounts can expose data whose
names lie inside the selected tree. Content reads/writes conservatively reject
multi-linked regular files, including hard links entirely inside the workspace,
and non-regular files. The opened descriptor is checked before reading or
truncating; this does not contain concurrent same-user changes to links or path
ancestors. Moves never fall back to copying contents across filesystems.
Selecting a broad root
also exposes credentials and executor configuration stored there. Avoid
selecting your home, filesystem root, or executor state/configuration folders.

### Layer 3: Command Auditing

Wire operations (`EXECUTE_COMMAND`, `FILE_*`, `COMPUTER_TOOL_CALL`) plus
shim-internal events (start/stop, WS
connect/disconnect, `JAIL_VIOLATION`, `TOKEN_REFRESHED`) attempt to append to
a JSONL audit log in the per-OS state directory (see
[CROSS_PLATFORM.md → Audit log location](CROSS_PLATFORM.md#audit-log-location)).
Each record carries an HMAC-SHA256 chain to detect modified records and
interior gaps. Deletion of an entire valid tail, rotated file, or log is not
detectable without an independently stored checkpoint. Full format, per-op fields, the tamper-evidence
algorithm, and the `autogpt-shim audit` CLI are spec'd in
[AUDIT_LOG.md](AUDIT_LOG.md). The audit key never leaves the machine;
the user provides it out-of-band when uploading a log for support.
Audit initialization is mandatory: if the key or writer cannot be created,
daemon construction fails and the CLI exits with `EX_CONFIG` rather than
running in an unaudited degraded mode.
An existing log must verify before appending; malformed stored keys or corrupt
logs fail closed without being silently replaced. Runtime audit writes remain
best-effort for most data operations: disk/writer failures can leave completed
operations unrecorded. Audit evidence is not an execution authorization
mechanism or a guaranteed complete record. A same-user attacker who obtains
the audit key can forge records.

### Layer 4: Rate Limiting (Shim-Side)
Shim enforces:
- Max 60 commands per minute across all child chats on the machine
- Max 10 concurrent commands across all child chats, additionally bounded by
  the machine-wide negotiated `max_concurrent`
- Max 10 MiB per file read/write by default; text responses reserve JSON framing headroom
- Max 10 screenshots per minute across all child chats (computer use)

Exceeding limits → `SHIM_OVERLOADED` error returned to platform.

### Layer 5: Optional Local Confirmation Prompts
Future: shim can be configured to require local confirmation (system notification + user click)
before executing commands matching certain patterns (e.g., `rm`, `sudo`, any write to paths
outside a sub-directory). Useful for cautious users who want human-in-the-loop.

### Layer 6: Network Isolation (Future)
No network namespace, outbound destination policy, or network restriction is
implemented. Enabled shell commands and controlled applications use the host's
normal network access. OS isolation and destination approval are future work.

---

## What the Shim Can and Cannot Do By Default

| Operation | Default | Override |
|-----------|---------|----------|
| Read files in allowed_root | ✅ | — |
| Write files in allowed_root | ✅ | — |
| Execute shell commands | ❌ | `start --enable-shell` (unrestricted user-level access) |
| Access files outside allowed_root | ❌ | Expand allowed_root (explicit) |
| Take screenshots | ❌ | Local `enable_computer_use` configuration plus platform capability grant |
| Inject mouse/keyboard | ❌ | Local `enable_computer_use` configuration plus platform capability grant |
| Launch programs with arguments | ❌ | Part of computer use (same configuration and grant). Like `--enable-shell`, this is user-account-level access: a launched program, or a terminal driven by keyboard input, runs whatever it is given. Each launch is audited with its arguments. |
| Access serial/USB/GPIO | ❌ | Operation dispatch is not implemented |
| Run elevated operations | Uses executor user's privileges | No privilege-dropping or elevation sandbox |
| Background tasks when user absent | Available while daemon is running | No separate background-work permission |
| Access the internet via commands | ❌ (shell is off) | Shell commands may use the network after `--enable-shell` |

---

## Token Security

- Tokens stored in OS keychain (never in dotfiles or env vars)
- Access-token lifetime and refresh rotation are controlled by the platform
- Refresh token stored encrypted in keychain
- Token pairs and checked endpoint metadata are stored in a deployment-specific
  keychain record, bound to the backend path, OAuth client, authorization and
  token/revocation endpoints, and WebSocket base. Changing an endpoint requires
  explicit authentication instead of forwarding another deployment's tokens.
  Legacy unscoped tokens are never forwarded, migrated, or deleted automatically;
  see [OAuth migration guidance](OAUTH_FLOW.md#deployment-bound-credentials).
- Authenticated WebSocket redirects are rejected, including same-origin path
  changes. OAuth HTTP requests do not follow redirects.
- `autogpt-shim revoke` posts both access and refresh tokens to
  `POST /api/oauth/revoke` as the public client, and only then clears the OS
  deployment's keychain record. Other deployments and legacy credentials are
  untouched. A network/server error preserves local tokens so revocation can be
  retried instead of falsely reporting success.
- A successful platform revocation also pushes session revocation to connected shims.
- Tokens carry `CONNECT_LOCAL_EXECUTOR`; the WebSocket additionally checks session ownership,
  client identity, and local capability gates.

### Per-OS Keychain Availability

| OS | Backend | Reliability | Fallback when unavailable |
|---|---|---|---|
| macOS | Keychain Services (via `keyring`) | Requires usable, unlocked keyring | No OAuth token fallback |
| Windows | Credential Manager (via `keyring`) | Requires usable keyring | No OAuth token fallback |
| Linux (GUI session, GNOME / KDE) | Secret Service (`gnome-keyring`, `kwallet`) via D-Bus | Depends on session setup | Configure a supported keyring |
| Linux (headless server, no D-Bus, no `pass`) | none | OAuth token storage fails | Configure a supported keyring |
| WSL2 (most distros) | none by default | OAuth token storage fails | Configure a supported keyring |

**OAuth tokens have no file fallback.** A legacy audit-key-only fallback uses
`AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE` and `audit-key.enc` under
`AUTOGPT_SHIM_FALLBACK_DIR` (default `~/.autogpt-local-executor`). Its format is
HMAC-derived XOR obfuscation, not authenticated encryption or a slow password
KDF. It does not make headless pairing work. Malformed stored keys fail closed;
an existing log detects a wrong decoded key before appending. Without an
existing log, the fallback cannot authenticate a passphrase or detect
valid-length ciphertext corruption. Prefer a supported OS keyring.

---

## Incident Response

If you believe your shim was compromised:

1. Stop the foreground daemon with Ctrl+C, or stop its launchd/systemd/Task Scheduler entry.
2. `autogpt-shim revoke` — revokes the configured deployment's saved OAuth tokens;
   repeat with each deployment's configuration as needed
3. Review the audit log at the per-OS location (see [AUDIT_LOG.md](AUDIT_LOG.md)) with `autogpt-shim audit tail` / `verify`
4. Change your AutoGPT account password and re-enable 2FA
5. Report to security@autogpt.net with the audit log

---

## Known Limitations of v0 (MVP)

- No command allow/deny lists yet (after opt-in, shell commands have the shim
  user's normal access and are not confined to `allowed_root`)
- No local confirmation prompts yet
- No network isolation
- No separate read-only file capability, recycle-bin deletion, or write rollback
- No protection against concurrent same-user path replacement
- Audit records are HMAC-chained; protect the audit key in the OS keychain.
- Computer use has no "sensitive region" masking (entire screen captured)
- Shim crash does not guarantee in-flight commands are cancelled

Prompt instructions cannot supply these controls. Treat files, webpages,
terminal output, and model output as untrusted input. Additional policy must
be enforced in the executor or an OS isolation layer before granting broader
access.
