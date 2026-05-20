# Security Model — Local PC Executor

> ⚠️ This feature is a security nightmare by design. The goal is to make it a *manageable* security nightmare.

---

## Threat Model

### What We're Protecting Against

| Threat | Severity | Mitigation |
|--------|----------|------------|
| Prompt injection causing arbitrary command execution | Critical | Command audit log; user-configurable allow/deny lists |
| Shim token stolen → attacker controls machine | Critical | OS keychain storage; token scoped to `local_executor` only |
| Path traversal outside allowed_root | High | Shim enforces path jail; platform validates paths before sending |
| Platform compromise → all shims pwned | High | Shim can block platform-side: capability gates, rate limits, local confirm prompts |
| Man-in-the-middle on WebSocket | High | TLS required; certificate pinning recommended |
| Runaway agent loops (infinite commands) | Medium | Per-turn command quota; shim-side rate limiter |
| Accidental file deletion/overwrite | Medium | Shim-side recycle bin mode (move to ~/.autogpt-trash instead of delete) |
| Screen capture leaking sensitive info | High | Computer use requires explicit opt-in per session; platform shows "screen access active" indicator |

### What We Are NOT Protecting Against

- A compromised AutoGPT platform account — if someone has your OAuth token, they can use your shim. Protect your account with 2FA.
- Physical access to the machine running the shim.
- A malicious Claude response that the user explicitly approved.
- Root-level operations — the shim runs as the user, not root. Don't run it as root.

---

## Defense Layers

### Layer 1: OAuth Scope Gates
Every capability requires an explicit OAuth scope granted by the user. The platform cannot
issue shell commands to a shim that only has `local_executor:files` scope.

### Layer 2: Allowed Root Path Jail
All file operations are jailed to `allowed_root` (configured by user at shim startup).
The full algorithm is in [CROSS_PLATFORM.md → Path Jail Strategy](CROSS_PLATFORM.md#path-jail-strategy);
a naive `path.startswith(allowed_root)` check is **not enough** and the shim
must use the prescribed algorithm. Violation → `PATH_OUTSIDE_ALLOWED_ROOT`
error, no execution.

Recommended: create a dedicated workspace directory, not your home dir.
```
~/.autogpt/workspace/   ← good
~/                      ← bad, don't do this
/                       ← extremely bad
```

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
| All | Path-jail TOCTOU: attacker swaps a file for a symlink between check and open | Use `O_NOFOLLOW` on Linux, `FILE_FLAG_OPEN_REPARSE_POINT` on Windows, and on macOS open by file descriptor after the realpath check. |
| All | Long-path attacks (`a/../a/../a/...` repeated) to exhaust normalization | Cap path length at OS-appropriate max (4096 Linux, 1024 macOS, 32767 Windows with `\\?\`) before normalization. |

### Layer 3: Command Auditing
Every `EXECUTE_COMMAND` is logged to `~/.autogpt/shim-audit.log` with timestamp,
session_id, command, cwd, exit_code. Log is append-only from shim's perspective.
User can review what ran on their machine.

### Layer 4: Rate Limiting (Shim-Side)
Shim enforces:
- Max 60 commands per minute per session
- Max 10 concurrent commands
- Max 100MB per file read/write
- Max 10 screenshots per minute (computer use)

Exceeding limits → `SHIM_OVERLOADED` error returned to platform.

### Layer 5: Optional Local Confirmation Prompts
Future: shim can be configured to require local confirmation (system notification + user click)
before executing commands matching certain patterns (e.g., `rm`, `sudo`, any write to paths
outside a sub-directory). Useful for cautious users who want human-in-the-loop.

### Layer 6: Network Isolation (Future)
Shim can optionally run commands inside a network namespace that blocks outbound internet
while allowing localhost. Useful for preventing data exfiltration via executed commands.
Requires Linux + root for namespace setup, or a bubblewrap wrapper.

---

## What the Shim Can and Cannot Do By Default

| Operation | Default | Override |
|-----------|---------|----------|
| Read files in allowed_root | ✅ | — |
| Write files in allowed_root | ✅ | — |
| Execute shell commands | ✅ | `--no-shell` flag |
| Access files outside allowed_root | ❌ | Expand allowed_root (explicit) |
| Take screenshots | ❌ | `local_executor:computer_use` scope |
| Inject mouse/keyboard | ❌ | `local_executor:computer_use` scope |
| Access serial/USB/GPIO | ❌ | `local_executor:hardware` scope |
| Run as root / sudo | ❌ | Not supported, period |
| Background tasks when user absent | ❌ | `local_executor:background` scope |
| Access the internet via commands | ✅ (via shell) | `--no-network` flag (Linux only) |

---

## Token Security

- Tokens stored in OS keychain (never in dotfiles or env vars)
- Access token short-lived (1 hour); refresh token used for renewal
- Refresh token stored encrypted in keychain
- `autogpt-shim revoke` — revokes all tokens and disconnects immediately
- On platform side: `POST /auth/revoke` invalidates all shim tokens for a user
- Tokens scoped to `local_executor:*` only — cannot be used to call other AutoGPT APIs

### Per-OS Keychain Availability

| OS | Backend | Reliability | Fallback when unavailable |
|---|---|---|---|
| macOS | Keychain Services (via `keyring`) | Always available | n/a |
| Windows | Credential Manager (via `keyring`) | Always available | n/a |
| Linux (GUI session, GNOME / KDE) | Secret Service (`gnome-keyring`, `kwallet`) via D-Bus | Usually available | Encrypted-file fallback (below) |
| Linux (headless server, no D-Bus, no `pass`) | none | **`keyring` fails** | Encrypted-file fallback (below) |
| WSL2 (most distros) | none by default | **`keyring` fails** | Encrypted-file fallback (below) |

**Encrypted-file fallback**: when the OS keychain is unavailable, the shim
stores tokens in `$XDG_STATE_HOME/autogpt-local-executor/tokens.enc`
(Linux/WSL2) using AES-256-GCM. The key is derived (Argon2id) from a
passphrase that must be supplied via the `AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE`
env var on every shim start. **No passphrase = no shim start.** This is
deliberately friction-heavy so users on capable systems install a real
keyring instead.

Security note: an attacker with read access to the file AND the env var
can decrypt the tokens. This is no worse than dotfile-based storage but no
better either — the keychain backends remain the recommended path.

---

## Incident Response

If you believe your shim was compromised:

1. `autogpt-shim stop` — immediately stops the daemon
2. `autogpt-shim revoke` — revokes all OAuth tokens
3. Review `~/.autogpt/shim-audit.log` to see what ran
4. Change your AutoGPT account password and re-enable 2FA
5. Report to security@autogpt.net with the audit log

---

## Known Limitations of v0 (MVP)

- No command allow/deny lists yet (all shell commands permitted within allowed_root)
- No local confirmation prompts yet
- No network isolation
- Audit log is not tamper-evident (no HMAC chain)
- Computer use has no "sensitive region" masking (entire screen captured)
- Shim crash does not guarantee in-flight commands are cancelled

These are tracked as issues. PRs welcome.
