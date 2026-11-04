# Cross-Platform Support — Local PC Executor

> **Status**: Draft v0.1 — subject to change
>
> This document specifies the per-OS behavior contracts that `LocalPCShim` must
> uphold and the assumptions the platform side may make. It complements
> [PROTOCOL.md](PROTOCOL.md) (wire format), [SECURITY.md](SECURITY.md) (per-OS
> attack surface), and [PLATFORM_HOOKS.md](PLATFORM_HOOKS.md) (adapter work on
> the platform side).

The shim's job is to make every supported OS look the same to the platform
adapter. Where that's impossible, the shim advertises what it actually supports
in `HELLO.capabilities` and the platform degrades gracefully.

---

## 1. Supported OS Matrix

| OS | Architecture | Tier | Notes |
|---|---|---|---|
| macOS 13+ (Ventura/Sonoma/Sequoia) | arm64 (Apple Silicon) | **Tier 1** | Primary dev target. Computer use requires Accessibility + Screen Recording TCC grant. |
| macOS 13+ | x86_64 (Intel) | **Tier 1** | Same code path as arm64; Rosetta not required. |
| Windows 10 / 11 | x86_64 | **Tier 1** | Computer use works on Win32 desktop; UAC-elevated windows are uncontrollable from a non-elevated shim. |
| Ubuntu 22.04 / 24.04, Debian 12 | x86_64 | **Tier 1** | Computer use works on X11. **Wayland sessions block input injection** — shim advertises `computer_use` only when `XDG_SESSION_TYPE=x11`. |
| Ubuntu / Debian | arm64 | **Tier 2** | Same code path; not all hardware features tested. |
| Fedora 39+ / RHEL 9+ | x86_64 | **Tier 2** | Same code path as Ubuntu. Wayland is default — same X11 caveat applies. |
| WSL2 (Ubuntu/Debian on Windows) | x86_64 | **Tier 2** | Treated as Linux. `/mnt/c/...` paths work but cross the Windows/Linux permission boundary; see [WSL2 section](#wsl2). No computer use (no display). |
| Windows on ARM | arm64 | **Best-effort** | Untested. Python 3.11+ runs natively; `pyautogui` should work. |
| Other Linux distros (Arch, openSUSE, Alpine, NixOS) | x86_64 / arm64 | **Best-effort** | Should work via pipx. Keychain backend depends on installed Secret Service implementation. |
| FreeBSD, OpenBSD, illumos | any | **Unsupported** | Out of scope. |

**Tier definitions**:
- **Tier 1**: CI runs the test suite; release blockers if broken.
- **Tier 2**: CI smoke-tests; bugs accepted but releases don't block.
- **Best-effort**: Code paths exist; community contributions welcome.

---

## 2. Cross-OS Dimensions

One row per dimension. `n/a` means "doesn't apply on this OS." `same` means
"identical behavior to the cell to the left."

### Path handling

| Dimension | macOS | Linux | Windows | Notes |
|---|---|---|---|---|
| Path separator | `/` | `/` | `\` (also accepts `/`) | Shim normalizes incoming wire paths with `pathlib.PurePath`. |
| Drive letters / UNC | n/a | n/a | `C:\`, `\\server\share` | Path jail must compare on the same drive; cross-drive paths reject. |
| Reserved filenames | none | none | `CON`, `PRN`, `AUX`, `NUL`, `COM1`–`COM9`, `LPT1`–`LPT9` (case-insensitive, with or without extension) | Reject in FILE_WRITE; reject reads if the name is `\\?\`-escaped to a real reserved device. |
| Invalid path chars | `:` (legacy HFS quirk; modern APFS allows) | none | `< > : " \| ? *` and control chars 0–31 | Reject on FILE_WRITE before opening. |
| Max path length | 1024 (PATH_MAX) | 4096 (PATH_MAX) | 260 (MAX_PATH) unless `\\?\` prefix → 32767 | Shim transparently uses `\\?\` on Windows for paths > 240 chars. |
| Alternate data streams | n/a | n/a | `file.txt:hidden` (NTFS ADS) | Path jail strips/rejects any path containing `:` after the drive component. |
| Case sensitivity (default FS) | **Insensitive** (APFS default) | **Sensitive** (ext4/btrfs/xfs) | **Insensitive** (NTFS default) | Path jail compares case-folded on macOS+Windows; raw-byte on Linux. See [Path Jail Strategy](#path-jail-strategy). |
| Home dir env var | `$HOME` | `$HOME` | `%USERPROFILE%` (also `%HOMEDRIVE%%HOMEPATH%`) | Shim resolves `~` via `pathlib.Path.home()` which handles all three. |

### Symlinks, junctions, and reparse points

| Dimension | macOS | Linux | Windows | Notes |
|---|---|---|---|---|
| Symbolic links | Yes (`ln -s`) | Yes | Yes (NTFS, requires `SeCreateSymbolicLinkPrivilege` or developer mode) | Resolve before jail check. |
| Junctions | n/a | n/a | Yes (`mklink /J`) | NTFS reparse points; `os.path.realpath` follows them. |
| Firmlinks | Yes (`/Users`, `/Applications` etc. on APFS) | n/a | n/a | Transparent; `realpath` resolves them. |
| Bind mounts | n/a | Yes | n/a | Not resolved by `realpath`; treated as part of the path tree. |
| Canonicalize call | `os.path.realpath` | `os.path.realpath` | `os.path.realpath` (Python 3.8+ resolves junctions) | Always run realpath before path-jail comparison. |

### Remote directory-browser roots

The persistent control channel starts at virtual host roots rather than
accepting a path from the platform.

| Platform | Roots exposed |
|---|---|
| macOS | Canonical home plus accessible real local data volumes immediately under `/Volumes`; system/startup volumes are excluded. |
| Linux | Canonical home plus accessible user mounts immediately under `/media/$USER` and `/run/media/$USER`; `/`, `/proc`, `/sys`, `/dev`, and general `/run` are never roots. |
| WSL2 | Linux home/user mounts plus accessible one-letter fixed-drive mounts immediately under `/mnt`. |
| Windows | Canonical `%USERPROFILE%` plus accessible fixed/removable drive roots; remote, optical, recovery/system, and redirecting reparse roots are excluded. A reparse point counts as redirecting when its tag has the name-surrogate bit (symlinks, junctions); OneDrive's cloud placeholders don't, and stay browsable. |

Each request lists one immediate level with fixed host-side scan/response caps.
Symlinks, junctions, inaccessible entries, files, and invalid wire names are
skipped. The result includes directory display names and canonical absolute
paths only; navigation and selection use short-lived opaque references.

### Shell selection

| Dimension | macOS | Linux | Windows | Notes |
|---|---|---|---|---|
| `shell: "auto"` runs | `bash` if on `PATH`, else `sh` | `bash` if on `PATH`, else `sh` | `cmd.exe` | Shim's default for `EXECUTE_COMMAND`. It ignores the user's login shell, so on macOS `auto` runs `bash`, not the default `zsh`. |
| Explicit `shell` values | `bash`, `sh`, `zsh`, `pwsh` | `bash`, `sh`, `zsh`, `pwsh` | `cmd`, `powershell`, `pwsh`, `bash` (Git Bash / WSL) | Each needs its binary on `PATH`; otherwise `SHELL_NOT_AVAILABLE`. See PROTOCOL.md. |
| POSIX `bash -c "..."` shape | Works | Works | Not available unless Git Bash / WSL installed | Platform must not assume bash on Windows — use `argv` array or `shell: "auto"`. |
| Argument quoting | `shlex.quote` | `shlex.quote` | `subprocess.list2cmdline` | Shim uses the right quoter per OS when given `argv` form. |

### Process control & signals

| Dimension | macOS | Linux | Windows | Notes |
|---|---|---|---|---|
| Graceful terminate | `SIGTERM` | `SIGTERM` | `CTRL_BREAK_EVENT` for console apps; `TerminateProcess` for GUI apps | Shim picks per-OS in command runner. |
| Hard kill | `SIGKILL` | `SIGKILL` | `TerminateProcess` | Always available. |
| Timeout enforcement | `asyncio.wait_for` + SIGTERM to the group → SIGKILL to whatever is left of it after 2s, even once the command itself has exited | same | `asyncio.wait_for` + CTRL_BREAK → wait 2s → TerminateProcess | Common timeout sequence; per-OS signal differs. |
| Process group / job control | `setsid` + kill PGID | same | `CREATE_NEW_PROCESS_GROUP` flag | So we kill child processes spawned by the command. |

### Environment variables

| Dimension | macOS | Linux | Windows | Notes |
|---|---|---|---|---|
| Name case sensitivity | Sensitive | Sensitive | **Insensitive** (`PATH` == `Path` == `path`) | Wire payload uses uppercase by convention; shim merges case-folded on Windows. |
| `PATH` separator | `:` | `:` | `;` | Platform should not synthesize `PATH`; let shim use the OS-native env. |
| `HOME` analog | `$HOME` | `$HOME` | `%USERPROFILE%` | Shim exposes both `HOME` and `USERPROFILE` for portability. |
| Temp dir env | `$TMPDIR` | `$TMPDIR` (often unset, fallback `/tmp`) | `%TEMP%` / `%TMP%` | Use `tempfile.gettempdir()` always. |
| Safe-env baseline | preserves `HOME`, `PATH`, `LANG`, `LC_*`, `TZ`, `TMPDIR`, `USER`, `SHELL` | same | preserves `USERPROFILE`, `HOMEDRIVE`, `HOMEPATH`, `Path`, `SystemRoot`, `TEMP`, `TMP`, `ComSpec`, `OS`, `windir`, `LOCALAPPDATA`, `APPDATA` | Shim drops everything else before injecting wire-supplied env. |

### Text encoding & line endings

| Dimension | macOS | Linux | Windows | Notes |
|---|---|---|---|---|
| Default text encoding | UTF-8 | UTF-8 | UTF-8 on disk; CP1252/locale on console | Shim always reads/writes files as bytes internally. |
| Line ending on disk | `\n` | `\n` | `\r\n` (when an app writes "text" mode) | Shim does NOT auto-translate. See [PROTOCOL.md FILE_READ semantics](PROTOCOL.md). |
| Console / subprocess encoding | UTF-8 | UTF-8 (depends on `LC_*`) | OEM code page (e.g., CP437) or UTF-8 if `PYTHONIOENCODING=utf-8` | Shim sets `PYTHONIOENCODING=utf-8` and `chcp 65001` on Windows before subprocess. |

### Computer use (screen capture + input injection)

| Dimension | macOS | Linux (X11) | Linux (Wayland) | Windows | Notes |
|---|---|---|---|---|---|
| Screen capture | `screencapture` / `mss` | `mss` / `pyautogui` | **Blocked** unless using portal+pipewire | `mss` / `pyautogui` | Wayland sessions: `capabilities` omits `computer_use`. |
| Input injection | `pyautogui` (Quartz) | `xdotool` / `pyautogui` | **Blocked** | `pyautogui` (Win32 SendInput) | Wayland: blocked. |
| Permission prompt | TCC: Accessibility (input) + Screen Recording (capture) | none | n/a | UAC: non-elevated shim cannot inject into elevated windows | `autogpt-shim doctor` proactively surfaces the macOS dialog via `AXIsProcessTrustedWithOptions({AXTrustedCheckOptionPrompt: True})`; the daemon refuses to bind WS (exit 78) when computer-use is requested but consent is missing, and launchd's `KeepAlive: SuccessfulExit=false` re-launches it after the user grants consent. Revocation mid-session returns `PERMISSION_PENDING` on the next AX-dependent op; the platform LLM retries on the next turn. See [COMPUTER_USE.md §Q5](COMPUTER_USE.md#q5--macos-tcc-first-prompt-ux-locked). |
| DPI scaling | Logical points: input coordinates and display rects are in points; screenshots are captured at the backing resolution | 1:1 typically | n/a | Per-monitor DPI; shim sets process DPI aware via `SetProcessDpiAwareness(2)` | Input coordinates on the wire are display-global logical units (points on macOS). A screenshot can be larger by its `display_scale`; to send a point in the image as a coordinate, divide it by `display_scale` and add the screenshot's `meta.origin`. |
| Multi-monitor | Supported | Supported | n/a | Supported | `monitor: -1` stitches; `monitor: N` selects. |

### Autostart / background daemon

| Dimension | macOS | Linux | Windows | Notes |
|---|---|---|---|---|
| Install mechanism | launchd LaunchAgent (`~/Library/LaunchAgents/net.autogpt.shim.plist`) | systemd user unit (`~/.config/systemd/user/autogpt-shim.service`) | Task Scheduler `\AutoGPT\Shim` (XML at `%LOCALAPPDATA%\autogpt-local-executor\autogpt-shim.xml`, registered via `schtasks /Create /XML ...`) | `autogpt-shim install` writes the file; the user registers it with the OS service manager (the CLI prints the exact command). `autogpt-shim uninstall` reverses it. |
| Start at login | `RunAtLoad` + `KeepAlive: {SuccessfulExit=false}` keys | `systemctl --user enable --now autogpt-shim.service` + `loginctl enable-linger $USER` for headless boxes | Trigger: `AT_LOGON` (Task Scheduler `<LogonTrigger>`) | All require user opt-in. CLI prints the systemctl / launchctl / schtasks commands but never runs them. The macOS `SuccessfulExit=false` flag is load-bearing: it powers the TCC re-launch chain (the daemon exits 78 when computer-use is requested but Accessibility / Screen Recording aren't granted; the OS re-spawns the daemon on the next TCC.db change, so granting consent in System Settings auto-recovers without the user touching launchctl). See [COMPUTER_USE.md §Q5](COMPUTER_USE.md#q5--macos-tcc-first-prompt-ux-locked). |
| Logs | `~/Library/Logs/autogpt-local-executor/*.log` | `journalctl --user -u autogpt-shim` + `$XDG_STATE_HOME/autogpt-local-executor/` | `%LOCALAPPDATA%\autogpt-local-executor\logs\` | Audit log location per row "Audit log" below. |
| Headless server / no DM | Works (still has launchd) | Works (with `enable-linger`) | Works | None require a logged-in GUI session for the WebSocket loop. |

An authentication failure (`start` exits 77, `EX_NOPERM`) stops the
service instead of restarting it, since a restart would only retry the
rejected credentials. The systemd unit sets `RestartPreventExitStatus=77`.
launchd's `KeepAlive` restarts any non-zero exit, so the LaunchAgent sets
`AUTOGPT_SHIM_SUPERVISOR=launchd` and `start` exits 0 for this case.
Task Scheduler retries three times, a minute apart. Run
`autogpt-shim auth`, then restart the service. Errors the shim won't
retry, such as a protocol mismatch or a revoked session, already exit 0
and are logged.

A keychain that can't be read yet (still locked at login, say) is not a
failed sign-in: the shim keeps retrying with its reconnect backoff. Only a
missing keychain backend stops it. The systemd unit doesn't restart on 78
either (`RestartPreventExitStatus=77 78`): a bad config or a preflight
failure fails the same way every time, and Linux has no permission prompt
to wait for. Under launchd, a bad config and a preflight failure no
restart can fix (a preview turned on, pyobjc missing) exit 0 for the same
reason. A missing macOS permission still exits 78, so launchd relaunches
the shim once it is granted.

### Packaging & install

| Dimension | macOS | Linux | Windows | Notes |
|---|---|---|---|---|
| Primary installer | `pipx` | `pipx` | `pipx` | All paths go through pipx in v0. Until the first PyPI release, install from git: `pipx install git+https://github.com/Significant-Gravitas/autogpt-local-executor.git`. After it, `pipx install autogpt-local-executor` works too. |
| Secondary | Homebrew tap (`brew install Significant-Gravitas/tap/autogpt-local-executor`) | distro packages (deb/rpm) — best-effort | Scoop / winget — best-effort | Future. |
| Signing | Apple Developer ID + notarization for `.pkg` | none (pipx wheels are unsigned) | Authenticode signing for `.exe` wrapper (SmartScreen) | Required if we ship native installers. |
| Python runtime | 3.11+ via pyenv/pipx-bundled | 3.11+ via distro or pyenv | 3.11+ via python.org or Microsoft Store | Pin to 3.11 minimum in `pyproject.toml`. |

### Keychain / credential storage

| Dimension | macOS | Linux | Windows | Notes |
|---|---|---|---|---|
| Backend | Keychain Services | Secret Service (D-Bus) — gnome-keyring or kwallet | Windows Credential Manager | Via the `keyring` library. |
| Headless fallback | Requires usable keyring | **`keyring` fails** if no D-Bus / no keyring daemon | Requires usable keyring | OAuth tokens have no file fallback. Configure a supported keyring before pairing. The legacy audit-key fallback does not store OAuth tokens. |
| WSL2 | n/a | Use a supported Linux keyring; none is configured by default in many distros | n/a | Pairing requires working keyring storage. |

### Machine ID source

| Dimension | macOS | Linux | Windows | Notes |
|---|---|---|---|---|
| Source | `ioreg -rd1 -c IOPlatformExpertDevice` → `IOPlatformUUID` | `/etc/machine-id` (fallback `/var/lib/dbus/machine-id`) | Registry `HKLM\SOFTWARE\Microsoft\Cryptography\MachineGuid` | All stable across reboots. |
| Rotates on | Logic-board replacement or fresh macOS install | `systemd-machine-id-setup --commit` (rare) | OS reinstall | Document as "stable identifier; rotates on reinstall." |
| WSL2 | n/a | Has its own `/etc/machine-id` — distinct from host Windows | n/a | A user's Windows host + WSL2 distro are two distinct shims, by design. |
| Wire format | UUID string, lowercase, dashes | same | same | Normalize in shim. |

### Audit log location

| OS | Path |
|---|---|
| macOS | `~/Library/Logs/autogpt-local-executor/audit.log` |
| Linux | `$XDG_STATE_HOME/autogpt-local-executor/audit.log` (fallback `~/.local/state/autogpt-local-executor/audit.log`) |
| Windows | `%LOCALAPPDATA%\autogpt-local-executor\logs\audit.log` |
| WSL2 | Linux path; `/mnt/c/...` is allowed but discouraged (slow, Windows ACLs). |

### Default `allowed_root`

| OS | Default | Rationale |
|---|---|---|
| macOS | `~/Documents/autogpt-workspace` | Documents is iCloud-syncable and Finder-visible. |
| Linux | `~/autogpt-workspace` | XDG dirs vary too much; flat home is universal. |
| Windows | `%USERPROFILE%\autogpt-workspace` | Same rationale as Linux; visible in Explorer sidebar. |
| WSL2 | `~/autogpt-workspace` inside the distro | Avoid `/mnt/c` by default; user can override. |

Shim creates the directory on first run if missing. User can override via
`--allowed-root /some/path` on the CLI or in `config.toml`.

### OAuth callback port

| Dimension | All OSes |
|---|---|
| Default port | `41899` |
| Fallback strategy | If `41899` is in use, scan `41899–41910` and bind the first free port. |
| Listener address | `127.0.0.1` only — never `0.0.0.0`. |
| Firewall behavior | Most OSes prompt for inbound localhost binds on first run (macOS Application Firewall, Windows Defender Firewall). User must approve. |

### Network isolation (future, not v0)

| OS | Mechanism | Status |
|---|---|---|
| Linux | network namespaces + `nftables` egress rules | Planned (root-required helper) |
| macOS | `pf` packet filter rules per-PID via `pfctl` anchor | Future |
| Windows | Windows Filtering Platform (WFP) callouts | Future |

### `HELLO.platform` and `HELLO.arch` values

See [PROTOCOL.md](PROTOCOL.md) for the wire enum. Summary:

| Python detection | Wire `platform` value |
|---|---|
| `sys.platform == "darwin"` | `darwin` |
| `sys.platform == "linux"` AND `/proc/version` lacks `microsoft` | `linux` |
| `sys.platform == "linux"` AND `/proc/version` contains `microsoft` | `wsl2` |
| `sys.platform == "win32"` | `windows` |

| `platform.machine()` | Wire `arch` value |
|---|---|
| `x86_64`, `AMD64`, `amd64` | `x86_64` |
| `arm64`, `aarch64`, `ARM64` | `arm64` |
| anything else | reject HELLO with `UNSUPPORTED_ARCH` |

---

## Path Jail Strategy

The path jail is the most security-critical component of the shim. A naive
prefix string check (`path.startswith(allowed_root)`) is **broken** in three
ways:

1. `/Users/alice/autogpt-workspace2` starts with `/Users/alice/autogpt-workspace`.
2. On case-insensitive FS, `/users/alice/...` is the same directory but a
   different string.
3. Symlinks, junctions, and reparse points let an inside-jail path resolve to
   outside-jail.

### Algorithm

For every FILE_* operation (READ, WRITE, STAT, LIST, DELETE, MOVE — and both
source AND destination of MOVE):

```
def is_inside_jail(requested_path: str, allowed_root: str) -> bool:
    # 1. Lexical normalize. Handle Windows drive letters, UNC, mixed slashes.
    p = pathlib.Path(requested_path).expanduser()

    # 2. Reject obvious bad shapes BEFORE filesystem access.
    if is_reserved_name(p):           # Windows: CON, PRN, NUL, COM1, ...
        return False
    if has_alternate_data_stream(p):  # Windows: "file.txt:hidden"
        return False
    if has_invalid_chars(p):          # Windows: < > " | ? * and ctrl chars
        return False

    # 3. Resolve symlinks, junctions, firmlinks, reparse points.
    #    Use os.path.realpath, NOT pathlib.Path.resolve() — realpath handles
    #    junctions correctly on Windows since Python 3.8; resolve() can hang
    #    on broken symlinks on some FS.
    try:
        resolved = os.path.realpath(p, strict=False)
    except OSError:
        return False

    root_resolved = os.path.realpath(allowed_root, strict=True)

    # 4. Exact prefix with a path separator boundary.
    #    Append os.sep to prevent /foo matching /foobar.
    root_prefix = root_resolved.rstrip(os.sep) + os.sep
    if resolved == root_resolved or resolved.startswith(root_prefix):
        return True

    # 5. Same directory, different spelling. A case-insensitive filesystem
    #    accepts any case, and macOS realpath keeps the spelling it was given.
    #    Never fold case to compare: `WS` beside a root `Ws` is a different
    #    directory wherever a directory on the way is case-sensitive (case-
    #    sensitive APFS, per-directory NTFS, case-folding ext4 off). Ask the
    #    filesystem: the path is inside only if the root is one of its
    #    existing ancestors, by device and inode.
    root_info = os.stat(root_resolved)
    if root_info.st_ino == 0:
        return False  # no inode numbers, no identity
    for ancestor in (Path(resolved), *Path(resolved).parents):
        try:
            info = os.stat(ancestor)
        except OSError:
            continue  # not created yet
        if (info.st_dev, info.st_ino) == (root_info.st_dev, root_info.st_ino):
            return True
    return False
```

Key invariants:

- **Always realpath both sides.** Never compare a not-yet-resolved path against
  a resolved root.
- **Always check with a separator boundary.** `/foo` must not match `/foobar`.
- **Never fold case.** Case sensitivity varies per volume and, on NTFS, per
  directory, so neither a hardcoded "macOS = insensitive" nor a probe of one
  directory is safe. Compare the exact spelling, then filesystem identity.
  The check never writes anything.
- **Validate complete destinations before creation.** FILE_WRITE may create
  a new file under an existing in-root parent; `create_parents: true` also
  allows missing in-root parents. FILE_MOVE may target a new in-root name.
  Every destination still passes complete canonical-path validation.

Content access has an additional check beyond canonical paths. FILE_READ and
FILE_WRITE reject non-regular files and regular files whose link count is not
exactly one. Hard links have no canonical "original" path, so this deliberately
also rejects multiple links entirely inside the workspace. The content opener
checks existing metadata before opening, opens without truncating, and checks
the opened descriptor with `fstat` before reading or truncating/writing. Where
available, nonblocking and no-follow open flags avoid FIFO waits and reject
replacement final-component symlinks. These are additional defenses, not
descriptor-based traversal of every ancestor or a same-user race guarantee.

FILE_MOVE uses filesystem rename/replace only, without a copy-and-delete
fallback. Cross-device moves fail without pre-deleting the destination;
overwriting a nonempty directory also fails. This prevents a move from reading
linked or special-file content indirectly during a cross-device copy. An
in-root hard-link entry may still be listed, statted, unlinked, or renamed;
these operations do not grant content access, and its content remains blocked.

FILE_DELETE and FILE_MOVE act on the entry a path names: a symlink or junction
is removed, renamed or replaced itself, never its target. The path must still
resolve inside the root, and so must the entry's directory, so neither
operation can act on the root itself.

### What the algorithm catches

| Attack | Caught at step |
|---|---|
| `../../../etc/passwd` | 1 (normalize) → 3 (realpath) → 4 (prefix mismatch) → 5 (root is no ancestor) |
| `/workspace/symlink-to-etc` | 3 (realpath resolves outside) → 4 → 5 |
| `/WORKSPACE/foo` on case-insensitive FS | 4 (spelling differs) → 5 (the root is an ancestor) → allowed (correct) |
| `/WS/foo` beside root `/Ws` in a case-sensitive directory | 4 → 5 (`/WS` is another directory) |
| `/workspace2/foo` (root is `/workspace`) | 4 (boundary check) → 5 |
| `C:\workspace\file.txt:hidden` | 2 (ADS check) |
| `C:\workspace\CON` | 2 (reserved name) |
| `\\?\C:\workspace\..\..\etc` | 1 → 3 → 4 → 5 |

---

## WSL2

Windows Subsystem for Linux v2 is a special case: a real Linux kernel running
under a Windows host. The shim runs as a normal Linux process and looks like
Linux to all Python APIs (`sys.platform == "linux"`).

### Detection

```python
def is_wsl2() -> bool:
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False
```

WSL1 also matches this check, but WSL1 is end-of-life and we don't support it.
If we ever need to distinguish, WSL2 has `WSL_INTEROP` env var set and a
working `/proc/sys/kernel/osrelease` containing `WSL2`.

### Behavior differences from native Linux

1. **`HELLO.platform = "wsl2"`** (not `"linux"`). The platform may want to
   surface "Running on WSL2" in the UI to set user expectations.
2. **`/mnt/c/...` paths cross the filesystem boundary.** They work, but are
   ~10× slower than native ext4 paths and use Windows ACLs instead of POSIX
   permissions. The shim allows them in `allowed_root` if the user explicitly
   configures one, but the default `allowed_root` stays inside the Linux home.
3. **No computer use.** WSL2 has no display server by default. If the user
   has WSLg (Windows 11) running an X server, X11 input injection technically
   works, but coordinates won't match what they see on the Windows host. Shim
   omits `computer_use` from capabilities regardless.
4. **Hardware access is limited.** USB/serial passthrough requires
   `usbipd-win` on the Windows host. Shim detects `/dev/ttyUSB*` presence
   as usual; absence isn't surprising.
5. **Keychain typically unavailable.** Most WSL2 distros don't run a Secret
   Service daemon. Configure a supported keyring before pairing; OAuth token
   storage has no file fallback.
6. **`machine_id` is per-distro, not per-host.** A user with Ubuntu WSL2 and
   Debian WSL2 has two separate shims to the platform — that's intentional;
   they have separate filesystems.

### What the platform should NOT assume about WSL2

- Don't assume the user's "real" desktop apps are reachable.
- Don't assume `/mnt/c/Users/<name>` is writable (Windows ACLs may block).
- Don't assume `127.0.0.1` from WSL2 reaches the Windows host's services —
  it doesn't on WSL2 unless `localhostForwarding` is configured.

---

## References

- [PROTOCOL.md](PROTOCOL.md) — Wire-format enums for `platform`, `arch`,
  `shell`, file ops.
- [SECURITY.md](SECURITY.md) — Per-OS path attack surface, keychain caveats.
- [PLATFORM_HOOKS.md](PLATFORM_HOOKS.md) — Platform-side adapter changes
  required to be cross-OS friendly.
- [OAUTH_FLOW.md](OAUTH_FLOW.md) — Localhost callback port + keychain backends.
