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

### Shell selection

| Dimension | macOS | Linux | Windows | Notes |
|---|---|---|---|---|
| Default interactive shell | `zsh` (macOS 10.15+) | `bash` (most distros) | `cmd.exe` (no PowerShell default) | Shim's default for `EXECUTE_COMMAND` when `shell == "auto"`. |
| Available shells | `sh`, `bash`, `zsh` | `sh`, `bash`, possibly `zsh`/`fish` | `cmd`, `powershell`, `pwsh` (if installed) | `shell` enum on the wire selects explicitly. See PROTOCOL.md. |
| POSIX `bash -c "..."` shape | Works | Works | Not available unless Git Bash / WSL installed | Platform must not assume bash on Windows — use `argv` array or `shell: "auto"`. |
| Argument quoting | `shlex.quote` | `shlex.quote` | `subprocess.list2cmdline` | Shim uses the right quoter per OS when given `argv` form. |

### Process control & signals

| Dimension | macOS | Linux | Windows | Notes |
|---|---|---|---|---|
| Graceful terminate | `SIGTERM` | `SIGTERM` | `CTRL_BREAK_EVENT` for console apps; `TerminateProcess` for GUI apps | Shim picks per-OS in command runner. |
| Hard kill | `SIGKILL` | `SIGKILL` | `TerminateProcess` | Always available. |
| Timeout enforcement | `asyncio.wait_for` + SIGTERM → wait 2s → SIGKILL | same | `asyncio.wait_for` + CTRL_BREAK → wait 2s → TerminateProcess | Common timeout sequence; per-OS signal differs. |
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
| Permission prompt | TCC: Accessibility (input) + Screen Recording (capture) | none | n/a | UAC: non-elevated shim cannot inject into elevated windows | First call surfaces the OS dialog. Shim catches denial and returns `CAPABILITY_NOT_GRANTED`. |
| DPI scaling | Logical points; shim multiplies by backing scale factor | 1:1 typically | n/a | Per-monitor DPI; shim sets process DPI aware via `SetProcessDpiAwareness(2)` | Coordinates on the wire are always physical pixels. |
| Multi-monitor | Supported | Supported | n/a | Supported | `monitor: -1` stitches; `monitor: N` selects. |

### Autostart / background daemon

| Dimension | macOS | Linux | Windows | Notes |
|---|---|---|---|---|
| Install mechanism | launchd LaunchAgent (`~/Library/LaunchAgents/net.autogpt.shim.plist`) | systemd user unit (`~/.config/systemd/user/autogpt-shim.service`) | Task Scheduler `\AutoGPT\Shim` (XML at `%LOCALAPPDATA%\autogpt-local-executor\autogpt-shim.xml`, registered via `schtasks /Create /XML ...`) | `autogpt-shim install` writes the file; the user registers it with the OS service manager (the CLI prints the exact command). `autogpt-shim uninstall` reverses it. |
| Start at login | `RunAtLoad` + `KeepAlive` keys | `systemctl --user enable --now autogpt-shim.service` + `loginctl enable-linger $USER` for headless boxes | Trigger: `AT_LOGON` (Task Scheduler `<LogonTrigger>`) | All require user opt-in. CLI prints the systemctl / launchctl / schtasks commands but never runs them. |
| Logs | `~/Library/Logs/autogpt-local-executor/*.log` | `journalctl --user -u autogpt-shim` + `$XDG_STATE_HOME/autogpt-local-executor/` | `%LOCALAPPDATA%\autogpt-local-executor\logs\` | Audit log location per row "Audit log" below. |
| Headless server / no DM | Works (still has launchd) | Works (with `enable-linger`) | Works | None require a logged-in GUI session for the WebSocket loop. |

### Packaging & install

| Dimension | macOS | Linux | Windows | Notes |
|---|---|---|---|---|
| Primary installer | `pipx install autogpt-local-executor` | `pipx install autogpt-local-executor` | `pipx install autogpt-local-executor` | All paths go through pipx in v0. |
| Secondary | Homebrew tap (`brew install autogpt/tap/local-executor`) | distro packages (deb/rpm) — best-effort | Scoop / winget — best-effort | Future. |
| Signing | Apple Developer ID + notarization for `.pkg` | none (pipx wheels are unsigned) | Authenticode signing for `.exe` wrapper (SmartScreen) | Required if we ship native installers. |
| Python runtime | 3.11+ via pyenv/pipx-bundled | 3.11+ via distro or pyenv | 3.11+ via python.org or Microsoft Store | Pin to 3.11 minimum in `pyproject.toml`. |

### Keychain / credential storage

| Dimension | macOS | Linux | Windows | Notes |
|---|---|---|---|---|
| Backend | Keychain Services | Secret Service (D-Bus) — gnome-keyring or kwallet | Windows Credential Manager | Via the `keyring` library. |
| Headless fallback | n/a (always available) | **`keyring` fails** if no D-Bus / no keyring daemon | n/a | Shim detects via `keyring.backend.SecretService.priority` check; on fail, prompts user to either install `gnome-keyring`/`pass` or fall back to an encrypted file at `$XDG_STATE_HOME/autogpt-local-executor/tokens.enc` with a passphrase derived from a `KEYCHAIN_PASSPHRASE` env var. |
| WSL2 | n/a | Use the Linux path (D-Bus on WSL2 is rare; encrypted-file fallback is the practical default) | n/a | Document the fallback in install guide. |

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

    # 4. Canonical-form comparison.
    if is_case_insensitive_fs(root_resolved):
        # macOS (APFS default), Windows (NTFS default), Linux on case-folded ext4
        a = os.path.normcase(resolved)
        b = os.path.normcase(root_resolved)
    else:
        a = resolved
        b = root_resolved

    # 5. Strict prefix with path separator boundary.
    #    Append os.sep to prevent /foo matching /foobar.
    if not a.endswith(os.sep):
        a_check = a + os.sep
    else:
        a_check = a
    if not b.endswith(os.sep):
        b_check = b + os.sep
    else:
        b_check = b

    return a_check.startswith(b_check) or a == root_resolved

def is_case_insensitive_fs(path: str) -> bool:
    # Cheapest heuristic: probe the actual filesystem.
    # On macOS, default APFS is case-insensitive (but user can format
    # case-sensitive APFS). On Windows, NTFS is case-insensitive by default
    # (but per-directory case-sensitivity flag exists since Win10 1803).
    # On Linux, ext4/btrfs/xfs are case-sensitive (but optional case-folding
    # since kernel 5.2).
    if sys.platform == "win32":
        return True
    if sys.platform == "darwin":
        # Check the actual volume. Use os.pathconf("/", "PC_CASE_SENSITIVE")
        # where available; otherwise probe by creating "X" and looking up "x".
        return _probe_case_insensitive(path)
    return _probe_case_insensitive(path)  # Linux default sensitive, but probe
```

Key invariants:

- **Always realpath both sides.** Never compare a not-yet-resolved path against
  a resolved root.
- **Always check with a separator boundary.** `/foo` must not match `/foobar`.
- **Per-volume case-sensitivity probe.** Don't hardcode "macOS = insensitive";
  modern users can have case-sensitive APFS or per-dir case-sensitive NTFS.
- **Refuse paths that don't exist when STRICT mode is required.** FILE_WRITE
  with `create_parents: true` is the only op that may target a non-existent
  path; even then, the *parent* must exist and be inside the jail.

### What the algorithm catches

| Attack | Caught at step |
|---|---|
| `../../../etc/passwd` | 1 (normalize → 3 (realpath) → 5 (prefix mismatch) |
| `/workspace/symlink-to-etc` | 3 (realpath resolves outside) → 5 |
| `/WORKSPACE/foo` on case-insensitive FS | 4 (normcase) → 5 (matches root) → allowed (correct) |
| `/workspace2/foo` (root is `/workspace`) | 5 (boundary check) |
| `C:\workspace\file.txt:hidden` | 2 (ADS check) |
| `C:\workspace\CON` | 2 (reserved name) |
| `\\?\C:\workspace\..\..\etc` | 1 → 3 → 5 |

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
   Service daemon. Encrypted-file fallback (`tokens.enc`) is the practical
   default. Document this in the install guide.
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
