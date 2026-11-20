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
| macOS 13+ (Ventura/Sonoma/Sequoia) | arm64 (Apple Silicon) | **Tier 1** | Primary dev target. Computer use requires Accessibility + Screen Recording for the app that runs the shim. |
| macOS 13+ | x86_64 (Intel) | **Tier 1** | Same code path as arm64; Rosetta not required. |
| Windows 10 / 11 | x86_64 | **Tier 1** | Computer use works on Win32 desktop; UAC-elevated windows are uncontrollable from a non-elevated shim. |
| Ubuntu 22.04 / 24.04, Debian 12 | x86_64 | **Tier 1** | Computer use needs a desktop session: X11, or Wayland through XWayland (or cua-driver's native Wayland backend, opt-in). |
| Ubuntu / Debian | arm64 | **Tier 2** | Same code path; not all hardware features tested. |
| Fedora 39+ / RHEL 9+ | x86_64 | **Tier 2** | Same code path as Ubuntu. Wayland is the default; computer use goes through XWayland. |
| WSL2 (Ubuntu/Debian on Windows) | x86_64 | **Tier 2** | Treated as Linux. `/mnt/c/...` paths work but cross the Windows/Linux permission boundary; see [WSL2 section](#wsl2). No computer use (no display). |
| Windows on ARM | arm64 | **Best-effort** | Untested. Python 3.11+ runs natively; cua-driver ships a `win_arm64` wheel. |
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
| Path separator | `/` | `/` | `\` (also accepts `/`) | The shim reads a wire path as the OS's own `pathlib.Path`. On macOS and Linux a `\` is an ordinary character of a name. |
| Drive letters / UNC | n/a | n/a | `C:\`, `\\server\share` | A path on another drive, relative to another drive's folder (`D:x`), on another share or host (`\\server\share\x`, `//server/share/x`), or in a device namespace (`\\.\`, `\\?\`) is refused by its name, before anything looks at it: resolving one would have Windows connect to the host and sign in with the person's account, or open the device. A root that is itself on a share or a mapped drive takes the paths under it. See [Path Jail Strategy](#path-jail-strategy). |
| Reserved filenames | none | none | `CON`, `PRN`, `AUX`, `NUL`, `CONIN$`, `CONOUT$`, `COM1`–`COM9`, `LPT1`–`LPT9`, and `COM`/`LPT` with a superscript `¹`, `²` or `³` (case-insensitive, with or without extension, and with spaces before the extension) | Refused for every request that names a path (each FILE_* message and a command's `cwd`), at any depth of the path, with `PATH_RESERVED_NAME`, before anything looks at the path. |
| Invalid path chars | NUL (`\0`) | NUL (`\0`) | `< > " \| ? *` and control chars 0–31, after the drive letter; a `:` there is the next row but one | Refused for every request that names a path, with `PATH_INVALID_CHARS`, before anything looks at the path. A `:` in a name on macOS goes to the filesystem as it is. |
| Max path length | 1024 (PATH_MAX) | 4096 (PATH_MAX) | 32767; a path over 260 (MAX_PATH) is opened only where Windows' long-path setting (`LongPathsEnabled`) is on | The jail refuses a longer path with `PATH_INVALID_CHARS`, counting the characters of the path as sent, with `~` expanded; the OS still applies its own limit to what passes. The shim never adds the `\\?\` prefix, and the jail refuses a path that has one: its `?` is no character of a path. |
| Alternate data streams | n/a | n/a | `file.txt:hidden` (NTFS ADS) | The jail refuses any path with a `:` after the drive letter, with `PATH_INVALID_CHARS`, for every request that names a path. It never strips one. |
| Case sensitivity (default FS) | **Insensitive** (APFS default) | **Sensitive** (ext4/btrfs/xfs) | **Insensitive** (NTFS default) | The jail never folds case to let a path in, on any OS. The resolved path must start with the resolved root exactly, or the filesystem itself must say that the root is one of the path's existing ancestors (the same device and inode), which is how another spelling of the root is recognised on a case-insensitive filesystem. Only the early refusal by name sets case aside, and it can only refuse. See [Path Jail Strategy](#path-jail-strategy). |
| Home dir env var | `$HOME` | `$HOME` | `%USERPROFILE%` (also `%HOMEDRIVE%%HOMEPATH%`) | The jail expands a wire path's leading `~` with `Path.expanduser()`, which reads these. |

### Symlinks, junctions, and reparse points

| Dimension | macOS | Linux | Windows | Notes |
|---|---|---|---|---|
| Symbolic links | Yes (`ln -s`) | Yes | Yes (NTFS, requires `SeCreateSymbolicLinkPrivilege` or developer mode) | Resolve before jail check. |
| Junctions | n/a | n/a | Yes (`mklink /J`) | NTFS reparse points; `os.path.realpath` follows them. |
| Firmlinks | Yes (`/Users`, `/Applications` etc. on APFS) | n/a | n/a | Transparent: a firmlink is not a symlink, so `realpath` leaves the path as it is. |
| Bind mounts | n/a | Yes | n/a | Not resolved by `realpath`; treated as part of the path tree. |
| Canonicalize call | `os.path.realpath` | `os.path.realpath` | `os.path.realpath` (Python 3.8+ resolves junctions) | Always run realpath before path-jail comparison. |

### Remote directory-browser roots

The persistent control channel starts at virtual host roots rather than
accepting a path from the platform.

| Platform | Roots exposed |
|---|---|
| macOS | Canonical home plus accessible real local data volumes immediately under `/Volumes`; system/startup volumes are excluded. |
| Linux | Canonical home plus accessible user mounts immediately under `/media/<login name>` and `/run/media/<login name>`, where udisks mounts a person's removable disks; the login name is the account's own from the user database, and the home folder's name is looked under as well where it differs (a home like `/home/kris@corp`, an overridden `HOME`). `/`, `/proc`, `/sys`, `/dev`, and general `/run` are never roots. |
| WSL2 | Linux home/user mounts plus accessible one-letter fixed-drive mounts immediately under `/mnt`. |
| Windows | Canonical `%USERPROFILE%` plus accessible fixed and removable drive roots (`GetDriveTypeW` 3 and 2); remote, optical and RAM-disk drives are excluded by their type. Only lettered volumes are listed, so the EFI and recovery partitions, which have no letter, are not; one given a letter is a fixed drive like any other. Redirecting reparse roots are excluded. A reparse point counts as redirecting when its tag has the name-surrogate bit (symlinks, junctions); OneDrive's cloud placeholders don't, and stay browsable. |

"Canonical home" is the home folder by its real place: one that is itself a
link (`/home/u -> /data/u`) is resolved first and offered under that path.
Every other link is refused, as below.

Each request lists one immediate level with fixed host-side scan/response caps.
Symlinks, junctions, inaccessible entries, files, and invalid wire names are
skipped. The result includes directory display names and canonical absolute
paths only; navigation and selection use short-lived opaque references.

An invalid wire name is one that isn't valid UTF-8, or one that could read
as another folder's name in the dialog: it holds a control character, a bidi
control, or a zero-width or other invisible format character. U+200C and
U+200D are the exception where a script uses them: inside a Persian word,
after an Indic virama, between the pictographs of an emoji sequence. The
rule is Unicode's own for names (CONTEXTJ, RFC 5892), and the same on every
OS; [PROTOCOL.md](PROTOCOL.md#directory_list_request--directory_list_response)
has it in full.

On Windows, a drive's top folder is listed without the system's own
folders: `Windows`, `Program Files`, `Program Files (x86)`, `ProgramData`,
`Recovery` and `Documents and Settings`. A folder of one of those names
further down is the person's (the `windows` folder of a Flutter project,
say) and is listed like any other. `$RECYCLE.BIN` and `System Volume
Information`, which the system keeps on every volume, are left out wherever
they are.

A folder is listed and can be opened when the account can read and search
it, and chosen for a chat only when it can also write to it. So a folder
the account can only read is still a way to one inside it that is its own:
a removable ext4 or xfs disk, whose top folder is root's as udisks mounts
it, is among the roots. How each system is asked:

| | Can be opened | Can be chosen |
|---|---|---|
| macOS, Linux, WSL2 | `access(2)` for read and search. It takes ACLs into account, and it doesn't open the folder: on macOS, opening one the privacy settings guard (Desktop, Documents, Downloads) would put a prompt on the screen for a folder the person only browsed past. | `access(2)` for write and search as well. |
| Windows | The folder is opened for listing. `os.access` looks at the read-only attribute alone and would call every folder usable, another account's profile under `C:\Users` among them. | The system is asked for a handle on the folder with the right to add a file to it (`FILE_ADD_FILE`), which its access list grants or refuses. No file is made to find out. |

On Windows the answer is the system's for the shim's own process. An
administrator's process whose backup and restore privileges are switched on
(one started from Git Bash; PowerShell and cmd.exe leave them off) is let
into a folder whatever its access list says, so both checks say yes for it.
Listing such a folder then works; writing a file in it the ordinary way
doesn't use those privileges and can still be refused. The shim isn't meant
to run as an administrator ([SECURITY.md](SECURITY.md)).

These looks are part of a listing, so they share its time limit
([PROTOCOL.md](PROTOCOL.md#a-folder-that-doesnt-answer)).

### Shell selection

| Dimension | macOS | Linux | Windows | Notes |
|---|---|---|---|---|
| `shell: "auto"` runs | `bash` if on `PATH`, else `sh` | `bash` if on `PATH`, else `sh` | `cmd.exe` | Shim's default for `EXECUTE_COMMAND`. It ignores the user's login shell, so on macOS `auto` runs `bash`, not the default `zsh`. |
| Explicit `shell` values | `bash`, `sh`, `zsh`, `pwsh` | `bash`, `sh`, `zsh`, `pwsh` | `cmd`, `powershell`, `pwsh`, `bash` (Git Bash / WSL) | Each needs its binary on `PATH`; otherwise `SHELL_NOT_AVAILABLE`. See PROTOCOL.md. |
| POSIX `bash -c "..."` shape | Works | Works | Not available unless Git Bash / WSL installed | Platform must not assume bash on Windows — use `argv` array or `shell: "auto"`. |
| Argument quoting | none: `argv` is executed as given | none: `argv` is executed as given | `subprocess.list2cmdline` joins `argv` into the one command line a Windows program gets | The shim adds no quoting to `command`: a shell gets it as sent. `cmd.exe` gets it on a command line the shim builds, `cmd.exe /d /s /c "..."`, since cmd.exe doesn't undo `list2cmdline`'s quoting. See [PROTOCOL.md](PROTOCOL.md#what-each-shell-is-handed). |

### Process control & signals

| Dimension | macOS | Linux | Windows | Notes |
|---|---|---|---|---|
| Graceful terminate | `SIGTERM` | `SIGTERM` | `CTRL_BREAK_EVENT` for console apps; `TerminateProcess` for GUI apps | Shim picks per-OS in command runner. |
| Hard kill | `SIGKILL` | `SIGKILL` | `TerminateJobObject` on the command's job, which every command that runs has | Always available. |
| Timeout enforcement | `asyncio.timeout` + SIGTERM to the group → SIGKILL to whatever is left of it after 2s, even once the command itself has exited; a cancelled command is stopped the same way | same | `asyncio.timeout` + CTRL_BREAK → wait 2s → end the job, even once the command itself has exited; a cancelled command is stopped the same way | Common timeout sequence; per-OS signal differs. |
| Process group / job control | `setsid` + kill PGID | same | `CREATE_NEW_PROCESS_GROUP` flag, and a job object per command: the command starts suspended and runs once it is in its job, so every process it starts is in the job; one that can't be put in a job (Windows out of resources, or an unusual restriction on the job the shim runs in) is refused with `INTERNAL_ERROR` before it runs, since its stop could end it alone | So we kill child processes spawned by the command. The job doesn't kill on close: what a command that finished started to outlive it keeps running, as on POSIX, and a program that asks to leave the job (`CREATE_BREAKAWAY_FROM_JOB`) may, as a POSIX one may leave its group. |

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
| Console / subprocess encoding | UTF-8 | UTF-8 (depends on `LC_*`) | OEM code page (e.g., CP437) or UTF-8 if `PYTHONIOENCODING=utf-8` | Shim sets `PYTHONIOENCODING=utf-8` for every command, and runs `chcp 65001` first in a `cmd.exe` command. |

### Computer use

Computer use runs on cua-driver; see [COMPUTER_USE.md](COMPUTER_USE.md) and
[CUA_DRIVER.md](CUA_DRIVER.md).

| Dimension | macOS | Linux | Windows |
|---|---|---|---|
| Driver runtime | the driver process the shim starts (`--direct`), never CuaDriver.app's service | the driver process (`--direct`), never a `cua-driver serve` daemon | the driver process (`--direct`), never a `cua-driver serve` daemon |
| Reading windows | Accessibility (AX) tree + ScreenCaptureKit | AT-SPI over the session bus + X11/Wayland capture | UI Automation tree + window capture |
| Acting | AX actions in the background, pixel and foreground input as fallbacks | AT-SPI actions, XSendEvent on X11; foreground input on Wayland | UIA patterns and window messages in the background, foreground input as fallback |
| Permissions | Accessibility + Screen Recording, for the app that runs the shim; `doctor` asks for them; a missing one makes the daemon exit 78 and launchd retry | none | a non-elevated shim can't drive elevated windows |
| Not supported | | sessions without a display | session 0 services; WSL2 (no Windows desktop) |

### Autostart / background daemon

| Dimension | macOS | Linux | Windows | Notes |
|---|---|---|---|---|
| Install mechanism | launchd LaunchAgent (`~/Library/LaunchAgents/co.agpt.shim.plist`) | systemd user unit (`~/.config/systemd/user/autogpt-shim.service`) | Task Scheduler `\AutoGPT\Shim`, a task of the account that ran `install` (XML at `%LOCALAPPDATA%\autogpt-local-executor\autogpt-shim.xml`, registered via `schtasks /Create /XML ...` from that account's own prompt: see [The Windows task's account](#the-windows-tasks-account)) | `autogpt-shim install` writes the file; the user registers it with the OS service manager (the CLI prints the exact command). `autogpt-shim uninstall` undoes both: see [Uninstalling](#uninstalling). |
| Start at login | `RunAtLoad` + `KeepAlive: {SuccessfulExit=false}` keys | `systemctl --user enable --now autogpt-shim.service` + `loginctl enable-linger $USER` for headless boxes | Trigger: that account's own logon (Task Scheduler `<LogonTrigger>` with the account's `UserId`), with no network condition: a network may not be up yet at the logon, and the shim waits for the platform itself | All require user opt-in: `install` prints the systemctl / launchctl / schtasks command that starts the service and never runs it. The macOS `SuccessfulExit=false` flag is load-bearing: launchd starts the daemon again after every nonzero exit, no more often than every 10 seconds (its default throttle). It doesn't watch TCC.db: while computer use is requested and Accessibility or Screen Recording isn't granted, each start runs the preflight, which starts cua-driver to check the grants, and exits 78 again, so the start after consent is granted in System Settings comes up without the user touching launchctl. See [COMPUTER_USE.md](COMPUTER_USE.md#macos). |
| Logs | `~/Library/Logs/autogpt-local-executor/*.log` | `journalctl --user -u autogpt-shim` + `$XDG_STATE_HOME/autogpt-local-executor/` | `%LOCALAPPDATA%\autogpt-local-executor\logs\` | Audit log location per row "Audit log" below. |
| Restart after an exit | launchd, on any non-zero exit (`KeepAlive: {SuccessfulExit=false}`), at most every 10 s | systemd, on failure after 5 s (`Restart=on-failure`), except after 77 and 78 | The shim itself: the task runs `autogpt-shim start --supervise` | Task Scheduler doesn't start a program again after it exits. Its restart-on-failure setting applies when the action can't be started at all; a program that starts and then exits, with whatever code, is a run that completed. So under the task the shim runs its daemon as a child process and starts it again after an exit that asks for it (75) or a crash, with a pause of 5 s that doubles up to a minute while the daemon doesn't stay up for a minute. 0, 77 and 78 end it, as under systemd. A plain `autogpt-shim start` in a terminal has no such loop. |
| Headless server / no DM | Works (still has launchd) | Works (with `enable-linger`) | Works | None require a logged-in GUI session for the WebSocket loop. |

#### The Windows task's account

The Windows task is one account's own. `install` asks `whoami` for the SID
of the account it runs as and writes it into the task's logon trigger and
its principal. So the task starts the shim at that account's logon and runs
it as that account, with that account's Credential Manager. Such a task is
the account's to register: the `schtasks /Create` command that `install`
prints takes no elevated prompt, and neither does `autogpt-shim uninstall`.
If the account can't be read, `install` writes nothing and says why.

A trigger and a principal that name no account make a task for every
account's logon. Only an administrator may register that, and it runs as
whoever registered it: registered from a prompt opened with an
administrator's credentials, it started the shim at the administrator's
logon and never for the person who had installed it.

Task Scheduler keeps one task of a name for the whole computer, where
launchd and systemd keep each account's services apart. So one account on a
computer can have `\AutoGPT\Shim`. A second account can't register another
under that name, and starts the shim itself with `autogpt-shim start`. From
an elevated prompt, which sees every account's tasks, `uninstall` ends and
deletes the task of that name whichever account it belongs to.

#### Uninstalling

`autogpt-shim uninstall` stops the service, unregisters it and removes the
autostart file, in that order: the service managers' own commands for it
(`launchctl unload <plist>`, `systemctl --user disable`) read the file, so
with the file removed first the shim would run on until the next logout.
It runs the service manager itself, and waits a bounded time for each step:

| OS | What it runs | It waits |
|---|---|---|
| macOS | `launchctl bootout <domain>/co.agpt.shim`, in the login session (`gui/<uid>`) and the background one (`user/<uid>`, where an agent loaded over SSH is), then asks launchd until the service has gone | 30 s (launchd kills a service that hasn't exited after 20 s) |
| Linux, WSL2 | `systemctl --user disable --now autogpt-shim.service` while the unit file is there, or `systemctl --user stop` when it is already gone; then `systemctl --user daemon-reload` | 100 s (systemd kills a unit that hasn't stopped after 90 s) |
| Windows | `schtasks /End` and then `schtasks /Delete` on `\AutoGPT\Shim`. Deleting alone would leave what the task started running | 30 s |

It stops only the service the autostart entry started. A shim started by
hand in a terminal isn't the service manager's, and keeps running.

The LaunchAgent's label is `co.agpt.shim`: a launchd label is the reversed
name of a domain its author owns, and the project's is agpt.co. It had
another before. A Mac that installed the service under that earlier label
still has it, and it would start a second shim for the same computer at
every login beside the new one, where the platform lets one connection per
computer stand. So on macOS `install` and `uninstall` also ask launchd for a
service under the earlier label (`FORMER_LAUNCHD_LABEL` in `install.py`),
unload it and remove its file, and say so; with none they say nothing.
`install` does it once the new entry is written, so a write that fails
leaves the earlier service as it was. If the earlier service can't be
unloaded, launchd can't be asked about it, or its file can't be removed,
`install` says so before the new one is loaded, and `uninstall` exits 1,
with the command that stops it. That code goes once no Mac has such a
service.

When the service can't be stopped, it says what failed, removes the file
all the same, prints one command that stops the service without the file,
and exits 1. That happens when the service manager's tool isn't on `PATH`,
when there is no user session bus to reach systemd through (a shell from
`su` or `sudo -u`), when the wait runs out, and on Windows when a prompt that
isn't elevated sees the task but may not end or delete it, as it may not
for one that was registered from an elevated prompt.
With nothing installed and nothing running it changes nothing and exits 0.

On Windows a prompt that isn't elevated sees the account's own task, the one
`install` has it register, and doesn't see another account's. To such a
prompt a task it can't see looks like one that isn't registered. `uninstall`
then says that this account has no task registered, and what such a prompt
can't see, and exits 0: another account's task isn't this account's service.

The systemd user unit sets `AUTOGPT_SHIM_SUPERVISOR=systemd`. It starts at
login, before the desktop exports its display to the user manager. So when
the service started without `DISPLAY` or `WAYLAND_DISPLAY`, the shim takes
them and the rest from the manager once they are there and follows them
across logins; a service that started with a display keeps its own. See
[COMPUTER_USE.md](COMPUTER_USE.md#linux).

An authentication failure (`start` exits 77, `EX_NOPERM`) stops the
service instead of restarting it, since a restart would only retry the
rejected credentials. The systemd unit sets `RestartPreventExitStatus=77`.
launchd's `KeepAlive` restarts any non-zero exit, so the LaunchAgent sets
`AUTOGPT_SHIM_SUPERVISOR=launchd` and `start` exits 0 for this case.
Under the Windows task the shim's own supervisor (`start --supervise`)
doesn't start the daemon again after 77 either. Run
`autogpt-shim auth`, then restart the service. Errors the shim won't
retry, such as a protocol mismatch or a revoked session, already exit 0
and are logged.

A keychain that can't be read yet (still locked at login, say) is not a
failed sign-in: the shim keeps retrying with its reconnect backoff. Nor is
no keychain at all when it reads its sign-in, as with a Secret Service the
desktop session starts after the systemd unit (once the audit key came from
the file fallback, say): the shim couldn't have signed in without one, so it
waits for it, by exiting 75 (below), since the keyring package picks its
backend once per process and only a new one finds a keychain that started
later. The systemd unit doesn't restart on 78
either (`RestartPreventExitStatus=77 78`): a bad config or a preflight
failure fails the same way every time, and Linux has no permission prompt
to wait for. Under launchd, a bad config and a preflight failure no
restart can fix (a preview turned on, no cua-driver) exit 0 for the same
reason. A missing macOS permission still exits 78, so launchd relaunches
the shim once it is granted.

A failure at start that can clear on its own exits 75 (`EX_TEMPFAIL`),
which systemd and launchd both restart, and the shim's supervisor does
under the Windows task: audit storage that is full or
not mounted yet, or a keychain still locked when the audit key is read.
So does no keychain at all when the audit key or the sign-in is read, as
with a Secret Service the desktop session starts after the systemd unit:
the shim can't run without one, so it waits for it. The audit-key file fallback stands
in only with `AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE` set: once its key exists,
or while there is no audit log yet that the keychain's key signed and a
new process, which picks keyring's backend afresh, finds no keychain
either. If that process finds one, the start exits 75 like the rest.
An audit log that fails verification can't clear that way, so under
launchd it exits 0 like the other permanent failures; check it with
`autogpt-shim audit verify` and move it aside to start a new log. Nor can
a keyring backend that `PYTHON_KEYRING_BACKEND` or keyring's configuration
file names but that can't be loaded (no such module or class): fix or
remove the setting. Nor can audit storage the shim may not write (permission denied, read-only) or
whose device failed or is gone (an I/O error): fix it, then start the
shim again. A legacy session's `allowed_root` that can't be created is
sorted the same way: a full disk restarts; no permission, a read-only
disk or a file in its place doesn't. The exception is a folder the app may
not open yet on macOS (Files and Folders), a missing permission that exits
78, so launchd relaunches the shim until it is granted.

### Packaging & install

| Dimension | macOS | Linux | Windows | Notes |
|---|---|---|---|---|
| Primary installer | `pipx` | `pipx` | `pipx` | All paths go through pipx in v0. Until the first PyPI release, install from git, at the revision the deployment's setup card names: `pipx install "git+https://github.com/Significant-Gravitas/autogpt-local-executor.git@<revision>"`. After it, `pipx install autogpt-local-executor==<version>` works too. |
| Secondary | Homebrew tap (`brew install Significant-Gravitas/tap/autogpt-local-executor`) | distro packages (deb/rpm) — best-effort | Scoop / winget — best-effort | Future. |
| Signing | Apple Developer ID + notarization for `.pkg` | none (pipx wheels are unsigned) | Authenticode signing for `.exe` wrapper (SmartScreen) | Required if we ship native installers. |
| Python runtime | 3.11+ via pyenv/pipx-bundled | 3.11+ via distro or pyenv | 3.11+ via python.org or Microsoft Store | Pin to 3.11 minimum in `pyproject.toml`. |

### Keychain / credential storage

| Dimension | macOS | Linux | Windows | Notes |
|---|---|---|---|---|
| Backend | Keychain Services | Secret Service (D-Bus) — gnome-keyring or kwallet | Windows Credential Manager | Via the `keyring` library. |
| Headless fallback | Requires usable keyring | **`keyring` fails** if no D-Bus / no keyring daemon | Requires usable keyring | OAuth tokens have no file fallback. Configure a supported keyring before pairing. The legacy audit-key fallback does not store OAuth tokens. |
| WSL2 | n/a | Use a supported Linux keyring; none is configured by default in many distros | n/a | Pairing requires working keyring storage. |

### Machine ID

HELLO's `machine_id` is the shim's own ID for the computer: the platform
knows the computer by it, and every audit record and folder grant carries
it. The shim derives it (HMAC-SHA256, written as a UUID) from two things
together, and sends, logs and prints neither:

- **The operating system's identifier**, in the table below. Other software
  on the computer reads the same value, and `machine-id(5)` asks
  applications not to send it over a network, so it is only an input.
- **A random value the shim makes itself** and keeps in its own folder:
  `~/.autogpt-local-executor/machine-id-seed`
  (`%USERPROFILE%\.autogpt-local-executor\machine-id-seed` on Windows; the
  folder is `AUTOGPT_SHIM_FALLBACK_DIR` when that is set).

With no identifier to read, the ID comes from the stored value alone. The
hostname is never part of it.

An identifier that is there and couldn't be read at one start is another
matter. macOS's is read by running `ioreg`, which can fail or not answer in
its five seconds at a busy login. The ID derived without it would be
another one, and for that start the platform would see a new computer, so
the start stops with exit 75 and the next one reads it again. On Linux and
Windows, a file or a registry value that isn't there, or that the account
may not read, stays that way from one start to the next, so it counts as no
identifier, and the ID is the same each time. A read that fails any other
way there (an I/O error) stops the start like the Mac's.

| Dimension | macOS | Linux | Windows | Notes |
|---|---|---|---|---|
| The operating system's identifier | `ioreg -rd1 -c IOPlatformExpertDevice` → `IOPlatformUUID` | `/etc/machine-id` (fallback `/var/lib/dbus/machine-id`) | Registry `HKLM\SOFTWARE\Microsoft\Cryptography\MachineGuid` | An input of the ID only: never sent, logged, printed or stored by the shim. |
| It changes on | Logic-board replacement or fresh macOS install | `systemd-machine-id-setup --commit` (rare) | OS reinstall | The machine ID changes with it. |
| WSL2 | n/a | Has its own `/etc/machine-id`, distinct from the Windows host's, and its own home folder | n/a | A Windows host and a WSL2 distribution on it are two computers, by design. |
| Form the derivation takes | 32 hex digits, lower case | same | same | The shim drops the space around the identifier, braces and dashes and lowers its case first, so one computer can't get two IDs from how its identifier is written. Anything that isn't 32 hex digits (an empty file, systemd's `uninitialized`) counts as no identifier. |
| Wire format of the machine ID | UUID string, lowercase, dashes | same | same | |

The derivation is fixed: a change would give every computer a new ID.
`tests/test_machine_identity.py` pins it with known inputs and outputs.

**What counts as one computer.** One account's shim on one installation of
an operating system. Each account's shim keeps its own stored value, so two
accounts on one computer have two IDs, and one person's several computers
each have their own:

- two computers whose operating systems hold the same identifier (made
  from one image) differ by their stored values;
- two computers that share a home folder (an NFS home, a roaming Windows
  profile) share the stored value and differ by their identifiers;
- two computers that share a home folder *and* hold the same identifier
  get the same ID, and the second to connect takes the first one's place.
  Give one of them a folder of its own with `AUTOGPT_SHIM_FALLBACK_DIR`
  (every command of that computer's shim needs the same value), or a
  `machine_id` of its own.

**Where the stored value is kept, and why there.** The folder is the one
every shim of an account uses whatever its config file and audit log: it
holds the lock a new audit key is made under, the stored sign-in's locks,
and the audit key itself where there is no keychain. So an account's shims
share one ID however they are set up, which the platform's one connection
for each computer goes by. Beside the audit log the value would follow
`audit_log_path`, and a second config would look like a second computer.
It isn't in the keychain: Windows' credential store can follow an account
between a domain's computers, and a locked keychain would hold up the ID.

**How it is made.** Once, by the first `autogpt-shim start` that needs an
ID (so by the autostart service too). No other command reads the
operating system's identifier or makes the value: `--help`, `status`,
`doctor`, `auth`, `revoke`, `install`, `uninstall` and the `audit`
commands leave the folder as it was. The value is written whole to a file
of its own, readable by its owner only, and then given its name in a way
that fails when the name is taken, so two starts at once end with one
value and one ID. That way is always one step: a hard link, else a rename
that refuses to replace (Windows' own, Linux's and macOS's with their flag
for it). On a filesystem with neither (FAT and exFAT, an SMB share without
links, sshfs) the name is made new, which fails everywhere if it exists,
and the value written under it; never a look for the name followed by a
rename, between which another start's value could take the name and be
replaced. A file under the name is therefore a whole value, or on those
last filesystems one another start is writing that moment, which a start
waits two seconds for. Once a whole value is under the name it stays, even
if syncing it to disk then fails, since another start may have read it by
then. A file there that then isn't a whole value stops
`start` (exit 78) rather than be replaced. So does a folder the value
can't be kept in (exit 78, or 75 when
a restart can fix it, such as a full disk): the shim doesn't run under an
ID made for one start, which the platform would see as a new computer each
time. The message names the file and the two ways out: point
`AUTOGPT_SHIM_FALLBACK_DIR` at a folder that can hold it (the folder a
first start needs for the audit key's lock as well), or set a `machine_id`.
No file locks are needed for it, so a home without them keeps the value
like any other.

**What keeps the ID, and what changes it.** The computer has the same ID
at every start, and after the shim is upgraded or reinstalled, for as long
as the stored value is kept and the operating system's identifier stays
the same. Deleting `machine-id-seed`, or the folder it is in, gives the
computer a new ID at the next start:

- The platform shows a new computer. It lists the shims that are
  connected, so the old one is simply no longer there.
- Each chat bound to the old ID asks for its computer and folder again.
  Nothing on the computer is lost: its files and its audit log stay.
- Those chats couldn't be restored in any case once the audit key is gone
  as well. A chat's folder comes back from a grant this shim signed with a
  key made from its audit key, and the grant names the machine ID. Where
  there is no keychain the audit key is a file in the same folder, so
  deleting the folder takes both; no grant verifies under a new key,
  whatever the ID. Where the key is in the keychain and only the stored
  value is deleted, the key still verifies the grants' signatures, and
  the grants are refused all the same, for naming another machine; the
  platform doesn't find the chat's computer among the connected ones
  either.

A `machine_id` set in the config file, or with `AUTOGPT_SHIM_MACHINE_ID`,
is sent as given, and nothing above runs.

**Telling two computers apart.** The platform's folder dialog lists each
connected computer by its `display_name` and its operating system. The
display name is the hostname unless you set another, so two computers with
the same hostname look the same there until one of them gets its own: put
`display_name = "Work laptop"` in the config file, or set
`AUTOGPT_SHIM_DISPLAY_NAME`, and start the shim again. `autogpt-shim
start` prints the machine ID it runs under.

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
`--allowed-root /some/path` on the CLI or in `config.toml`. A relative path
in `config.toml` is taken from the folder the shim starts in, which for the
autostart service is the home folder (systemd) rather than the one `install`
ran in, so `install` refuses one; a relative `--allowed-root` given to
`install` is made absolute against its folder.

### OAuth callback port

| Dimension | All OSes |
|---|---|
| Default port | `41899` |
| Fallback strategy | If `41899` is in use, scan `41899–41910` and bind the first free port. |
| Listener address | Loopback only: `127.0.0.1`, and `::1` on the same port where the machine has IPv6 (the redirect says `localhost`, which a browser may try as either). Never `0.0.0.0` or `::`. A port taken on either address is skipped. |
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
    if exceeds_path_max(p):           # 1024, 4096 or 32767 characters
        return False
    if is_reserved_name(p):           # Windows: CON, PRN, NUL, COM1, ...
        return False
    if has_alternate_data_stream(p):  # Windows: "file.txt:hidden"
        return False
    if has_invalid_chars(p):          # Windows: < > " | ? * and ctrl chars;
        return False                  # elsewhere NUL

    # 3. Resolve the root, the folder the person chose, and refuse by its
    #    name a path that is plainly outside it, BEFORE anything looks at the
    #    path. Resolving `\\host\share\x` has Windows connect to the host and
    #    sign in with the person's account, `\\.\pipe\x` opens a device, and
    #    on macOS `/net/<host>/x` has autofs contact the host. A relative
    #    path is taken from the root. The path has to begin with the root
    #    as it is written and stay under it name by name, each `..` followed
    #    in its place: one that leaves the root and comes back
    #    (`/net/<host>/../../workspace/x`) would have the folder outside
    #    looked at on the way. The names a path inside the root can start
    #    with are the root as configured, the root as resolved, and the root
    #    by the home folder's own name (what `~` expands to); case, the form
    #    of an accented letter and, on Windows, the separator are set aside.
    root_resolved = os.path.realpath(allowed_root, strict=True)
    if plainly_outside(p, root_spellings(allowed_root, root_resolved)):
        return False

    # 4. Resolve symlinks, junctions, firmlinks, reparse points.
    #    Use os.path.realpath, NOT pathlib.Path.resolve() — realpath handles
    #    junctions correctly on Windows since Python 3.8; resolve() can hang
    #    on broken symlinks on some FS. A relative path is taken from the
    #    root, as the platform means one, never from the shim's working
    #    directory.
    try:
        resolved = os.path.realpath(os.path.join(allowed_root, p), strict=False)
    except OSError:
        return False

    # 5. Exact prefix with a path separator boundary.
    #    Append os.sep to prevent /foo matching /foobar.
    root_prefix = root_resolved.rstrip(os.sep) + os.sep
    if resolved == root_resolved or resolved.startswith(root_prefix):
        return True

    # 6. Same directory, different spelling. A case-insensitive filesystem
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

- **Look at nothing the name rules out.** A path that is outside the root by
  its name is refused before `realpath`, `stat` or `open` is called on it:
  another drive, share, host or device namespace on Windows, and on every
  OS an absolute path that isn't under the root or one that climbs above
  it through `..` at any point, even to come back down into it: resolving
  looks at every folder a path passes through, so where it ends doesn't
  decide what is looked at. Step 3 only ever lets a path through to be resolved; it
  never lets one in. It compares names more loosely than the filesystem
  may (any case), so what passes is still decided by steps 4 to 6.
- **The root's names are the ones the shim gave out.** The platform sends
  paths under the root the shim reported, already normalized, and a
  command's output names the resolved root. A path that reaches the root
  through some other link outside it (`/var/folders/...` for a root chosen
  as `/private/var/folders/...`, say) is refused: name it by the root. On
  Windows an 8.3 name counts where the root was configured with it
  (`C:\Users\RUNNER~1\ws`): a root given by its long name isn't reached by
  8.3 names.
- **A link inside the root is still followed.** Its name is inside, so it
  is resolved, which is how the jail learns where it points. One that
  points at a network path (`\\host\share`, `/net/<host>`) has the system
  contact that host before the request is refused. Only someone who can
  already write in the shared folder can put such a link there.
- **Always realpath both sides.** Never compare a not-yet-resolved path against
  a resolved root.
- **Always check with a separator boundary.** `/foo` must not match `/foobar`.
- **Never fold case to let a path in.** Case sensitivity varies per volume
  and, on NTFS, per directory, so neither a hardcoded "macOS = insensitive"
  nor a probe of one directory is safe. Compare the exact spelling, then
  filesystem identity. (Step 3 sets case aside only to choose what is
  looked at.) The check never writes anything.
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
| `../../../etc/passwd` | 1 (normalize) → 3 (outside by its name; nothing is looked at) |
| `/etc/passwd`, `/workspace/../etc/passwd` | 3 (outside by its name) |
| `\\host\share\x`, `//host/share/x` (Windows) | 3 (another share: no connection is made) |
| `\\.\pipe\x`, `\\.\C:\workspace\x` (Windows) | 3 (a device namespace) |
| `D:\x`, `D:x` (root on `C:`) | 3 (another drive) |
| `/net/<host>/x` (macOS autofs) | 3 (outside by its name: the host isn't contacted) |
| `/net/<host>/../../workspace/x`, `/workspace/../workspace/x` | 3 (it leaves the root on its way, though it ends inside: nothing outside is looked at) |
| `/workspace/sub/../x` | 3 (passes: its `..` stays inside the root) → 4 → 5 → allowed |
| `/workspace/symlink-to-etc` | 4 (realpath resolves outside) → 5 → 6 |
| `/WORKSPACE/foo` on case-insensitive FS | 3 (passes: case is set aside) → 5 (spelling differs) → 6 (the root is an ancestor) → allowed (correct) |
| `/WS/foo` beside root `/Ws` in a case-sensitive directory | 3 (passes) → 5 → 6 (`/WS` is another directory) |
| `/workspace2/foo` (root is `/workspace`) | 3 (the name only starts like the root's) |
| `C:\workspace\file.txt:hidden` | 2 (ADS check) |
| `C:\workspace\CON` | 2 (reserved name) |
| `\\?\C:\workspace\..\..\etc` | 2 (`?` is no character of a path) |

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
   they have separate filesystems, and each has its own `/etc/machine-id`
   and its own home folder for the shim's stored value
   ([Machine ID](#machine-id)).

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
- [PLATFORM_HOOKS.md](PLATFORM_HOOKS.md) — The platform-side adapter's
  cross-OS work list, and how each entry landed.
- [OAUTH_FLOW.md](OAUTH_FLOW.md) — Localhost callback port + keychain backends.
