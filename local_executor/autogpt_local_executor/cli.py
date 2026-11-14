"""
CLI entry point — `autogpt-shim <command>`

Commands:
    auth       Run the OAuth flow to authenticate with the AutoGPT platform
    start      Start the shim daemon (foreground)
    status     Show connection status
    revoke     Revoke OAuth tokens and disconnect
    install    Register the shim as a per-user autostart entry
    uninstall  Remove the autostart entry
    audit      Read/verify/export/rotate/prune the audit log (see subcommands)
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import errno
import logging
import os
import signal
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

if TYPE_CHECKING:
    from .config import ShimConfig


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _configure_logging(args.log_level)

    print("WARNING: This is experimental, untested software.")
    print("   It can give the AutoGPT platform access to local files and optional capabilities.")
    print("   Shell access, when explicitly enabled, is unrestricted at your user account level.")
    print("   Read docs/SECURITY.md before continuing.\n")

    # install / uninstall don't need a full config load — they only need the
    # config path so the rendered autostart file points at the right file.
    if args.command == "install":
        _cmd_install(args)
        return
    if args.command == "uninstall":
        _cmd_uninstall(args)
        return

    config = _config_or_exit(lambda: _build_config(args))

    if args.command == "auth":
        exit_code = _cmd_auth(config)
        if exit_code:
            sys.exit(exit_code)
    elif args.command == "start":
        exit_code = asyncio.run(_cmd_start(config))
        if exit_code:
            sys.exit(exit_code)
    elif args.command == "status":
        _cmd_status(config, args)
    elif args.command == "revoke":
        exit_code = asyncio.run(_cmd_revoke(config))
        if exit_code:
            sys.exit(exit_code)
    elif args.command == "doctor":
        from .doctor import run_doctor

        sys.exit(run_doctor(config))
    elif args.command == "audit":
        exit_code = _cmd_audit(config, args)
        if exit_code:
            sys.exit(exit_code)
    else:
        parser.print_help()
        sys.exit(1)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # websockets logs every handshake header at DEBUG, the bearer token
    # included, and the start of every frame, so it never goes below INFO.
    root_level = logging.getLogger().getEffectiveLevel()
    logging.getLogger("websockets").setLevel(max(logging.INFO, root_level))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autogpt-shim",
        description="EXPERIMENTAL: AutoGPT Local PC Executor shim",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_installed_version()}")
    parser.add_argument(
        "--allowed-root",
        type=Path,
        default=None,
        help="Override the directory that FILE_* operations are jailed to.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a TOML config file (defaults to per-OS location).",
    )
    parser.add_argument(
        "--platform-url",
        default=None,
        help="Base URL of the AutoGPT platform (default: https://platform.autogpt.net).",
    )
    parser.add_argument(
        "--platform-oauth-url",
        default=None,
        help="Frontend OAuth base URL, including /auth, when separate from the backend platform URL.",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Legacy direct copilot session. Omit for the persistent machine control channel.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        type=str.upper,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Python logging level.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("auth", help="Authenticate with the AutoGPT platform (OAuth)")
    start_p = sub.add_parser("start", help="Start the shim daemon")
    start_p.add_argument(
        "--session-id",
        dest="session_id",
        default=argparse.SUPPRESS,
        help="Legacy direct copilot session (also accepted globally).",
    )
    start_p.add_argument(
        "--enable-shell",
        action="store_true",
        default=False,
        help="Enable unrestricted user-level shell execution. allowed_root does not sandbox it.",
    )
    start_p.add_argument(
        "--enable-computer-use",
        action="store_true",
        default=False,
        help="Advertise the computer_use capability, served by cua-driver "
        "(see docs/CUA_DRIVER.md).",
    )
    start_p.add_argument(
        "--enable-clipboard",
        action="store_true",
        default=False,
        help="Offer cua-driver's clipboard tools (default off; see docs/COMPUTER_USE.md).",
    )
    start_p.add_argument(
        "--enable-clipboard-read-foreign",
        action="store_true",
        default=False,
        help="With --enable-clipboard, also permit reading clipboard contents "
        "the shim didn't write, except content a password manager marked private.",
    )
    start_p.add_argument(
        "--enable-recording",
        action="store_true",
        default=False,
        help="Reserved design-preview flag; currently fails startup closed on every OS.",
    )
    sub.add_parser("status", help="Show the configuration and autostart status")
    sub.add_parser("revoke", help="Revoke tokens and disconnect")
    sub.add_parser(
        "doctor",
        help="Run preflight checks (allowed_root, keychain, OS permissions, display).",
    )
    install_p = sub.add_parser(
        "install",
        help="Write a per-OS autostart entry (launchd / systemd / Task Scheduler)",
    )
    install_p.add_argument(
        "--at-login",
        action="store_true",
        default=True,
        help="Trigger autostart at login (default; v0 supports no other trigger).",
    )
    sub.add_parser(
        "uninstall",
        help="Remove the autostart entry written by `autogpt-shim install`",
    )

    audit_p = sub.add_parser("audit", help="Inspect / verify the audit log")
    audit_sub = audit_p.add_subparsers(dest="audit_command", required=True)
    audit_sub.add_parser("tail", help="Tail the current audit log (follow appends).")
    show_p = audit_sub.add_parser("show", help="Pretty-print a single record by seq.")
    show_p.add_argument("seq", type=int, help="Record sequence number.")
    audit_sub.add_parser("verify", help="Chain-verify the current audit log.")
    audit_sub.add_parser("verify-all", help="Chain-verify every audit log in the directory.")
    export_p = audit_sub.add_parser(
        "export",
        help="Zip current + rotated logs and sign the manifest for upload.",
    )
    export_p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output zip path. Defaults to ./audit-export-{timestamp}.zip.",
    )
    audit_sub.add_parser("rotate", help="Explicit rotation; starts a fresh chain.")
    prune_p = audit_sub.add_parser("prune", help="Delete rotated files older than threshold.")
    prune_p.add_argument(
        "--older-than",
        default="90d",
        help="Threshold like `90d` or `30d`. Default 90d.",
    )

    return parser


def _installed_version() -> str:
    """The installed distribution's version, which releases are tagged with;
    pyproject.toml's when running from a source tree (_version)."""
    from ._version import __version__

    return __version__


def _config_error_exit_code() -> int:
    """78 (EX_CONFIG), except under the LaunchAgent. launchd restarts any
    non-zero exit and a bad config fails the same way every time, so there
    the shim stops instead of looping."""
    if os.environ.get("AUTOGPT_SHIM_SUPERVISOR") == "launchd":
        _print_launchagent_stopped("Fix the configuration")
        return 0
    return 78


def _print_launchagent_stopped(fix: str) -> None:
    print(
        f"The LaunchAgent will not restart on its own. {fix}, then run "
        "`launchctl kickstart -k gui/$(id -u)/net.autogpt.shim`.",
        file=sys.stderr,
        flush=True,
    )


def _build_config(args):
    from .config import load_config

    overrides = {}
    if args.allowed_root is not None:
        overrides["allowed_root"] = args.allowed_root
    if args.platform_url is not None:
        overrides["platform_url"] = args.platform_url
    if args.platform_oauth_url is not None:
        overrides["platform_oauth_url"] = args.platform_oauth_url
    if args.session_id is not None:
        overrides["session_id"] = args.session_id
    # Subcommand-specific flags (start) — argparse only sets them when
    # the corresponding subparser ran, so guard with hasattr.
    if getattr(args, "enable_computer_use", False):
        overrides["enable_computer_use"] = True
    if getattr(args, "enable_shell", False):
        overrides["enable_shell"] = True
    if getattr(args, "enable_clipboard", False):
        overrides["enable_clipboard"] = True
    if getattr(args, "enable_clipboard_read_foreign", False):
        overrides["enable_clipboard_read_foreign"] = True
    if getattr(args, "enable_recording", False):
        overrides["enable_recording"] = True
    return load_config(config_path=args.config, overrides=overrides)


def _cmd_auth(config) -> int:
    from .auth import CredentialScopeError, KeychainTokenStore, OAuthDenied, OAuthFlow

    try:
        flow = OAuthFlow(config=config, token_store=KeychainTokenStore(config))
        flow.run()
    except CredentialScopeError as exc:
        print(f"Authentication failed: {exc}")
        return 1
    except OAuthDenied as exc:
        from .doctor import _printable

        # The platform's own reason, from the redirect: safe to show, with
        # what stdout's encoding can't carry (a legacy Windows code page, a
        # redirect to a file) written as escapes rather than raising.
        print(_printable(f"Sign-in was refused: {exc}"))
        return 1
    except Exception:
        print(
            "Authentication did not complete. Check the deployment configuration, "
            "network connection, and OS keychain, then run `autogpt-shim auth` again."
        )
        return 1
    return 0


async def _cmd_start(config) -> int:
    from .auth import KeychainTokenStore
    from .daemon import DaemonAuthenticationError, DaemonPreflightError, ShimDaemon

    print(f"Starting shim daemon (machine_id={config.machine_id})")
    if config.session_id and config.session_id.strip():
        print(f"Legacy direct session: {config.session_id.strip()}")
        print(f"Allowed root: {config.allowed_root}")
    else:
        print("Mode: persistent machine control (folders are selected per chat)")
    if config.enable_shell:
        print("WARNING: shell is enabled and can access anything your user account can access.")
    print("Press Ctrl+C to stop.\n")
    daemon = None
    # systemd and launchd stop the service with SIGTERM, whose default action
    # ends the process at once: no stop, no SHIM_STOP record, and a running
    # command's process group left behind. Cancel this task instead, as
    # asyncio.run does for Ctrl+C. (No asyncio signal handlers on Windows.)
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    sigterm_handled = False
    if task is not None:
        try:
            loop.add_signal_handler(signal.SIGTERM, task.cancel)
            sigterm_handled = True
        except (NotImplementedError, RuntimeError, ValueError):
            pass
    try:
        daemon = ShimDaemon(config=config, token_store=KeychainTokenStore(config))
        await daemon.run()
    except DaemonPreflightError as exc:
        print(f"Preflight failed: {exc}", file=sys.stderr, flush=True)
        if exc.transient:
            # It can clear on its own (audit storage that is full or not
            # mounted yet, a keychain still locked): 75 (EX_TEMPFAIL), which
            # systemd and launchd both restart.
            return 75
        if exc.permanent and os.environ.get("AUTOGPT_SHIM_SUPERVISOR") == "launchd":
            # A setting or a missing package fails the same way on every
            # restart, and launchd restarts any non-zero exit, so stop until
            # the user has fixed it. (The systemd unit won't restart on 78.)
            _print_launchagent_stopped("Fix the above")
            return 0
        # Exit 78 (EX_CONFIG): launchd's KeepAlive: SuccessfulExit=false
        # starts us again after any nonzero exit (at most every ten seconds),
        # and the start after the grant comes up. See COMPUTER_USE.md "macOS".
        return 78
    except DaemonAuthenticationError as exc:
        print(f"Authentication failed: {exc}", file=sys.stderr, flush=True)
        if os.environ.get("AUTOGPT_SHIM_SUPERVISOR") == "launchd":
            # The LaunchAgent's KeepAlive restarts any non-zero exit, so 77
            # would retry the rejected credentials every few seconds. Signing
            # in again is the fix, so stop here until the user has done it.
            print(
                "The LaunchAgent will not restart on its own. Run `autogpt-shim auth`, "
                "then `launchctl kickstart -k gui/$(id -u)/net.autogpt.shim`.",
                file=sys.stderr,
                flush=True,
            )
            return 0
        return 77
    except (KeyboardInterrupt, asyncio.CancelledError):
        # On Ctrl+C, asyncio.run cancels this task; KeyboardInterrupt only
        # reaches code that runs outside the event loop.
        if daemon is not None:
            await daemon.stop()
        print("\nShim stopped.")
    finally:
        if sigterm_handled:
            loop.remove_signal_handler(signal.SIGTERM)
    return 0


def _cmd_status(config, args) -> None:
    print("This shows the configuration; your service manager reports whether the daemon runs.")
    print(f"Configured platform: {config.platform_url}")
    if config.session_id and config.session_id.strip():
        print(f"Legacy session: {config.session_id.strip()}")
        print(f"Configured allowed_root: {config.allowed_root}")
    else:
        print("Mode: persistent machine control; allowed roots are selected per chat")
    print()
    _print_autostart_status()


async def _cmd_revoke(config) -> int:
    from .auth import CredentialScopeError, KeychainTokenStore, OAuthFlow

    store = KeychainTokenStore(config)
    flow = OAuthFlow(config=config, token_store=store)
    try:
        await flow.revoke_tokens()
    except Exception as exc:
        detail = (
            str(exc)
            if isinstance(exc, CredentialScopeError)
            else "Check the deployment configuration, network connection, and OS keychain."
        )
        print(
            "Token revocation did not complete; local credentials were left in place "
            f"where possible so you can retry. {detail}"
        )
        return 1
    print("Tokens revoked. Run `autogpt-shim auth` to re-authenticate.")
    return 0


def _config_or_exit(load: Callable[[], ShimConfig]) -> ShimConfig:
    """The config `load` returns, or, when it can't be loaded, a message
    saying why and an exit with the config error's status."""
    from .config import ConfigFileError

    try:
        return load()
    except ValidationError as exc:
        for error in exc.errors(include_input=False, include_context=False, include_url=False):
            field = ".".join(str(part) for part in error["loc"])
            print(f"Configuration error ({field}): {error['msg']}", file=sys.stderr)
    except ConfigFileError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
    sys.exit(_config_error_exit_code())


# The settings the service gets as options: the ones `install` was given,
# else the ones this shell's environment sets, since the service never sees
# that environment.
_SERVICE_OPTIONS = (
    ("--platform-url", "platform_url"),
    ("--platform-oauth-url", "platform_oauth_url"),
    ("--allowed-root", "allowed_root"),
    ("--session-id", "session_id"),
)
# Settings that pick the deployment, and so its saved credentials, that no
# option carries to the service.
_UNCARRIED_DEPLOYMENT_SETTINGS = ("platform_ws_url", "oauth_client_id")


def _cmd_install(args) -> None:
    from . import install as _install
    from .config import environment_settings, environment_variable

    # The service starts with the same config file and options; a set it
    # can't load would stop it at every start, where nobody sees the error.
    config = _config_or_exit(lambda: _build_config(args))
    environment = environment_settings()
    config_file = args.config if args.config is not None else _install.default_config_path()
    if (
        args.allowed_root is None
        and "allowed_root" not in environment
        and not config.allowed_root.is_absolute()
    ):
        # From the config file, which the service reads in the folder it
        # starts in (the home folder, under systemd): a legacy session there
        # would be jailed to another folder, a wider one for `.`. A flag's
        # or the environment's is made absolute against this shell's
        # (_service_global_args).
        print(
            f"Configuration error: allowed_root in {config_file} is {config.allowed_root}, "
            "a relative path, which the service would take from the folder it starts in. "
            "Make it absolute there, or pass --allowed-root.",
            file=sys.stderr,
        )
        sys.exit(_config_error_exit_code())
    uncarried = [field for field in _UNCARRIED_DEPLOYMENT_SETTINGS if field in environment]
    for field in uncarried:
        print(
            f"Configuration error: {environment_variable(field)} is set in this shell, but "
            "the service doesn't get this shell's environment, so it would look for another "
            f"deployment's credentials. Set it in {config_file} instead.",
            file=sys.stderr,
        )
    if uncarried:
        sys.exit(_config_error_exit_code())
    status = _install.install_autostart(
        at_login=args.at_login,
        config_path=args.config,
        global_args=_service_global_args(args, environment),
    )
    print(f"Wrote autostart entry: {status.target_path}")
    print(f"Platform: {status.platform}")
    if status.enable_command:
        print("\nTo activate, run:")
        print(f"  {status.enable_command}")
    if status.disable_command:
        print("\nTo deactivate later, run:")
        print(f"  {status.disable_command}")
    for note in status.notes:
        print(f"\nNote: {note}")
    carried = {field for _flag, field in _SERVICE_OPTIONS}
    for field in sorted(set(environment) - carried):
        print(
            f"\nNote: {environment_variable(field)} is set in this shell, but the service "
            f"doesn't get this shell's environment. Set it in {config_file} to keep it."
        )


def _service_global_args(args, environment: dict[str, object]) -> tuple[str, ...]:
    """The global options `start` needs too, from `install`'s own or this
    shell's environment. Without the platform URLs the service would talk to
    the default platform, with credentials scoped to another one, and fail
    at every start."""
    from . import install as _install

    out: list[str] = []
    for flag, field in _SERVICE_OPTIONS:
        value = getattr(args, field)
        if value is None:
            value = environment.get(field)
        if value is None:
            continue
        if field == "allowed_root":
            # The service runs in another working directory. A `~` the
            # shell left alone (quoted, after `=`, from the environment) is
            # the home directory, as the config reads it.
            value = _install._absolute_as_given(os.path.expanduser(str(value)))
        out += [flag, str(value)]
    return tuple(out)


def _cmd_uninstall(args) -> None:
    from . import install as _install

    status = _install.uninstall_autostart()
    if status.installed:
        # Shouldn't happen — uninstall_autostart already removed it.
        print(f"WARNING: autostart entry still present at {status.target_path}")
    else:
        print(f"Removed autostart entry (or it was already absent): {status.target_path}")
    if status.disable_command:
        print("\nIf the OS service manager already registered it, also run:")
        print(f"  {status.disable_command}")


def _print_autostart_status() -> None:
    from . import install as _install

    plat = _install._current_platform_for_install()
    status = _install.status_for(plat)
    marker = "installed" if status.installed else "not installed"
    print(f"Autostart entry ({status.platform}): {marker}")
    print(f"  path: {status.target_path}")
    if status.enable_command:
        print(f"  enable: {status.enable_command}")
    if status.disable_command:
        print(f"  disable: {status.disable_command}")
    for note in status.notes:
        print(f"  note: {note}")


# ── `audit` subcommands ───────────────────────────────────────────────────────


def _cmd_audit(config, args) -> int:
    """Dispatch one of the `audit` subcommands. Returns process exit code."""
    sub = args.audit_command
    if sub == "tail":
        return _audit_tail(config.audit_log_path)
    if sub in ("show", "verify", "verify-all", "export", "prune"):
        from .audit import AuditLockError

        try:
            if sub == "show":
                return _audit_show(config.audit_log_path, args.seq)
            if sub == "verify":
                return _audit_verify(config.audit_log_path)
            if sub == "verify-all":
                return _audit_verify_all(config.audit_log_path)
            if sub == "export":
                return _audit_export(config.audit_log_path, args.output)
            return _audit_prune(config.audit_log_path, args.older_than)
        except AuditLockError as exc:
            # Without the writers' lock a read could meet a half-written record,
            # and a prune could delete a file an export is reading.
            print(f"Could not lock the audit log: {exc}", file=sys.stderr)
            return 2
    if sub == "rotate":
        return _audit_rotate(config.audit_log_path)
    print(f"Unknown audit subcommand: {sub}")
    return 2


def _audit_tail(path: Path) -> int:
    """Print existing records then follow appends. Like `tail -f` for the
    audit log — but we strict-decode each line so a truncated final write
    is visible as a parse error rather than silently garbled output."""
    import time

    from .doctor import _printable

    if not path.is_file():
        print(_printable(f"Audit log not found: {path}"))
        return 1
    try:
        for line in _follow(path, wait=lambda: time.sleep(0.5)):
            _print_record_line(line)
    except KeyboardInterrupt:
        pass  # Ctrl+C, the only way to stop following: an end, not a failure
    return 0


def _follow(path: Path, wait) -> Iterator[bytes]:
    """Every complete line of `path`, then each one appended, like `tail -F`.
    A line still being written is held until its newline arrives. When a
    rotation moves the log away, the old file is read to its end and the new
    one at `path` is followed from its start."""
    f = _open_for_following(path)
    try:
        pending = b""
        while True:
            # Asked before the read: nothing is written to a file once a
            # rotation has moved it, so a read that comes back empty after
            # the move has seen its last record. Asked after, a record and a
            # rotation landing between the two were lost.
            moved = _moved_away(path, f)
            chunk = f.readline()
            if chunk:
                pending += chunk
                if pending.endswith(b"\n"):
                    yield pending
                    pending = b""
                continue
            if os.fstat(f.fileno()).st_size < f.tell():
                # Truncated in place: the same file, now shorter than where
                # reading had got to. Start again from its beginning.
                f.seek(0)
                pending = b""
                continue
            if moved:
                try:
                    reopened = _open_for_following(path)
                except FileNotFoundError:
                    wait()
                    continue
                if pending:
                    yield pending
                    pending = b""
                f.close()
                f = reopened
                continue
            wait()
    finally:
        f.close()


def _open_for_following(path: Path):
    """Open `path` for reading without blocking a rotation. On Windows a
    plain open() shares no delete access, so while tail held the log open
    the writer could not rename it away; this handle shares it."""
    if sys.platform != "win32":
        return open(path, "rb")
    import ctypes
    import errno
    import msvcrt
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    generic_read, open_existing, attribute_normal = 0x80000000, 3, 0x80
    share_all = 0x1 | 0x2 | 0x4  # read, write and delete
    handle = create_file(
        str(path), generic_read, share_all, None, open_existing, attribute_normal, None
    )
    if handle is None or handle == ctypes.c_void_p(-1).value:
        error = ctypes.get_last_error()
        if error in (2, 3):  # ERROR_FILE_NOT_FOUND, ERROR_PATH_NOT_FOUND
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(path))
        raise ctypes.WinError(error)
    return os.fdopen(msvcrt.open_osfhandle(handle, os.O_RDONLY), "rb")


def _moved_away(path: Path, f) -> bool:
    """Whether `path` now names another file than the one `f` has open."""
    try:
        now = os.stat(path)
    except FileNotFoundError:
        return False
    opened = os.fstat(f.fileno())
    return (now.st_dev, now.st_ino) != (opened.st_dev, opened.st_ino)


def _audit_show(current: Path, seq: int) -> int:
    """Print every record with this seq across current + rotated files. seq
    restarts at 1 in each file, so each match is printed with its file name.
    Useful for jumping to a specific event referenced in a bug report
    without leaving the terminal."""
    import json

    from .audit import holding_log_lock
    from .doctor import _printable

    found = False
    # The listing and the current log's contents share one hold of the
    # writers' lock: a rotation in between would move records to a name the
    # listing lacks. Each rotated log, which never changes but may be pruned,
    # is read under a hold of its own, and searched and printed after it, so
    # the daemon's audit writes wait neither for the scan nor for a slow
    # terminal or pager.
    with holding_log_lock(current):
        files = _audit_files(current)
        contents = _read_audit_file(files[0]) if files else None
    for index, path in enumerate(files):
        if index:
            with holding_log_lock(current):
                contents = _read_audit_file(path)
        if contents is None:
            continue
        for raw in contents.splitlines():
            if not raw.strip():
                continue
            try:
                rec = json.loads(raw.decode("utf-8"))
            except (ValueError, RecursionError):  # past Python's int or nesting limit
                continue
            if isinstance(rec, dict) and rec.get("seq") == seq:
                # Escaped where stdout can't carry a character (a Windows code
                # page, redirected), and the record as JSON's ASCII escapes,
                # which a lone surrogate from a non-UTF-8 path needs anywhere.
                print(_printable(f"# from {path.name}"))
                print(json.dumps(rec, indent=2))
                found = True
    if not found:
        print(_printable(f"No record with seq={seq} in {current} or rotated files."))
        return 1
    return 0


def _audit_verify(current: Path) -> int:
    from .audit import AuditWriter, holding_log_lock, load_audit_key

    if not current.is_file():
        print(f"Audit log not found: {current}")
        return 1
    try:
        key = load_audit_key()
    except Exception as exc:
        print(f"Could not load audit key: {exc}")
        return 2
    # The daemon may be appending; a half-written record would read as a
    # break. So the log is read whole under the writers' lock, and checked
    # after it, as verify-all does: checking parses and HMACs every record,
    # and the daemon's audit writes, and the requests that wait on them,
    # would wait for all of it.
    with holding_log_lock(current):
        contents = _read_audit_file(current)
    if contents is None:
        print(f"Audit log not found: {current}")
        return 1
    return _print_violations(current, AuditWriter.verify_records(contents, key))


def _same_file(a: Path, b: Path) -> bool:
    """Whether two paths name one file: the same spelling once resolved, or,
    when both exist, the same file under another name (a hard link, say)."""
    try:
        if a.resolve() == b.resolve():
            return True
        return a.exists() and b.exists() and os.path.samefile(a, b)
    except OSError:
        return False


def _audit_files(current: Path) -> list[Path]:
    """The current log, if there is one, then its rotated files, oldest first."""
    from .audit import list_rotated_files

    files = [current] if current.is_file() else []
    files.extend(list_rotated_files(current))
    return files


def _audit_verify_all(current: Path) -> int:
    from .audit import AuditWriter, holding_log_lock, load_audit_key

    try:
        key = load_audit_key()
    except Exception as exc:
        print(f"Could not load audit key: {exc}")
        return 2
    # Each file is read whole under the writers' lock, so no record is half
    # written and no prune removes the file part way, and checked after it:
    # checking parses and HMACs every record, and the daemon's audit writes
    # wait on that lock. The listing and the read of the current log share one
    # hold, since a rotation between them would move records to a name the
    # listing lacks. A rotated file never changes, though a prune may remove it.
    with holding_log_lock(current):
        files = _audit_files(current)
        contents = _read_audit_file(files[0]) if files else None
    if not files:
        print("No audit logs found.")
        return 0
    failures = 0
    checked = 0
    for index, path in enumerate(files):
        if index:
            with holding_log_lock(current):
                contents = _read_audit_file(path)
        if contents is None:
            print(f"{path.name}: removed while the check ran")
            continue
        violations = AuditWriter.verify_records(contents, key)
        checked += 1
        if violations:
            failures += 1
        _print_violations(path, violations)
    if failures:
        print(f"\n{failures} of {checked} files have violations.")
        return 1
    print(f"\nAll {checked} files verified clean.")
    return 0


def _read_audit_file(path: Path) -> bytes | None:
    """The contents of an audit log file, or None when it's gone."""
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


# The names an export archive keeps its manifest and signature under.
_EXPORT_METADATA = frozenset({"manifest.json", "manifest.sig"})


def _audit_export(current: Path, output: Path | None) -> int:
    """Zip current + rotated files and sign the manifest with the audit key.

    The manifest pins each file's size + SHA-256, and is itself HMACed with
    the audit key. An operator with the key (the user) can prove the bundle
    is intact. We use HMAC (not RSA) because the audit key is symmetric;
    the trust path is "user shares key out-of-band with support"."""
    import datetime
    import hashlib
    import hmac
    import json
    import zipfile

    from .audit import (
        KEYCHAIN_AUDIT_KEY_USERNAME,
        _file_identity,
        _lock_path,
        holding_log_lock,
        load_audit_key,
    )
    from .fsops import temporary_sibling

    try:
        key = load_audit_key()
    except Exception as exc:
        print(f"Could not load audit key: {exc}")
        return 2

    # The default name has one-second precision, so a second export in that
    # second gets a numbered name rather than replacing the first archive.
    numbered = output is None
    if output is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        output = Path.cwd() / f"audit-export-{stamp}.zip"

    manifest: dict = {
        "exported_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "files": [],
        "key_username": KEYCHAIN_AUDIT_KEY_USERNAME,
    }
    # Take one consistent set under the lock, then read the rotated logs and
    # write the archive after releasing it: appends wait on that lock, and a
    # slow destination would hold them, and the requests behind them, until
    # the export finished. The current log, which grows, is read into memory.
    # A rotated log never changes, so its identity is noted and checked when
    # it is read, one at a time: holding them all open ran out of file
    # descriptors with enough of them. A log pruned or replaced in between
    # fails the export, which then writes nothing.
    current_data = b""
    listed: list[tuple[Path, tuple[int, int, int] | None]] = []
    with holding_log_lock(current):
        files = _audit_files(current)
        if not files:
            print("Nothing to export — no audit logs present.")
            return 1
        # The finished archive replaces `output`, so it must not be a log being exported.
        if any(_same_file(output, path) for path in (current, *files)):
            print(f"Refusing to write the export over an audit log: {output}")
            return 2
        # Nor the writers' lock: a writer that then opened the new file would
        # take a lock the daemon, holding the old one, doesn't, and both
        # would append.
        if _same_file(output, _lock_path(current)):
            print(f"Refusing to write the export over the audit log's lock: {output}")
            return 2
        # Nor named like a rotated log beside it, which verify-all would read.
        if _same_file(output.parent, current.parent) and output.name.startswith(current.name + "."):
            print(f"Refusing to name the export like a rotated audit log: {output}")
            return 2
        # Nor holding a log under a name the archive's manifest takes: two
        # members of one name, where a reader by name gets the manifest, or,
        # unpacked where case is ignored, one in place of the other.
        for path in files:
            if path.name.casefold() in _EXPORT_METADATA:
                print(f"Refusing to export an audit log named like the export's manifest: {path}")
                return 2
        for path in files:
            if path == current:
                current_data = path.read_bytes()
            listed.append((path, None if path == current else _file_identity(path)))
    # Built beside the output and moved into place when complete, so a failed
    # export leaves nothing behind. Owner-only, like the logs it holds,
    # whatever the umask.
    temp = temporary_sibling(output)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(temp, flags, 0o600)
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        with (
            os.fdopen(descriptor, "wb") as archive,
            zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf,
        ):
            for path, identity in listed:
                # A rotated log whose identity couldn't be read fails the check.
                data = current_data if path == current else _read_unchanged(path, identity)
                digest = hashlib.sha256(data).hexdigest()
                manifest["files"].append({"name": path.name, "size": len(data), "sha256": digest})
                zf.writestr(path.name, data)
            manifest_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
            signature = hmac.new(key, manifest_bytes, hashlib.sha256).hexdigest()
            zf.writestr("manifest.json", manifest_bytes)
            zf.writestr("manifest.sig", signature)
        output = _install_export(temp, output, numbered=numbered)
    except _ExportSetChanged as exc:
        temp.unlink(missing_ok=True)
        print(
            f"{exc} was pruned or replaced during the export, so nothing was written. "
            "Run the export again."
        )
        return 1
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    print(f"Wrote {output}")
    print(f"  files: {len(files)}")
    print(f"  signature (HMAC-SHA256): {signature}")
    return 0


class _ExportSetChanged(Exception):
    """A rotated log listed for an export is gone, or is another file."""


def _read_unchanged(path: Path, identity: tuple[int, int, int] | None) -> bytes:
    """The contents of a rotated log, if it is still the file listed."""
    try:
        with open(path, "rb") as source:
            info = os.fstat(source.fileno())
            if (info.st_dev, info.st_ino, info.st_size) != identity:
                raise _ExportSetChanged(path.name)
            return source.read()
    except FileNotFoundError:
        raise _ExportSetChanged(path.name) from None


def _install_export(temp: Path, output: Path, *, numbered: bool) -> Path:
    """Move a finished archive to `output`. An --output that exists is
    replaced, as asked; the default name gets a number instead."""
    if not numbered:
        os.replace(temp, output)
        return output
    from .fsops import rename_no_replace

    candidate = output
    for number in range(1, 1000):
        try:
            rename_no_replace(temp, candidate)
            return candidate
        except FileExistsError:
            candidate = output.with_name(f"{output.stem}-{number}{output.suffix}")
    raise FileExistsError(errno.EEXIST, "no free name for the export", str(output))


def _audit_rotate(current: Path) -> int:
    from .audit import AuditLogCorruptError, AuditWriter, load_audit_key

    if not current.is_file():
        print(f"Audit log not found: {current}")
        return 1
    try:
        key = load_audit_key()
    except Exception as exc:
        print(f"Could not load audit key: {exc}")
        return 2
    # Opening the log verifies it first, as every writer does.
    try:
        writer = AuditWriter(path=current, audit_key=key)
    except AuditLogCorruptError as exc:
        print(
            f"{exc}: {current}. Check it with `autogpt-shim audit verify`, and move it "
            "aside to start a new log."
        )
        return 1
    except OSError as exc:
        print(f"Could not open {current}: {exc}")
        return 1
    try:
        rotated = writer.force_rotate()
    except OSError as exc:
        print(f"Could not rotate {current}: {exc}")
        return 1
    if rotated is None:
        print("Audit log was empty; nothing to rotate.")
        return 0
    print(f"Rotated to {rotated}")
    return 0


def _audit_prune(current: Path, older_than: str) -> int:
    """Delete rotated files where the *newest* record is older than `older_than`.

    Threshold parser accepts `Nd` (days), `Nh` (hours), `Nm` (minutes). The
    shim never auto-prunes — this is operator-initiated."""
    import time

    seconds = _parse_duration(older_than)
    if seconds is None:
        print(f"Could not parse duration: {older_than!r}. Use e.g. 90d, 12h, 30m.")
        return 2
    cutoff = time.time() - seconds
    return _prune_rotated(current, cutoff)


def _prune_rotated(current: Path, cutoff: float) -> int:
    from .audit import _file_identity, holding_log_lock, list_rotated_files

    rotated = list_rotated_files(current)
    if not rotated:
        print("No rotated audit files to prune.")
        return 0
    # A rotated log never changes, so it is read without the writers' lock:
    # reading every line of every one under it held off every audit write.
    # The deletes run under the lock, so verify-all and show, which read each
    # file under it, never lose a file part way, and only of files that are
    # still the ones read. What they did is printed after the lock is let go:
    # a terminal or pipe that stops reading would hold off the audit writes.
    expired: list[tuple[Path, tuple[int, int, int], float]] = []
    for path in rotated:
        newest = _newest_record_ts(path)
        if newest is not None and newest[1] < cutoff:
            expired.append((path, *newest))
    deleted = 0
    report: list[str] = []
    with holding_log_lock(current) if expired else contextlib.nullcontext():
        for path, identity, newest_ts in expired:
            if _file_identity(path) != identity:
                report.append(f"Kept {path.name}: it changed after it was read.")
                continue
            try:
                path.unlink()
                deleted += 1
                report.append(f"Deleted {path.name} (newest_ts={newest_ts:.0f})")
            except OSError as exc:
                report.append(f"Could not delete {path.name}: {exc}")
    for line in report:
        print(line)
    print(f"\nPruned {deleted} of {len(rotated)} rotated files.")
    return 0


def _newest_record_ts(path: Path) -> tuple[tuple[int, int, int], float] | None:
    """A rotated log's identity and the latest timestamp of any record in it;
    None when it can't be read, is empty, or a record has no timestamp.

    The latest, not the last line's: a clock that went back while the file
    was written leaves its last record older than one before it, and judging
    by the last would delete the newer one with the file. A record can run
    past any fixed window (an EXECUTE_COMMAND with a long command, say), so
    every line is read whole."""
    import json

    try:
        newest_ts: float | None = None
        with open(path, "rb") as f:
            info = os.fstat(f.fileno())
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                ts = record.get("ts") if isinstance(record, dict) else None
                if ts is None:
                    print(f"Kept {path.name}: a record in it has no timestamp.")
                    return None
                # The writer writes a finite number. float() would turn
                # -Infinity, as a JSON token or a string, into a time older
                # than any cutoff, and delete the file; an integer past a
                # float's range raises OverflowError instead.
                if not isinstance(ts, int | float) or isinstance(ts, bool) or not _finite(ts):
                    print(f"Kept {path.name}: a record in it has a timestamp that isn't a time.")
                    return None
                newest_ts = float(ts) if newest_ts is None else max(newest_ts, float(ts))
        if newest_ts is None:
            return None
        return (info.st_dev, info.st_ino, info.st_size), newest_ts
    except (OSError, TypeError, ValueError, RecursionError):
        # RecursionError: a line nested past Python's limit, read as unreadable.
        return None


def _finite(number: int | float) -> bool:
    import math

    try:
        return math.isfinite(number)
    except OverflowError:
        return False


def _parse_duration(s: str) -> float | None:
    """Tiny duration parser: `90d`, `12h`, `30m`, `45s`."""
    if not s:
        return None
    units = {"d": 86400, "h": 3600, "m": 60, "s": 1}
    suffix = s[-1].lower()
    if suffix not in units:
        return None
    number = s[:-1]
    # A positive whole number only: `-1d` would put the cutoff a day ahead,
    # and prune would delete every rotated file.
    if not (number.isascii() and number.isdigit()):
        return None
    try:
        value = int(number)
    except ValueError:
        return None  # more digits than int() converts (4300 by default)
    if value == 0:
        return None
    seconds = value * units[suffix]
    # The cutoff is a float: one past its range would raise OverflowError.
    return seconds if _finite(seconds) else None


def _print_record_line(raw: bytes) -> None:
    import json

    # Escaped where stdout can't carry a character (a Windows code page,
    # redirected), as _audit_show does: a truncated record's CJK path, or the
    # U+FFFD a byte that isn't UTF-8 reads as, ended tail with
    # UnicodeEncodeError at the damaged record a person needs to see.
    from .doctor import _printable

    s = raw.decode("utf-8", errors="replace").rstrip("\n")
    if not s.strip():
        return
    try:
        rec = json.loads(s)
    except (ValueError, RecursionError):
        # ValueError too for an integer past Python's limit; RecursionError
        # for a line nested past it.
        rec = None
    if not isinstance(rec, dict):
        print(_printable(f"[unparseable] {s}"))
        return
    # As text: a null, list or object seq or op can't take a width.
    seq = str(rec.get("seq", "?"))
    op = str(rec.get("op", "?"))
    result = rec.get("result")
    if not isinstance(result, dict):
        result = {}
    ok = result.get("ok")
    marker = "ok" if ok else f"err:{result.get('error_code')}"
    print(_printable(f"seq={seq:>5}  {op:<22}  {marker}"))


def _print_violations(path: Path, violations: list) -> int:
    """Pretty-print a verify result. Returns shell exit code (0 clean, 1 dirty)."""
    if not violations:
        print(f"{path.name}: OK")
        return 0
    print(f"{path.name}: {len(violations)} violation(s)")
    for v in violations:
        seq_part = f"seq={v.seq}" if v.seq is not None else "seq=?"
        print(f"  [{v.kind}] {seq_part}: {v.message}")
    return 1
