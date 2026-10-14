"""Claude Code subscription auth helpers.

Handles locating the SDK-bundled CLI binary, provisioning credentials from
environment variables, and validating that subscription auth is functional.
"""

import contextlib
import functools
import json
import logging
import os
import shutil
import stat
import subprocess

from backend.copilot.sdk.launch_env import inherited_env
from backend.copilot.sdk.route_credentials import subscription_env

logger = logging.getLogger(__name__)


def find_bundled_cli() -> str:
    """Locate the Claude CLI binary bundled inside ``claude_agent_sdk``.

    Falls back to ``shutil.which("claude")`` if the SDK bundle is absent.
    """
    try:
        from claude_agent_sdk._internal.transport.subprocess_cli import (
            SubprocessCLITransport,
        )

        path = SubprocessCLITransport._find_bundled_cli(None)  # type: ignore[arg-type]
        if path:
            return str(path)
    except Exception:
        pass
    system_path = shutil.which("claude")
    if system_path:
        return system_path
    raise RuntimeError(
        "Claude CLI not found — neither the SDK-bundled binary nor a "
        "system-installed `claude` could be located."
    )


def provision_credentials_file() -> None:
    """Write ``~/.claude/.credentials.json`` from env when running headless.

    If ``CLAUDE_CODE_OAUTH_TOKEN`` is set (an OAuth *access* token obtained
    from ``claude auth status`` or extracted from the macOS keychain), this
    helper writes a minimal credentials file so the bundled CLI can
    authenticate without an interactive ``claude login``.

    A ``CLAUDE_CODE_REFRESH_TOKEN`` env var is optional but recommended —
    it lets the CLI silently refresh an expired access token.

    The directory is the owner's alone (0700) and the file is created
    owner-only (0600), exclusively and without following a link. An
    existing file (e.g. a volume mount) is kept, once it is a regular file
    only its owner can read; anything else is refused.
    """
    access_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip()
    if not access_token:
        return

    creds_dir = os.path.expanduser("~/.claude")
    creds_path = os.path.join(creds_dir, ".credentials.json")
    _ensure_private_dir(creds_dir)

    # Don't overwrite an existing credentials file (e.g. from a volume mount).
    try:
        existing = os.lstat(creds_path)
    except FileNotFoundError:
        existing = None
    if existing is not None:
        _require_private_file(creds_path, existing)
        logger.debug("Credentials file already exists at %s — skipping", creds_path)
        return

    creds = {
        "claudeAiOauth": {
            "accessToken": access_token,
            "refreshToken": os.environ.get("CLAUDE_CODE_REFRESH_TOKEN", "").strip(),
            "expiresAt": 0,
            "scopes": [
                "user:inference",
                "user:profile",
                "user:sessions:claude_code",
            ],
        }
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(creds_path, flags, 0o600)
    except FileExistsError:
        raise RuntimeError(
            f"{creds_path} appeared while the Claude credentials were being "
            "written; not writing them"
        ) from None
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(creds, f)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(creds_path)
        raise
    logger.info("Provisioned Claude credentials file at %s", creds_path)


def _ensure_private_dir(path: str) -> None:
    """Create *path* owner-only, or hold an existing one to that."""
    os.makedirs(path, mode=0o700, exist_ok=True)
    info = os.stat(path)
    if not stat.S_ISDIR(info.st_mode):
        raise RuntimeError(f"{path} is not a directory; not writing Claude credentials")
    if os.name != "posix":
        return
    if info.st_uid != os.geteuid():
        raise RuntimeError(
            f"{path} belongs to another user; not writing Claude credentials"
        )
    if stat.S_IMODE(info.st_mode) & 0o077:
        os.chmod(path, 0o700)


def _require_private_file(path: str, info: os.stat_result) -> None:
    """Refuse an existing credentials file anyone but its owner could read
    or replace, or one that is not a plain file."""
    if stat.S_ISLNK(info.st_mode):
        raise RuntimeError(f"{path} is a symbolic link; refusing to use it")
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"{path} is not a regular file; refusing to use it")
    if os.name != "posix":
        return
    if info.st_uid != os.geteuid():
        raise RuntimeError(f"{path} belongs to another user; refusing to use it")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise RuntimeError(
            f"{path} is readable or writable by other users; "
            "make it owner-only (chmod 600) to use it"
        )


@functools.cache
def validate_subscription() -> None:
    """Validate the bundled Claude CLI is reachable and authenticated.

    Cached so the blocking subprocess check runs at most once per process
    lifetime.  On first call, also provisions ``~/.claude/.credentials.json``
    from the ``CLAUDE_CODE_OAUTH_TOKEN`` env var when available.
    """
    provision_credentials_file()

    cli = find_bundled_cli()
    result = subprocess.run(
        [cli, "--version"],
        capture_output=True,
        text=True,
        timeout=10,
        env=inherited_env(),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Claude CLI check failed (exit {result.returncode}): "
            f"{result.stderr.strip()}"
        )
    logger.info(
        "Claude Code subscription mode: CLI version %s",
        result.stdout.strip(),
    )

    # Verify the CLI is actually authenticated.
    auth_result = subprocess.run(
        [cli, "auth", "status"],
        capture_output=True,
        text=True,
        timeout=10,
        env={**inherited_env(), **subscription_env()},
    )
    if auth_result.returncode != 0:
        raise RuntimeError(
            "Claude CLI is not authenticated. Either:\n"
            "  • Set CLAUDE_CODE_OAUTH_TOKEN env var (from `claude auth status` "
            "or macOS keychain), or\n"
            "  • Mount ~/.claude/.credentials.json into the container, or\n"
            "  • Run `claude login` inside the container."
        )
    try:
        status = json.loads(auth_result.stdout)
        if not status.get("loggedIn"):
            raise RuntimeError(
                "Claude CLI reports loggedIn=false. Set CLAUDE_CODE_OAUTH_TOKEN "
                "or run `claude login`."
            )
        logger.info("Claude subscription auth: method=%s", status.get("authMethod"))
    except json.JSONDecodeError:
        logger.warning("Could not parse `claude auth status` output")
