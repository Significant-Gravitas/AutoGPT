"""Claude Code subscription auth helpers.

Handles locating the SDK-bundled CLI binary, provisioning credentials from
environment variables, and validating that subscription auth is functional.
"""

import functools
import json
import logging
import os
import shutil
import subprocess

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
    """
    access_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip()
    if not access_token:
        return

    creds_dir = os.path.expanduser("~/.claude")
    creds_path = os.path.join(creds_dir, ".credentials.json")

    # Don't overwrite an existing credentials file (e.g. from a volume mount).
    if os.path.exists(creds_path):
        logger.debug("Credentials file already exists at %s — skipping", creds_path)
        return

    os.makedirs(creds_dir, exist_ok=True)

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
    with open(creds_path, "w") as f:
        json.dump(creds, f)
    logger.info("Provisioned Claude credentials file at %s", creds_path)


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
        env={
            **os.environ,
            "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_AUTH_TOKEN": "",
            "ANTHROPIC_BASE_URL": "",
        },
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
        logger.info(
            "Claude subscription auth: method=%s, email=%s",
            status.get("authMethod"),
            status.get("email"),
        )
    except json.JSONDecodeError:
        logger.warning("Could not parse `claude auth status` output")
