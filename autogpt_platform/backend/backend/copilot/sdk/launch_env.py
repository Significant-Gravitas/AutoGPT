"""The environment a Claude Code CLI launch starts from.

The SDK merges a launch's ``options.env`` over the backend's whole
environment, which also holds settings the CLI never reads: database URLs,
service and integration keys, telemetry credentials. A launch keeps only
what the operating system and the network need, the CLI's own configuration
(``ANTHROPIC_*`` and ``CLAUDE_CODE_*``), and its own variables: every other
parent variable is blanked in ``options.env``. The blanks are worked out
from ``os.environ`` at each launch, so a variable the backend gains later is
blanked too, without a list of names to keep up to date.

The CLI's configuration stays because the copilot's routes and operators
set the CLI up through it: the direct-Anthropic route's key and base URL
come from the parent, and every other route sets the credential variables
it uses (``sdk/env.py``).

One thing falls outside it: a variable added to ``os.environ`` between
``launch_options`` and the spawn, which the SDK merges in as it spawns. The
backend writes no secret to ``os.environ`` after import.

The SDK's own ``claude -v`` version check would start the CLI on the parent's
whole environment before the launch, so ``launch_options`` turns it off: each
launch has already checked its CLI's version (``require_verbatim_prompts_cli``).
"""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeAgentOptions

# The parent variables every CLI process keeps: the OS runtime, locale,
# proxies and CA bundles, and where the CLI keeps its configuration.
# Compared upper-cased.
INHERITED_NAMES = frozenset(
    {
        "PATH",
        "PATHEXT",
        "HOME",
        "USERPROFILE",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "APPDATA",
        "LOCALAPPDATA",
        "TEMP",
        "TMP",
        "TMPDIR",
        "SHELL",
        "LANG",
        "LANGUAGE",
        "TZ",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "ALL_PROXY",
        "NODE_EXTRA_CA_CERTS",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "CLAUDE_CONFIG_DIR",
    }
)
_INHERITED_PREFIXES = ("LC_",)
# The CLI's own configuration, which a launch keeps as well. A process that
# only asks the CLI its version does not. Other CLI settings, such as
# ``CLAUDE_AUTOCOMPACT_PCT_OVERRIDE``, reach a launch only through its own
# ``env`` (``sdk/env.py``).
CLI_CONFIG_PREFIXES = ("ANTHROPIC_", "CLAUDE_CODE_")
# Set or dropped by the SDK itself when it spawns the CLI: left alone so its
# own values stand.
SDK_MANAGED_NAMES = frozenset(
    {
        "CLAUDECODE",
        "CLAUDE_CODE_ENTRYPOINT",
        "CLAUDE_AGENT_SDK_VERSION",
        "CLAUDE_CODE_SDK_READS_SESSION_STATE",
        "CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING",
        "PWD",
        "TRACEPARENT",
        "TRACESTATE",
    }
)
# Every variable the CLI reads a Claude OAuth login from: blanked on a launch
# that carries an API key, so it never holds the host's login as well.
CLAUDE_OAUTH_BLANK: dict[str, str] = {
    "CLAUDE_CODE_OAUTH_TOKEN": "",
    "CLAUDE_CODE_OAUTH_REFRESH_TOKEN": "",
    "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR": "",
    "CLAUDE_CODE_REFRESH_TOKEN": "",
}
# The SDK's switch for its own version check, which it runs outside a launch's
# options, on the parent's whole environment.
SKIP_SDK_VERSION_CHECK = "CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK"


def inherited(name: str) -> bool:
    """Whether every CLI process keeps the parent's variable *name*."""
    upper = name.upper()
    return upper in INHERITED_NAMES or upper.startswith(_INHERITED_PREFIXES)


def kept_by_launch(name: str) -> bool:
    """Whether a launch keeps the parent's variable *name*."""
    return inherited(name) or name.upper().startswith(CLI_CONFIG_PREFIXES)


def inherited_env() -> dict[str, str]:
    """The parent variables a CLI started outside the SDK keeps, for one that
    only reports its version."""
    return {name: value for name, value in os.environ.items() if inherited(name)}


def kept_launch_env() -> dict[str, str]:
    """The parent variables a launch keeps, for a CLI command started outside
    the SDK that needs the CLI's configuration."""
    return {name: value for name, value in os.environ.items() if kept_by_launch(name)}


def isolated_launch_env(env: Mapping[str, str]) -> dict[str, str]:
    """*env* as a launch's ``options.env``: with every parent variable the
    launch does not keep blanked."""
    blanked = {
        name: ""
        for name in os.environ
        if name not in env
        and not kept_by_launch(name)
        and name.upper() not in SDK_MANAGED_NAMES
    }
    return {**blanked, **env}


def launch_options(options: ClaudeAgentOptions) -> ClaudeAgentOptions:
    """A copy of *options* with the environment a launch starts from, worked
    out now.

    Also turns the SDK's own version check off for the process, as the
    copilot executor does when it starts.
    """
    os.environ[SKIP_SDK_VERSION_CHECK] = "1"
    launched = copy.copy(options)
    launched.env = isolated_launch_env(options.env)
    return launched
