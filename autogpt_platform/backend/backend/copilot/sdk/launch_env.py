"""The environment a Claude Code CLI launch starts from.

The SDK merges a launch's ``options.env`` over the backend's whole
environment, which holds other routes' keys, integration keys, database
URLs and telemetry credentials. A launch keeps only what the operating
system and the network need from it, plus its own variables: every other
parent variable is blanked in ``options.env``. The blanks are worked out
from ``os.environ`` at each launch, so a variable the backend gains later is
blanked too, without a list of names to keep up to date.

Two things fall outside it: a variable added to ``os.environ`` between
``launch_options`` and the spawn, which the SDK merges in as it spawns, and
the SDK's own ``claude -v`` version check, which runs on the parent
environment (AutoPilot's workers skip it). The backend writes no secret to
``os.environ`` after import.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeAgentOptions

# The parent variables a launch keeps: the OS runtime, locale, proxies and CA
# bundles, and where the CLI keeps its configuration. Compared upper-cased.
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


def inherited(name: str) -> bool:
    """Whether a launch keeps the parent's variable *name*."""
    upper = name.upper()
    return upper in INHERITED_NAMES or upper.startswith(_INHERITED_PREFIXES)


def inherited_env() -> dict[str, str]:
    """The parent variables a launch keeps, for a process started with an
    environment of its own rather than through the SDK."""
    return {name: value for name, value in os.environ.items() if inherited(name)}


def isolated_launch_env(env: Mapping[str, str]) -> dict[str, str]:
    """*env* as a launch's ``options.env``: with every parent variable the
    launch does not keep blanked."""
    blanked = {
        name: ""
        for name in os.environ
        if name not in env
        and not inherited(name)
        and name.upper() not in SDK_MANAGED_NAMES
    }
    return {**blanked, **env}


def launch_options(options: ClaudeAgentOptions) -> ClaudeAgentOptions:
    """A copy of *options* with the environment a launch starts from, worked
    out now."""
    launched = copy.copy(options)
    launched.env = isolated_launch_env(options.env)
    return launched
