"""Secrets and the service environment.

A port of single-container/entrypoint.sh `configure_environment`. Secrets come
from the same generator the appliance uses (runtime_config.py), so the two
distributions can never disagree about which secrets exist. Values the user
supplies (API keys, OAuth apps) go in config/settings.env and win over
nothing generated: they are layered underneath the runtime's own wiring.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import secrets
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import quote

from autogpt_desktop.layout import Bundle, DataDir, write_private

logger = logging.getLogger("autogpt_desktop")

DB_CONNECTION_LIMIT = 5
# The connection pool's settings, with the appliance's default and bounds
# (entrypoint.sh normalize_integer): default, lowest, highest.
DB_SETTINGS = {
    "DB_CONNECTION_LIMIT": (DB_CONNECTION_LIMIT, 1, DB_CONNECTION_LIMIT),
    "DB_CONNECT_TIMEOUT": (60, 1, 600),
    "DB_POOL_TIMEOUT": (300, 1, 3600),
}
FRONTEND_DB_ROLE = "autogpt_frontend"

TELEMETRY_DISABLED_ENV = {
    "SENTRY_DSN": "",
    "DISABLE_SENTRY": "true",
    "POSTHOG_API_KEY": "",
    "POSTHOG_SECRET_KEY": "",
    "LAUNCH_DARKLY_SDK_KEY": "",
    "LANGFUSE_PUBLIC_KEY": "",
    "LANGFUSE_SECRET_KEY": "",
    "LANGFUSE_TRACING_ENABLED": "false",
    "LANGSMITH_TRACING": "false",
    "LANGSMITH_OTEL_ENABLED": "false",
    "LANGCHAIN_TRACING": "false",
    "LANGCHAIN_TRACING_V2": "false",
    "OTEL_SDK_DISABLED": "true",
    "OTEL_TRACES_EXPORTER": "none",
    "OTEL_METRICS_EXPORTER": "none",
    "OTEL_LOGS_EXPORTER": "none",
    "CLAUDE_CODE_ENABLE_TELEMETRY": "0",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "DISABLE_TELEMETRY": "1",
    "DISABLE_ERROR_REPORTING": "1",
    "MEM0_TELEMETRY": "false",
    "GRAPHITI_TELEMETRY_ENABLED": "false",
    "DATAFAST_API_KEY": "",
}
TELEMETRY_PREFIXES = (
    "SENTRY_",
    "POSTHOG_",
    "LANGFUSE_",
    "LANGSMITH_",
    "LANGCHAIN_",
    "OTEL_",
    "NEXT_PUBLIC_POSTHOG_",
    "NEXT_PUBLIC_COOKIEBOT_",
)


def without_telemetry(values: dict[str, str]) -> dict[str, str]:
    return {
        name: value
        for name, value in values.items()
        if name.upper() not in TELEMETRY_DISABLED_ENV
        and not name.upper().startswith(TELEMETRY_PREFIXES)
    }


# Never taken from settings.env. With one of them set, the backend writes a
# Claude Code credentials file into the home directory
# (backend/copilot/sdk/subscription.py), and the app neither stores a Claude
# sign-in nor asks for one: the only sign-in it uses is the one the user made
# by running `claude` themselves (claude_code.py).
CLAUDE_TOKEN_SETTINGS = ("CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_REFRESH_TOKEN")
CLAUDE_SUBSCRIPTION = "CHAT_USE_CLAUDE_CODE_SUBSCRIPTION"
# Where AutoPilot's per-session working directories go, in place of the
# backend's /tmp/copilot-<session> (build/backend_patches.py makes the backend
# read it). The path has to contain `tmp/copilot-`: the backend only sweeps
# transcript directories whose name, made from this path, has `-tmp-copilot-`.
WORKSPACE_PREFIX_VARIABLE = "COPILOT_WORKSPACE_PREFIX"

# Desktop-only secrets. runtime_config.py rejects unknown keys in its own file,
# so these live beside it rather than in it.
DESKTOP_SECRETS = ("AUTOGPT_FRONTEND_DB_PASSWORD",)

SETTINGS_TEMPLATE = """\
# AutoGPT desktop settings. Restart AutoGPT after editing.
# Connect ChatGPT through onboarding or Settings -> Integrations.
# To use API keys, uncomment and fill only the providers you use:
# OPEN_ROUTER_API_KEY=
# OPENAI_API_KEY=
# ANTHROPIC_API_KEY=
# GROQ_API_KEY=
# OpenAI-compatible local inference (for example Ollama):
# CHAT_USE_LOCAL=true
# CHAT_BASE_URL=http://127.0.0.1:11434/v1
# CHAT_API_KEY=ollama
# The first account created is the owner and an admin, and registration then
# closes by itself. To let more people create accounts (ordinary users):
# AUTH_ALLOW_NEW_ACCOUNTS=true
"""

# What the Next server is given of the backend's environment: the appliance's
# list (single-container/run-frontend.sh) without the social sign-in
# providers. Those send the whole window to the provider and back, and the
# shell keeps the window on the app (src/navigation.js).
FRONTEND_PASSTHROUGH = (
    "AGPT_SERVER_URL",
    "AGPT_WS_SERVER_URL",
    "AUTH_ALLOW_NEW_ACCOUNTS",
    "AUTH_DB_SCHEMA",
    "AUTH_REQUIRE_EMAIL_VERIFICATION",
    "AUTH_SIGNUP_ALLOWLIST",
    "BETTER_AUTH_INTERNAL_URL",
    "BETTER_AUTH_SECRET",
    "BETTER_AUTH_URL",
    "OPENAI_API_BASE_URL",
    "OPENAI_API_KEY",
    "TRANSCRIPTION_API_BASE_URL",
    "TRANSCRIPTION_API_KEY",
    "TRANSCRIPTION_MODEL",
)


def load_runtime_config_module(bundle: Bundle) -> ModuleType:
    candidates = (
        bundle.root / "assets" / "runtime_config.py",
        Path(__file__).resolve().parents[3] / "single-container" / "runtime_config.py",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise RuntimeError("runtime_config.py is missing from the bundle")
    spec = importlib.util.spec_from_file_location("autogpt_runtime_config", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if sys.platform == "win32":
        vars(module)["os"] = WindowsOs(os)
    return module


class WindowsOs:
    """`os` as the appliance's runtime_config.py needs it on Windows.

    That file is upstream's and is loaded as it is. Writing runtime.env, it
    calls os.fchmod, which Windows has only since Python 3.13, and it opens
    the config directory to fsync the rename, which Windows refuses: a
    directory is not a file there, and NTFS journals the rename anyway. The
    module reaches all of it through its global `os`, so it is handed this
    in its place. upstream/runtime-config-windows.patch is the same fix
    offered upstream; when the appliance has it, delete this class.
    """

    def __init__(self, real: Any) -> None:
        self._real = real
        self._directories: set[int] = set()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    def fchmod(self, descriptor: int, mode: int) -> None:
        change = getattr(self._real, "fchmod", None)
        if change:
            change(descriptor, mode)

    def open(self, path: str | os.PathLike[str], flags: int, mode: int = 0o777) -> int:
        if not self._real.path.isdir(path):
            return self._real.open(path, flags, mode)
        # A real descriptor stands in for the directory's, so that closing
        # it is an ordinary close; fsync is what must not reach it.
        descriptor = self._real.open(self._real.devnull, self._real.O_RDONLY)
        self._directories.add(descriptor)
        return descriptor

    def fsync(self, descriptor: int) -> None:
        if descriptor not in self._directories:
            self._real.fsync(descriptor)

    def close(self, descriptor: int) -> None:
        self._directories.discard(descriptor)
        self._real.close(descriptor)


def ensure_secrets(bundle: Bundle, data: DataDir) -> dict[str, str]:
    runtime_config = load_runtime_config_module(bundle)
    values = dict(runtime_config.ensure_runtime_config(data.runtime_env, {}))
    values.update(_ensure_desktop_secrets(data.config / "desktop.env"))
    return values


def read_user_settings(data: DataDir) -> dict[str, str]:
    path = data.config / "settings.env"
    if not path.exists():
        write_private(path, SETTINGS_TEMPLATE)
    return {key: value for key, value in _read_env(path).items() if value}


def backend_environment(
    bundle: Bundle,
    data: DataDir,
    ports: dict[str, int],
    secret: dict[str, str],
    user: dict[str, str],
    defaults: dict[str, str] | None = None,
) -> dict[str, str]:
    """`defaults` (the profile's pool sizes) give way to settings.env, which
    gives way to the runtime's own wiring: later keys win."""
    # Whatever their case: Windows has one variable for every spelling.
    user = {
        name: value
        for name, value in without_telemetry(user).items()
        if name.upper() not in CLAUDE_TOKEN_SETTINGS
    }
    public_url = f"http://127.0.0.1:{ports['public']}"
    database = _database_url(
        "postgres", secret["POSTGRES_PASSWORD"], ports["postgres"], "platform"
    )
    frontend_origin = f"http://127.0.0.1:{ports['frontend']}"
    pool = _database_settings(user)
    env = {
        **without_telemetry(defaults or {}),
        **user,
        **TELEMETRY_DISABLED_ENV,
        "AUTOGPT_DESKTOP": "1",
        **_bundled_tools(bundle),
        # Where the backend's file logging goes if settings.env turns it on.
        # Its default is a `logs` directory beside the code: in the bundle.
        "LOG_DIR": str(data.logs / "backend"),
        "AUTOGPT_PUBLIC_URL": public_url,
        "APP_ENV": "dev",
        "BEHAVE_AS": "local",
        "ENABLE_AUTH": "true",
        "AUTH_REQUIRE_EMAIL_VERIFICATION": "false",
        "JWT_VERIFY_KEY": "",
        "SUPABASE_JWT_SECRET": "",
        "BETTER_AUTH_SECRET": secret["BETTER_AUTH_SECRET"],
        "BETTER_AUTH_URL": public_url,
        "BETTER_AUTH_INTERNAL_URL": frontend_origin,
        "JWT_JWKS_URL": f"{frontend_origin}/api/auth/jwks",
        "ENCRYPTION_KEY": secret["ENCRYPTION_KEY"],
        "INTERNAL_API_KEY": secret["INTERNAL_API_KEY"],
        "UNSUBSCRIBE_SECRET_KEY": secret["UNSUBSCRIBE_SECRET_KEY"],
        "VAPID_PRIVATE_KEY": secret["VAPID_PRIVATE_KEY"],
        "VAPID_PUBLIC_KEY": secret["VAPID_PUBLIC_KEY"],
        "VAPID_CLAIM_EMAIL": user.get("VAPID_CLAIM_EMAIL", "mailto:admin@localhost"),
        **pool,
        "DATABASE_URL": f"{database}&connection_limit={pool['DB_CONNECTION_LIMIT']}"
        f"&connect_timeout={pool['DB_CONNECT_TIMEOUT']}"
        f"&pool_timeout={pool['DB_POOL_TIMEOUT']}",
        "DIRECT_URL": f"{database}&connect_timeout={pool['DB_CONNECT_TIMEOUT']}",
        "DB_HOST": "127.0.0.1",
        "DB_PORT": str(ports["postgres"]),
        "DB_USER": "postgres",
        "DB_NAME": "postgres",
        "DB_PASS": secret["POSTGRES_PASSWORD"],
        "DB_SCHEMA": "platform",
        "AUTH_DB_SCHEMA": "platform",
        "PRISMA_SCHEMA": str(bundle.backend_dir / "schema.prisma"),
        "PRISMA_QUERY_ENGINE_BINARY": str(bundle.prisma_engine("query-engine")),
        "PRISMA_SCHEMA_ENGINE_BINARY": str(bundle.prisma_engine("schema-engine")),
        "REDIS_HOST": "127.0.0.1",
        "REDIS_PORT": str(ports["valkey"]),
        "REDIS_CLUSTER_HOST": "127.0.0.1",
        "REDIS_CLUSTER_PORT": str(ports["valkey"]),
        "REDIS_PASSWORD": secret["REDIS_PASSWORD"],
        "REDIS_USE_ANNOUNCED_ADDRESS": "false",
        "RABBITMQ_HOST": "127.0.0.1",
        "RABBITMQ_PORT": str(ports["rabbitmq"]),
        "RABBITMQ_CLUSTER_HOST": "127.0.0.1",
        "RABBITMQ_CLUSTER_PORT": str(ports["rabbitmq"]),
        "RABBITMQ_VHOST": "/",
        "RABBITMQ_DEFAULT_USER": secret["RABBITMQ_DEFAULT_USER"],
        "RABBITMQ_DEFAULT_PASS": secret["RABBITMQ_DEFAULT_PASS"],
        # Graphiti's FalkorDB store is a Linux Redis module under the SSPL;
        # the desktop build ships without it.
        "FORCE_FLAG_GRAPHITI_MEMORY": "false",
        "CLAMAV_SERVICE_ENABLED": "false",
        "MEM0_TELEMETRY": "false",
        "GRAPHITI_TELEMETRY_ENABLED": "false",
        "CHAT_DAILY_COST_LIMIT_MICRODOLLARS": user.get(
            "CHAT_DAILY_COST_LIMIT_MICRODOLLARS", "-1"
        ),
        "CHAT_WEEKLY_COST_LIMIT_MICRODOLLARS": user.get(
            "CHAT_WEEKLY_COST_LIMIT_MICRODOLLARS", "-1"
        ),
        "FRONTEND_BASE_URL": public_url,
        "PLATFORM_BASE_URL": f"{public_url}/_agpt",
        "PLATFORM_LINK_BASE_URL": f"{public_url}/link",
        "BACKEND_CORS_ALLOW_ORIGINS": json.dumps([public_url]),
        "AGPT_SERVER_URL": f"http://127.0.0.1:{ports['agent_api']}/api",
        "AGPT_WS_SERVER_URL": f"ws://127.0.0.1:{ports['websocket']}/ws",
        "WORKSPACE_STORAGE_DIR": str(data.workspaces),
        **_copilot_workspaces(data),
        "NODE_ENV": "production",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONPATH": str(bundle.root / "assets" / "python"),
        "XDG_CACHE_HOME": str(data.backend_cache),
        **_home(data.home),
        # AutoPilot's coding agents keep their sign-in and transcripts under
        # the home directory. Like the appliance, the app has its own: no
        # service may pick up, or write into, the user's personal Claude Code
        # or Codex setup. The one exception is made on purpose, for one
        # process: claude_code_host_environment.
        "CLAUDE_CONFIG_DIR": str(data.home / ".claude"),
        "CODEX_HOME": str(data.home / ".codex"),
        **_service_addresses(ports),
    }
    return env


def claude_code_host_environment(
    env: dict[str, str],
    data: DataDir,
    user: dict[str, str],
    home: Path,
    cli: Path | None = None,
) -> dict[str, str]:
    """`env` (backend_environment) for the process of the copilot executor,
    which is then the only service in it (apps.copilot_alone), when AutoPilot
    runs on the user's own Claude Code sign-in (claude_code.py).

    The CLI finds that sign-in the way the user's terminal does: under the
    real home directory, with no CLAUDE_CONFIG_DIR of the app's. (The app's
    home plus CLAUDE_CONFIG_DIR=~/.claude would not do: the macOS Keychain
    entry is named after that variable, and the CLI then keeps a second copy
    of its state file.) What the services of that process would otherwise
    leave in the user's home is sent to the data directory by name; no other
    XDG_* directory is moved, because the CLI's own install lives under them.

    `cli` is the resolved CLI to use when it is not the one the SDK bundles.
    Both SDK execution and subscription validation use CHAT_CLAUDE_AGENT_CLI_PATH.
    Its directory also goes first on PATH for command discovery."""
    host = {
        **env,
        **_home(home),
        CLAUDE_SUBSCRIPTION: "true",
        # The user's own Claude Code keeps updating itself when they run it;
        # the app never updates one.
        "DISABLE_AUTOUPDATER": "1",
        "MEM0_DIR": str(data.home / ".mem0"),
    }
    del host["CLAUDE_CONFIG_DIR"]
    if user.get("CLAUDE_CONFIG_DIR"):
        host["CLAUDE_CONFIG_DIR"] = user["CLAUDE_CONFIG_DIR"]
    if cli:
        host["CHAT_CLAUDE_AGENT_CLI_PATH"] = str(cli)
        path = env.get("PATH") or os.environ.get("PATH", "")
        host["PATH"] = os.pathsep.join([str(cli.parent), path])
    return host


def _copilot_workspaces(data: DataDir) -> dict[str, str]:
    """Windows has no /tmp: the backend's default would be a `tmp` directory
    at the top of whatever drive the app is installed on, shared by every
    user of the machine and left behind by an uninstall. macOS and Linux keep the backend's default,
    which its shell sandbox mounts by that name."""
    if sys.platform != "win32":
        return {}
    return {WORKSPACE_PREFIX_VARIABLE: str(data.root / "tmp" / "copilot-")}


def clear_copilot_workspaces(data: DataDir) -> None:
    """Remove what earlier runs left of AutoPilot's workspaces in the data
    directory. The backend deletes a session's workspace after every turn,
    but Windows refuses while the CLI that worked in it is still exiting, and
    a run that was killed deletes nothing. Called before any service starts:
    no session is at work."""
    for prefix in _copilot_workspaces(data).values():
        shutil.rmtree(Path(prefix).parent, ignore_errors=True)


def _database_settings(user: dict[str, str]) -> dict[str, str]:
    """DB_SETTINGS as the services get them. The backend reads each from the
    environment in preference to DATABASE_URL, and takes its query timeout
    from DB_POOL_TIMEOUT (backend/data/db.py), so they are exported as well as
    written into the URLs. settings.env may move one inside the appliance's
    bounds. Where the appliance refuses to start over a value outside them,
    the default is used here: a typo must not keep the app from opening."""
    chosen = {}
    for name, (default, lowest, highest) in DB_SETTINGS.items():
        value = user.get(name, "")
        if value.isascii() and value.isdigit() and lowest <= int(value) <= highest:
            chosen[name] = str(int(value))
            continue
        if value:
            logger.warning(
                f"{name}={value} in settings.env is not a number from {lowest} to "
                f"{highest}; using {default}"
            )
        chosen[name] = str(default)
    return chosen


def _bundled_tools(bundle: Bundle) -> dict[str, str]:
    """ffmpeg, for the three ways the backend looks for it: by name on PATH
    (its own subprocess calls, and yt-dlp's), and through imageio-ffmpeg,
    which reads IMAGEIO_FFMPEG_EXE before anything else. The bundle's
    directory goes first on PATH, so the app behaves the same whatever the
    machine has installed. Nothing is set for a bundle without the tool:
    imageio-ffmpeg then still finds a copy of its own, or the system's."""
    if not bundle.ffmpeg.is_file():
        logger.warning(
            f"the bundle has no {bundle.ffmpeg.name}; video tools may not work"
        )
        return {}
    return {
        "PATH": os.pathsep.join([str(bundle.tools_bin), os.environ.get("PATH", "")]),
        "IMAGEIO_FFMPEG_EXE": str(bundle.ffmpeg),
    }


def closes_registration(configured: str | None) -> bool:
    """Whether the database refuses accounts after the owner's. It does unless
    settings.env says AUTH_ALLOW_NEW_ACCOUNTS=true in so many words."""
    return (configured or "").strip().lower() != "true"


def registration_gate(configured: str | None, identities: int | None) -> str:
    """AUTH_ALLOW_NEW_ACCOUNTS as the frontend gets it: what settings.env
    says, otherwise open until the owner exists and closed from then on. The
    frontend reads it once, so it is the next start that shows a refused
    sign-up the proper message; until then the database refuses it.

    With no account at all it is open whatever settings.env says: closed, the
    install could never get its owner. `identities` is None when the accounts
    could not be counted, and then only an explicit setting closes it."""
    if identities == 0:
        if configured and closes_registration(configured):
            logger.info(
                f"AUTH_ALLOW_NEW_ACCOUNTS={configured} takes effect once the "
                "owner account exists; until then the first sign-up is allowed"
            )
        return "true"
    if configured:
        return "false" if closes_registration(configured) else "true"
    return "false" if identities else "true"


def frontend_environment(
    backend: dict[str, str],
    ports: dict[str, int],
    secret: dict[str, str],
    data: DataDir,
) -> dict[str, str]:
    database = _database_url(
        FRONTEND_DB_ROLE,
        secret["AUTOGPT_FRONTEND_DB_PASSWORD"],
        ports["postgres"],
        None,
    )
    return {
        **{name: backend[name] for name in FRONTEND_PASSTHROUGH if name in backend},
        "DATABASE_URL": f"{database}&connection_limit=10",
        "PORT": str(ports["frontend"]),
        "HOSTNAME": "127.0.0.1",
        "TRUST_X_FORWARDED_FOR": "true",
        "NODE_ENV": "production",
        "NEXT_TELEMETRY_DISABLED": "1",
        "DISABLE_SENTRY": "true",
        "OTEL_SDK_DISABLED": "true",
        "XDG_CACHE_HOME": str(data.next_cache),
        **_home(data.frontend_home),
    }


def _home(directory: Path) -> dict[str, str]:
    """The home directory, as each OS spells it. Python and Node both ignore
    HOME on Windows and read USERPROFILE."""
    home = {"HOME": str(directory)}
    if sys.platform == "win32":
        home["USERPROFILE"] = str(directory)
    return home


def _service_addresses(ports: dict[str, int]) -> dict[str, str]:
    hosts = (
        "PYRO_HOST",
        "AGENTSERVER_HOST",
        "SCHEDULER_HOST",
        "DATABASEMANAGER_HOST",
        "SWAPCREDENTIALSERVICE_HOST",
        "EXECUTIONMANAGER_HOST",
        "NOTIFICATIONMANAGER_HOST",
        "PLATFORMLINKINGMANAGER_HOST",
        "COPILOTEXECUTOR_HOST",
        "COPILOTCHATBRIDGE_HOST",
        "AGENT_API_HOST",
        "WEBSOCKET_SERVER_HOST",
    )
    return {
        **{name: "127.0.0.1" for name in hosts},
        "WEBSOCKET_SERVER_PORT": str(ports["websocket"]),
        "EXECUTION_MANAGER_PORT": str(ports["execution_manager"]),
        "EXECUTION_SCHEDULER_PORT": str(ports["execution_scheduler"]),
        "DATABASE_API_PORT": str(ports["database_api"]),
        "SWAP_CREDENTIAL_SERVICE_PORT": str(ports["swap_credential"]),
        "AGENT_API_PORT": str(ports["agent_api"]),
        "NOTIFICATION_SERVICE_PORT": str(ports["notification"]),
        "COPILOT_EXECUTOR_PORT": str(ports["copilot_executor"]),
        "PLATFORM_LINKING_SERVICE_PORT": str(ports["platform_linking"]),
        "COPILOT_CHAT_BRIDGE_PORT": str(ports["copilot_chat_bridge"]),
        "BATCH_EXECUTOR_PORT": str(ports["batch_executor"]),
    }


def _database_url(user: str, password: str, port: int, schema: str | None) -> str:
    query = f"schema={schema}" if schema else "sslmode=disable"
    return f"postgresql://{user}:{quote(password, safe='')}@127.0.0.1:{port}/postgres?{query}"


def _ensure_desktop_secrets(path: Path) -> dict[str, str]:
    values = _read_env(path) if path.exists() else {}
    missing = [name for name in DESKTOP_SECRETS if not values.get(name)]
    if missing:
        values.update({name: secrets.token_urlsafe(36) for name in missing})
        write_private(path, "".join(f"{k}={v}\n" for k, v in values.items()))
    return values


def _read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        values[name.strip()] = value.strip()
    return values
