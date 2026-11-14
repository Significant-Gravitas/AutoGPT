"""
ShimConfig — runtime configuration for the shim.

Loaded from (highest precedence first):
1. CLI flags (constructed via ShimConfig(...) overrides)
2. Environment variables (AUTOGPT_SHIM_*)
3. TOML file at ~/.config/autogpt-local-executor/config.toml (or per-OS equivalent)
4. Built-in defaults below

Per-OS defaults for allowed_root and audit_log_path follow the tables in
docs/CROSS_PLATFORM.md "Default allowed_root" and "Audit log location".
"""

from __future__ import annotations

import ipaddress
import json
import re
import socket
import stat
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, EnvSettingsSource, SettingsConfigDict

from . import platform_info


def _validate_platform_endpoint(value: str, schemes: set[str]) -> str:
    if any(ord(char) <= 32 or ord(char) == 127 for char in value) or "\\" in value:
        raise ValueError("Platform endpoint contains invalid URL characters")
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        raise ValueError("Platform endpoint has an invalid host or port") from None
    if (
        parsed.scheme not in schemes
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "Platform endpoint requires a supported URL without credentials, query, or fragment"
        )
    if port is not None and port < 1:
        raise ValueError("Platform endpoint requires a valid port")
    if parsed.scheme in {"http", "ws"}:
        loopback = hostname == "localhost"
        if not loopback:
            try:
                loopback = ipaddress.ip_address(hostname).is_loopback
            except ValueError:
                pass
        if not loopback:
            raise ValueError(
                "Platform endpoints require TLS except for explicit loopback development URLs"
            )
    return parsed.geturl()


def _default_config_path() -> Path:
    """Per-OS config file location. Used by `load_config()`."""
    plat = platform_info.detect_platform()
    if plat == "windows":
        import os as _os

        base = Path(_os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
        return base / "autogpt-local-executor" / "config.toml"
    if plat == "darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "autogpt-local-executor"
            / "config.toml"
        )
    # linux / wsl2
    import os as _os

    xdg = _os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "autogpt-local-executor" / "config.toml"


class ShimConfig(BaseSettings):
    """All configuration for the local executor shim.

    Wire-visible fields (`allowed_root`, capability flags, `machine_id`) are
    advertised in HELLO. Server-side overrides (`max_concurrent`,
    `command_timeout_seconds`, `max_file_size_bytes`) arrive in HELLO_ACK and
    the daemon updates the live config to honor them.
    """

    model_config = SettingsConfigDict(
        env_prefix="AUTOGPT_SHIM_",
        extra="ignore",
        env_file=None,
        validate_assignment=True,
        hide_input_in_errors=True,
    )

    # ── Platform / OAuth ───────────────────────────────────────────────────
    platform_url: str = Field(
        default="wss://platform.autogpt.net",
        description="Base URL of the AutoGPT platform. WebSocket and OAuth "
        "endpoints are derived from this.",
    )
    platform_ws_url: str | None = Field(
        default=None,
        description="Override for the WebSocket base URL. Defaults to "
        "<platform_url>/ws/local-executor (with ws/wss adjusted).",
    )
    platform_oauth_url: str | None = Field(
        default=None,
        description="Override for the OAuth base URL. Defaults to "
        "<platform_url>/auth (with http/https adjusted).",
    )

    @field_validator("allowed_root", "audit_log_path", "recording_buffer_dir")
    @classmethod
    def expand_home(cls, value: Path | None) -> Path | None:
        """`~` is the user's home, as on the command line, where the shell
        expands it. From the config file or the environment it went through
        whole, so the shim took a folder named `~` in its working directory
        (the home directory, under systemd) for the root, or made one."""
        if value is None:
            return None
        try:
            return value.expanduser()
        except RuntimeError:
            # pathlib's error for `~name` naming no account. Pydantic reports
            # only a ValueError as a bad setting; this one ended `start` in a
            # traceback, which a service manager restarts.
            raise ValueError(f"no home directory found for {value.parts[0]}") from None

    @field_validator("platform_url", "platform_ws_url", "platform_oauth_url")
    @classmethod
    def validate_platform_endpoint(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        schemes = {
            "platform_url": {"https", "wss", "http", "ws"},
            "platform_ws_url": {"wss", "ws"},
            "platform_oauth_url": {"https", "http"},
        }
        return _validate_platform_endpoint(value, schemes[info.field_name or "platform_url"])

    oauth_client_id: str = Field(default="autogpt-local-executor")
    oauth_redirect_port: int = Field(
        default=41899,
        ge=1,
        le=65535,
        description="First port in the 41899-41910 fallback range to bind "
        "for the OAuth callback listener.",
    )

    # ── Session ────────────────────────────────────────────────────────────
    session_id: str | None = Field(
        default=None,
        description="Legacy platform copilot session. Omit for persistent machine control mode.",
    )

    # ── Identity ───────────────────────────────────────────────────────────
    machine_id: str = Field(
        default_factory=lambda: platform_info.detect_machine_id(),
        description="Stable identifier for this machine.",
    )
    display_name: str = Field(
        default_factory=socket.gethostname,
        min_length=1,
        max_length=128,
        description="Human-readable machine label shown in remote folder pickers.",
    )

    # ── File jail / audit ──────────────────────────────────────────────────
    allowed_root: Path = Field(
        default_factory=lambda: platform_info.default_allowed_root(),
        description="All file ops are jailed to this directory.",
    )
    audit_log_path: Path = Field(
        default_factory=lambda: platform_info.default_audit_log_path(),
        description="Append-only command audit log.",
    )

    # ── Capabilities (advertised in HELLO) ─────────────────────────────────
    enable_shell: bool = Field(
        default=False,
        description="Allow unrestricted user-level shell execution. The allowed_root jail "
        "does not constrain shell commands.",
    )
    enable_computer_use: bool = Field(default=False)
    enable_local_llm: bool = Field(
        default=False,
        description="Reserved local-model preview. Startup rejects this option until "
        "end-to-end routing and cancellation are supported.",
    )
    enable_hardware: bool = Field(default=False)

    # ── Limits (also negotiated via HELLO_ACK) ─────────────────────────────
    max_concurrent: int = Field(default=4, gt=0, description="Concurrent in-flight requests.")
    max_concurrent_commands: int = Field(
        default=10,
        gt=0,
        description="Local ceiling for concurrent EXECUTE_COMMAND requests.",
    )
    command_timeout_seconds: int = Field(default=30, gt=0)
    max_file_size_bytes: int = Field(default=10 * 1024 * 1024, gt=0)  # 10 MiB
    max_commands_per_minute: int = Field(default=60, gt=0)

    # ── Reconnect ──────────────────────────────────────────────────────────
    # Positive: a zero or negative delay turns the reconnect backoff into a
    # tight loop against the platform. Finite: asyncio.sleep(inf) never
    # returns, so the shim would stop reconnecting.
    # At most a day each: near the largest float the delay plus its jitter
    # overflowed to infinity, and asyncio.sleep(inf) never returns.
    reconnect_base_delay: float = Field(default=1.0, gt=0, le=86_400, allow_inf_nan=False)
    reconnect_max_delay: float = Field(default=60.0, gt=0, le=86_400, allow_inf_nan=False)
    reconnect_jitter_seconds: float = Field(default=5.0, ge=0, le=86_400, allow_inf_nan=False)

    # ── Local LLM (see docs/LOCAL_LLM.md) ──────────────────────────────────
    ollama_url: str = Field(
        default="http://localhost:11434",
        description="Base URL of the Ollama HTTP API. Override to point at "
        "a remote Ollama on the LAN or a non-default localhost port. "
        "Probed at HELLO time to populate HELLO.local_llm_models.",
    )

    # ── Computer use (see docs/CUA_DRIVER.md) ──────────────────────────────
    cua_driver_path: str | None = Field(
        default=None,
        description="The cua-driver executable to run. By default the one installed "
        "with the computer-use extra, else one on PATH. It must be the cua-driver "
        "release this shim was reviewed against.",
    )
    max_screenshots_per_minute: int = Field(
        default=10,
        gt=0,
        description="The most screenshots computer use sends the platform in a minute; "
        "results past it come back without their images.",
    )
    enable_clipboard: bool = Field(
        default=False,
        description="Offer cua-driver's clipboard_read and clipboard_write. Without "
        "this flag they are not offered. See docs/COMPUTER_USE.md.",
    )
    enable_clipboard_read_foreign: bool = Field(
        default=False,
        description="When set together with --enable-clipboard, allow "
        "clipboard_read to return contents the shim didn't write, except "
        "content a password manager marked private. Without this, "
        "clipboard_read returns only the shim's own writes from the last 30 s. "
        "See docs/COMPUTER_USE.md.",
    )

    # ── Workflow recording (see docs/WORKFLOW_RECORDING.md) ────────────────
    enable_recording: bool = Field(
        default=False,
        description="Design-preview switch. Startup intentionally fails closed "
        "when enabled because capture and interpretation are not production-ready. "
        "The recording capability is never advertised.",
    )
    recording_buffer_dir: Path | None = Field(
        default=None,
        description="Directory for the encrypted recording buffer. Defaults "
        "to a 'recordings' subdir next to the audit log. Evicted buffers are "
        "best-effort secure-erased (§9).",
    )
    recording_default_interpretation_route: str = Field(
        default="extract_then_cloud",
        description="Default interpretation route when the platform doesn't "
        "pin one and the probe is inconclusive. See WORKFLOW_RECORDING.md §3.",
    )
    recording_channels: list[str] = Field(
        default_factory=lambda: ["floor"],
        description="Reserved design-preview capture channels. They are not "
        "advertised until recording is production-ready end to end.",
    )
    recording_retention_limit: int = Field(
        default=10,
        ge=1,
        description="Maximum number of finalized recordings retained for fetch/review.",
    )

    @property
    def derived_recording_buffer_dir(self) -> Path:
        """Resolved recording buffer dir — explicit override, else next to the
        audit log."""
        if self.recording_buffer_dir is not None:
            return self.recording_buffer_dir
        return self.audit_log_path.parent / "recordings"

    @property
    def derived_ws_url(self) -> str:
        if self.platform_ws_url:
            return _validate_platform_endpoint(self.platform_ws_url, {"wss", "ws"})
        return self._platform_base(websocket=True) + "/ws/local-executor"

    @property
    def derived_oauth_url(self) -> str:
        if self.platform_oauth_url:
            return _validate_platform_endpoint(self.platform_oauth_url, {"https", "http"})
        return self._platform_base() + "/auth"

    @property
    def derived_oauth_token_url(self) -> str:
        """Backend OAuth token endpoint for code exchange and refresh."""
        return self._platform_base() + "/api/oauth/token"

    @property
    def derived_oauth_revoke_url(self) -> str:
        """Backend OAuth revocation endpoint."""
        return self._platform_base() + "/api/oauth/revoke"

    def _platform_base(self, *, websocket: bool = False) -> str:
        base = _validate_platform_endpoint(self.platform_url, {"https", "wss", "http", "ws"})
        parsed = urlsplit(base)
        secure = parsed.scheme in {"https", "wss"}
        scheme = ("wss" if secure else "ws") if websocket else ("https" if secure else "http")
        return parsed._replace(scheme=scheme).geturl().rstrip("/")


class ConfigFileError(ValueError):
    """The config file exists but can't be read or isn't valid TOML, or an
    AUTOGPT_SHIM_* environment variable can't be read."""


def _non_directory_ancestor(path: Path) -> Path | None:
    """The nearest of `path`'s ancestors that exists, if it isn't a folder."""
    for parent in path.parents:
        try:
            info = parent.stat()
        except FileNotFoundError:
            continue
        except OSError:
            return None
        return None if stat.S_ISDIR(info.st_mode) else parent
    return None


def load_config(
    config_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> ShimConfig:
    """Layered config load: defaults → TOML file → env (via pydantic-settings)
    → caller overrides (dict, typically from CLI args).
    """
    overrides = dict(overrides or {})
    file_overrides: dict[str, Any] = {}
    path = config_path or _default_config_path()
    # A stat, not is_file(): that raises PermissionError past here up to
    # Python 3.13 and answers False from 3.14, so a config the shim can't
    # reach crashed it, or was skipped for the defaults.
    try:
        is_file = stat.S_ISREG(path.stat().st_mode)
    except FileNotFoundError:
        is_file = False
        if config_path is not None:
            # A --config that names nothing ran the shim on the defaults
            # without a word. (Without --config, a missing default file
            # means defaults.)
            raise ConfigFileError(f"{path}: no such config file") from None
        if path.is_symlink():
            # A config.toml left linking to a file that was moved or deleted:
            # its settings would be dropped for the defaults.
            raise ConfigFileError(f"{path}: a link whose target doesn't exist") from None
        # Windows reports a file where one of the path's folders belongs as a
        # missing path, where POSIX raises NotADirectoryError (below).
        blocker = _non_directory_ancestor(path)
        if blocker is not None:
            raise ConfigFileError(f"{path}: {blocker} is a file, not a folder") from None
    except OSError as exc:
        # NotADirectoryError among them: a file where one of the path's
        # directories belongs, such as the config written to the directory's
        # own name, which the shim would otherwise skip for the defaults.
        raise ConfigFileError(f"{path}: {exc}") from exc
    else:
        if not is_file:
            # A directory where the config belongs, with or without --config.
            raise ConfigFileError(f"{path}: not a regular file")
    if is_file:
        try:
            with open(path, "rb") as f:
                file_overrides = tomllib.load(f)
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            # TOML is UTF-8, and tomllib decodes before it parses.
            raise ConfigFileError(f"{path}: {exc}") from exc
        # Coerce string paths into Path objects so pydantic doesn't choke.
        for key in ("allowed_root", "audit_log_path", "recording_buffer_dir"):
            if key in file_overrides and isinstance(file_overrides[key], str):
                file_overrides[key] = Path(file_overrides[key])
    # ShimConfig(**kwargs) ranks every kwarg above the environment, so the
    # environment is read here and layered over the file explicitly.
    from_env = environment_settings()
    merged = {**file_overrides, **from_env, **overrides}
    return ShimConfig(**merged)


def environment_variable(field: str) -> str:
    """The environment variable that sets `field`."""
    return ShimConfig.model_config["env_prefix"] + field.upper()


def environment_settings() -> dict[str, Any]:
    """The settings the AUTOGPT_SHIM_* variables give, by field, read the way
    pydantic-settings reads them.

    The prefix is passed in: pydantic-settings 2.0.0 doesn't take it from
    the model, and read each field's bare name (PLATFORM_URL, ENABLE_SHELL),
    which then outranked the config file."""
    try:
        return EnvSettingsSource(ShimConfig, env_prefix=ShimConfig.model_config["env_prefix"])()
    except ValueError as exc:
        # pydantic-settings' SettingsError. A list setting takes JSON, so
        # AUTOGPT_SHIM_RECORDING_CHANNELS=floor can't be read. It isn't a
        # ValidationError, so it got past the CLI as a traceback and status 1.
        field = re.search(r'field "([^"]+)"', str(exc))
        name = (
            environment_variable(field.group(1))
            if field
            else "An AUTOGPT_SHIM_ environment variable"
        )
        if isinstance(exc.__cause__, json.JSONDecodeError):
            raise ConfigFileError(f"{name} must be JSON: {exc.__cause__}") from exc
        raise ConfigFileError(f"{name}: {exc}") from exc
