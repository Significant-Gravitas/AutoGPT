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
import socket
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
    reconnect_base_delay: float = Field(default=1.0)
    reconnect_max_delay: float = Field(default=60.0)
    reconnect_jitter_seconds: float = Field(default=5.0)

    # ── Local LLM (see docs/LOCAL_LLM.md) ──────────────────────────────────
    ollama_url: str = Field(
        default="http://localhost:11434",
        description="Base URL of the Ollama HTTP API. Override to point at "
        "a remote Ollama on the LAN or a non-default localhost port. "
        "Probed at HELLO time to populate HELLO.local_llm_models.",
    )

    # ── Computer use ───────────────────────────────────────────────────────
    max_screenshots_per_minute: int = Field(default=10, gt=0)
    enable_clipboard: bool = Field(
        default=False,
        description="Allow CLIPBOARD_READ / CLIPBOARD_WRITE. Without this "
        "flag clipboard sub-ops return FEATURE_NOT_SUPPORTED. See "
        "docs/COMPUTER_USE.md Q3.",
    )
    enable_clipboard_read_foreign: bool = Field(
        default=False,
        description="When set together with --enable-clipboard, allow "
        "CLIPBOARD_READ to return foreign (non-shim-written) contents, "
        "subject to ConcealedType / CF_PRIVATE checks. Without this, "
        "CLIPBOARD_READ is writeback-only with a 30s window. See "
        "docs/COMPUTER_USE.md Q3.",
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
    if path.is_file():
        with open(path, "rb") as f:
            file_overrides = tomllib.load(f)
        # Coerce string paths into Path objects so pydantic doesn't choke.
        for key in ("allowed_root", "audit_log_path", "recording_buffer_dir"):
            if key in file_overrides and isinstance(file_overrides[key], str):
                file_overrides[key] = Path(file_overrides[key])
    # ShimConfig(**kwargs) ranks every kwarg above the environment, so the
    # environment is read here and layered over the file explicitly.
    from_env = EnvSettingsSource(ShimConfig)()
    merged = {**file_overrides, **from_env, **overrides}
    return ShimConfig(**merged)
