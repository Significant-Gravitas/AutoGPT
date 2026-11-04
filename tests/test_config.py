"""Tests for ShimConfig and load_config layered loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from autogpt_local_executor.config import ShimConfig, load_config


def test_default_values_present() -> None:
    cfg = ShimConfig()
    assert cfg.oauth_client_id == "autogpt-local-executor"
    assert cfg.oauth_redirect_port == 41899
    assert cfg.max_concurrent == 4
    assert cfg.command_timeout_seconds == 30
    assert cfg.max_file_size_bytes == 10 * 1024 * 1024
    assert cfg.enable_shell is False
    assert cfg.recording_retention_limit == 10


def test_env_var_override(monkeypatch) -> None:
    monkeypatch.setenv("AUTOGPT_SHIM_OAUTH_REDIRECT_PORT", "42000")
    cfg = ShimConfig()
    assert cfg.oauth_redirect_port == 42000


def test_load_config_with_toml_overrides(tmp_path: Path) -> None:
    toml_file = tmp_path / "config.toml"
    toml_file.write_text(
        'platform_url = "http://localhost:8000"\n'
        "oauth_redirect_port = 42010\n"
        f'allowed_root = "{tmp_path.as_posix()}/ws"\n'
    )
    cfg = load_config(config_path=toml_file)
    assert cfg.platform_url == "http://localhost:8000"
    assert cfg.oauth_redirect_port == 42010
    assert cfg.allowed_root == tmp_path / "ws"


def test_load_config_overrides_win(tmp_path: Path) -> None:
    toml_file = tmp_path / "config.toml"
    toml_file.write_text("oauth_redirect_port = 42010\n")
    cfg = load_config(
        config_path=toml_file,
        overrides={"oauth_redirect_port": 42050},
    )
    assert cfg.oauth_redirect_port == 42050


def test_the_environment_outranks_the_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defaults, then the TOML file, then AUTOGPT_SHIM_* variables, then the
    caller's overrides: a variable set for one run must beat the file."""
    toml_file = tmp_path / "config.toml"
    toml_file.write_text('platform_url = "http://localhost:8000"\noauth_redirect_port = 42010\n')
    monkeypatch.setenv("AUTOGPT_SHIM_PLATFORM_URL", "http://localhost:9000")
    monkeypatch.setenv("AUTOGPT_SHIM_OAUTH_REDIRECT_PORT", "42020")

    from_env = load_config(config_path=toml_file)
    overridden = load_config(config_path=toml_file, overrides={"oauth_redirect_port": 42050})

    assert from_env.platform_url == "http://localhost:9000"
    assert from_env.oauth_redirect_port == 42020
    assert overridden.oauth_redirect_port == 42050


def test_derived_ws_url_from_https() -> None:
    cfg = ShimConfig(platform_url="https://platform.example.com")
    assert cfg.derived_ws_url == "wss://platform.example.com/ws/local-executor"


def test_derived_ws_url_from_http() -> None:
    cfg = ShimConfig(platform_url="http://localhost:8000")
    assert cfg.derived_ws_url == "ws://localhost:8000/ws/local-executor"


def test_derived_oauth_url_from_wss() -> None:
    cfg = ShimConfig(platform_url="wss://platform.example.com")
    assert cfg.derived_oauth_url == "https://platform.example.com/auth"


def test_derived_oauth_token_url_from_wss() -> None:
    cfg = ShimConfig(platform_url="wss://platform.example.com")
    assert cfg.derived_oauth_token_url == "https://platform.example.com/api/oauth/token"


def test_derived_oauth_revoke_url_from_wss() -> None:
    cfg = ShimConfig(platform_url="wss://platform.example.com")
    assert cfg.derived_oauth_revoke_url == "https://platform.example.com/api/oauth/revoke"


def test_explicit_overrides_take_priority() -> None:
    cfg = ShimConfig(
        platform_url="https://x",
        platform_ws_url="wss://explicit-override/ws",
    )
    assert cfg.derived_ws_url == "wss://explicit-override/ws"


@pytest.mark.parametrize("field", ["platform_url", "platform_ws_url", "platform_oauth_url"])
@pytest.mark.parametrize("host", ["platform.example.com", "192.168.1.20", "localhost.example.com"])
def test_platform_endpoints_reject_cleartext_remote_urls(field: str, host: str) -> None:
    scheme = "ws" if field == "platform_ws_url" else "http"
    with pytest.raises(ValidationError, match="TLS"):
        ShimConfig.model_validate({field: f"{scheme}://{host}:8000"})


@pytest.mark.parametrize("field", ["platform_url", "platform_ws_url", "platform_oauth_url"])
@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "[::1]"])
def test_platform_endpoints_allow_explicit_loopback_urls(field: str, host: str) -> None:
    scheme = "ws" if field == "platform_ws_url" else "http"
    url = f"{scheme}://{host}:8000"
    assert getattr(ShimConfig.model_validate({field: url}), field) == url


@pytest.mark.parametrize(
    "url",
    [
        "https://user:password@platform.example.com",
        "https://platform.example.com?redirect=elsewhere",
        "https://platform.example.com#fragment",
        "https:///no-host",
        "https://platform.example.com:70000",
        "https://platform.example.com\\@localhost",
        "https://platform.example.com\n",
        "file:///tmp/platform",
    ],
)
def test_platform_endpoints_reject_ambiguous_urls(url: str) -> None:
    with pytest.raises(ValidationError):
        ShimConfig(platform_url=url)


@pytest.mark.parametrize(
    ("field", "url"),
    [
        ("platform_ws_url", "https://platform.example.com"),
        ("platform_oauth_url", "wss://platform.example.com"),
    ],
)
def test_platform_overrides_require_matching_transport(field: str, url: str) -> None:
    with pytest.raises(ValidationError):
        ShimConfig.model_validate({field: url})


def test_platform_endpoints_validate_assignment() -> None:
    config = ShimConfig()
    with pytest.raises(ValidationError, match="TLS"):
        config.platform_ws_url = "ws://platform.example.com/ws"


def test_uppercase_secure_scheme_derives_correct_transports() -> None:
    config = ShimConfig(platform_url="HTTPS://platform.example.com")
    assert config.derived_ws_url == "wss://platform.example.com/ws/local-executor"
    assert config.derived_oauth_token_url == "https://platform.example.com/api/oauth/token"


@pytest.mark.parametrize(
    ("field", "property_name", "url"),
    [
        ("platform_url", "derived_ws_url", "http://platform.example.com"),
        ("platform_url", "derived_oauth_token_url", "http://platform.example.com"),
        ("platform_url", "derived_oauth_revoke_url", "http://platform.example.com"),
        ("platform_ws_url", "derived_ws_url", "ws://platform.example.com/ws"),
        ("platform_oauth_url", "derived_oauth_url", "http://platform.example.com/auth"),
    ],
)
def test_platform_endpoints_are_revalidated_at_use(
    field: str, property_name: str, url: str
) -> None:
    config = ShimConfig().model_copy(update={field: url})
    with pytest.raises(ValueError, match="TLS"):
        getattr(config, property_name)


@pytest.mark.parametrize(
    "field",
    [
        "max_concurrent",
        "max_concurrent_commands",
        "command_timeout_seconds",
        "max_file_size_bytes",
        "max_commands_per_minute",
        "max_screenshots_per_minute",
    ],
)
def test_security_limits_must_be_positive(field: str) -> None:
    with pytest.raises(ValidationError):
        ShimConfig.model_validate({field: 0})


@pytest.mark.parametrize("field", ["reconnect_base_delay", "reconnect_max_delay"])
def test_reconnect_delays_must_be_positive(tmp_path: Path, field: str) -> None:
    """A zero or negative delay turns the reconnect backoff into a tight loop."""
    with pytest.raises(ValidationError):
        ShimConfig(allowed_root=tmp_path, **{field: 0})


def test_reconnect_jitter_must_not_be_negative(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        ShimConfig(allowed_root=tmp_path, reconnect_jitter_seconds=-1)


def test_a_malformed_config_file_names_the_file(tmp_path: Path) -> None:
    from autogpt_local_executor.config import ConfigFileError

    config = tmp_path / "config.toml"
    config.write_text("allowed_root = [unclosed\n")

    with pytest.raises(ConfigFileError, match="config.toml"):
        load_config(config_path=config)


def test_the_cli_exits_78_on_a_malformed_config_file(tmp_path: Path, monkeypatch, capsys) -> None:
    import sys

    from autogpt_local_executor import cli

    config = tmp_path / "config.toml"
    config.write_text("allowed_root = [unclosed\n")
    monkeypatch.setattr(sys, "argv", ["autogpt-shim", "--config", str(config), "status"])

    with pytest.raises(SystemExit) as exited:
        cli.main()

    assert exited.value.code == 78
    assert "config.toml" in capsys.readouterr().err
