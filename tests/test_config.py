"""Tests for ShimConfig and load_config layered loading."""

from __future__ import annotations

import os
import sys
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


# TOML that doesn't parse, and a file that isn't UTF-8, which TOML must be.
# The second escaped load_config as a UnicodeDecodeError, so the shim died
# with a traceback and status 1, which systemd restarts every 5 seconds.
_MALFORMED_CONFIGS = pytest.mark.parametrize(
    "content",
    [b"allowed_root = [unclosed\n", b'allowed_root = "/home/\xff\xfe"\n'],
    ids=["bad-toml", "not-utf-8"],
)


@_MALFORMED_CONFIGS
def test_a_malformed_config_file_names_the_file(tmp_path: Path, content: bytes) -> None:
    from autogpt_local_executor.config import ConfigFileError

    config = tmp_path / "config.toml"
    config.write_bytes(content)

    with pytest.raises(ConfigFileError, match="config.toml"):
        load_config(config_path=config)


@_MALFORMED_CONFIGS
def test_the_cli_exits_78_on_a_malformed_config_file(
    tmp_path: Path, monkeypatch, capsys, content: bytes
) -> None:
    import sys

    from autogpt_local_executor import cli

    config = tmp_path / "config.toml"
    config.write_bytes(content)
    monkeypatch.setattr(sys, "argv", ["autogpt-shim", "--config", str(config), "status"])

    with pytest.raises(SystemExit) as exited:
        cli.main()

    assert exited.value.code == 78
    assert "config.toml" in capsys.readouterr().err


def test_a_boolean_variable_outranks_the_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Strings and integers were covered; a false from the environment must
    also beat a true in the file."""
    toml_file = tmp_path / "config.toml"
    toml_file.write_text("enable_shell = true\n")
    monkeypatch.setenv("AUTOGPT_SHIM_ENABLE_SHELL", "false")

    assert load_config(config_path=toml_file).enable_shell is False


@pytest.mark.parametrize(
    "field", ["reconnect_base_delay", "reconnect_max_delay", "reconnect_jitter_seconds"]
)
@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_reconnect_delays_must_be_finite(tmp_path: Path, field: str, value: float) -> None:
    """An infinite delay passes gt=0, and asyncio.sleep(inf) never returns."""
    with pytest.raises(ValidationError):
        ShimConfig(allowed_root=tmp_path, **{field: value})


def test_a_config_file_named_with_config_that_does_not_exist_is_an_error(tmp_path: Path) -> None:
    """A typo in --config ran the shim on the defaults without a word."""
    from autogpt_local_executor.config import ConfigFileError

    with pytest.raises(ConfigFileError, match="no such config file"):
        load_config(config_path=tmp_path / "missing.toml")


def test_without_config_a_missing_default_file_means_the_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import autogpt_local_executor.config as config_module

    monkeypatch.setattr(config_module, "_default_config_path", lambda: tmp_path / "absent.toml")

    assert isinstance(load_config(), ShimConfig)


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX permissions, not root"
)
def test_a_default_config_that_cannot_be_reached_is_an_error(tmp_path: Path, monkeypatch) -> None:
    """Path.is_file() raised PermissionError past load_config (Python 3.13
    and earlier) or answered False (3.14), so the shim crashed, or started on
    the defaults while ignoring the user's config."""
    from autogpt_local_executor import config as config_module
    from autogpt_local_executor.config import ConfigFileError

    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "config.toml").write_text("max_concurrent = 2\n")
    monkeypatch.setattr(config_module, "_default_config_path", lambda: locked / "config.toml")
    locked.chmod(0)
    try:
        with pytest.raises(ConfigFileError, match="config.toml"):
            load_config()
    finally:
        locked.chmod(0o700)


def test_an_environment_list_that_is_not_json_is_a_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A list setting takes JSON from the environment. Anything else raised
    pydantic-settings' SettingsError, which isn't a ValidationError, so it
    got past the CLI's config handling."""
    import autogpt_local_executor.config as config_module
    from autogpt_local_executor.config import ConfigFileError

    monkeypatch.setattr(config_module, "_default_config_path", lambda: tmp_path / "absent.toml")
    monkeypatch.setenv("AUTOGPT_SHIM_RECORDING_CHANNELS", "floor")

    with pytest.raises(ConfigFileError, match="AUTOGPT_SHIM_RECORDING_CHANNELS"):
        load_config()


def test_the_cli_exits_78_on_an_environment_list_that_is_not_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """It ended in a traceback and status 1, which a service manager
    restarts, instead of the config error's 78."""
    import autogpt_local_executor.config as config_module
    from autogpt_local_executor import cli

    monkeypatch.setattr(config_module, "_default_config_path", lambda: tmp_path / "absent.toml")
    monkeypatch.setenv("AUTOGPT_SHIM_RECORDING_CHANNELS", "floor")
    monkeypatch.setattr(sys, "argv", ["autogpt-shim", "status"])

    with pytest.raises(SystemExit) as exited:
        cli.main()

    assert exited.value.code == 78
    assert "AUTOGPT_SHIM_RECORDING_CHANNELS" in capsys.readouterr().err


def test_a_default_config_path_that_is_not_a_file_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory where the default config.toml belongs ran the shim on the
    defaults without a word, though --config naming one is an error."""
    import autogpt_local_executor.config as config_module
    from autogpt_local_executor.config import ConfigFileError

    (tmp_path / "config.toml").mkdir()
    monkeypatch.setattr(config_module, "_default_config_path", lambda: tmp_path / "config.toml")

    with pytest.raises(ConfigFileError, match="not a regular file"):
        load_config()
