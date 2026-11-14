"""Tests for ShimConfig and load_config layered loading."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_settings import EnvSettingsSource

import autogpt_local_executor.config as config_module
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
        f'allowed_root = "{tmp_path.as_posix()}/ws"\n',
        encoding="utf-8",
    )
    cfg = load_config(config_path=toml_file)
    assert cfg.platform_url == "http://localhost:8000"
    assert cfg.oauth_redirect_port == 42010
    assert cfg.allowed_root == tmp_path / "ws"


def test_load_config_overrides_win(tmp_path: Path) -> None:
    toml_file = tmp_path / "config.toml"
    toml_file.write_text("oauth_redirect_port = 42010\n", encoding="utf-8")
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
    toml_file.write_text(
        'platform_url = "http://localhost:8000"\noauth_redirect_port = 42010\n', encoding="utf-8"
    )
    monkeypatch.setenv("AUTOGPT_SHIM_PLATFORM_URL", "http://localhost:9000")
    monkeypatch.setenv("AUTOGPT_SHIM_OAUTH_REDIRECT_PORT", "42020")

    from_env = load_config(config_path=toml_file)
    overridden = load_config(config_path=toml_file, overrides={"oauth_redirect_port": 42050})

    assert from_env.platform_url == "http://localhost:9000"
    assert from_env.oauth_redirect_port == 42020
    assert overridden.oauth_redirect_port == 42050


class _PrefixlessEnvSource(EnvSettingsSource):
    """EnvSettingsSource as pydantic-settings 2.0.0 builds it, which the
    dependency range allows: without an env_prefix passed in it reads no
    prefix, where 2.0.1 and later take the model's."""

    def __init__(
        self, settings_cls, case_sensitive=None, env_prefix=None, env_nested_delimiter=None
    ) -> None:
        super().__init__(
            settings_cls,
            case_sensitive=False if case_sensitive is None else case_sensitive,
            env_prefix="" if env_prefix is None else env_prefix,
            env_nested_delimiter=env_nested_delimiter,
        )


@pytest.mark.parametrize(
    "source", [EnvSettingsSource, _PrefixlessEnvSource], ids=["installed", "2.0.0"]
)
def test_only_prefixed_variables_set_the_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: type[EnvSettingsSource]
) -> None:
    """With pydantic-settings 2.0.0 the environment was read without the
    AUTOGPT_SHIM_ prefix: AUTOGPT_SHIM_ALLOWED_ROOT no longer outranked the
    config file, and another program's PLATFORM_URL or ENABLE_SHELL set the
    shim's own."""
    monkeypatch.setattr(config_module, "EnvSettingsSource", source)
    toml_file = tmp_path / "config.toml"
    toml_file.write_text(
        f'allowed_root = "{(tmp_path / "from-file").as_posix()}"\n', encoding="utf-8"
    )
    monkeypatch.setenv("AUTOGPT_SHIM_ALLOWED_ROOT", str(tmp_path / "from-env"))
    monkeypatch.setenv("PLATFORM_URL", "https://elsewhere.example")
    monkeypatch.setenv("ENABLE_SHELL", "true")

    config = load_config(config_path=toml_file)

    assert config.allowed_root == tmp_path / "from-env"
    assert config.platform_url == ShimConfig.model_fields["platform_url"].default
    assert config.enable_shell is False


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
    toml_file.write_text("enable_shell = true\n", encoding="utf-8")
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
    (locked / "config.toml").write_text("max_concurrent = 2\n", encoding="utf-8")
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


def test_a_file_where_the_default_configs_directory_belongs_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A config written to the directory's own name (autogpt-local-executor
    instead of autogpt-local-executor/config.toml) made the default path
    unreachable, and the shim ran on the defaults without a word, its
    platform and allowed_root settings ignored."""
    import autogpt_local_executor.config as config_module
    from autogpt_local_executor.config import ConfigFileError

    (tmp_path / "autogpt-local-executor").write_text(
        'platform_url = "https://example.com"\n', encoding="utf-8"
    )
    monkeypatch.setattr(
        config_module,
        "_default_config_path",
        lambda: tmp_path / "autogpt-local-executor" / "config.toml",
    )

    with pytest.raises(ConfigFileError, match="config.toml"):
        load_config()


def test_a_file_in_the_default_configs_way_is_an_error_where_it_reads_as_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows reports a file where one of the path's folders belongs as a
    missing path (FileNotFoundError), not NotADirectoryError, so there the
    shim still ran on the defaults without a word."""
    import errno

    import autogpt_local_executor.config as config_module
    from autogpt_local_executor.config import ConfigFileError

    (tmp_path / "autogpt-local-executor").write_text(
        'platform_url = "https://example.com"\n', encoding="utf-8"
    )
    blocked = tmp_path / "autogpt-local-executor" / "config.toml"
    monkeypatch.setattr(config_module, "_default_config_path", lambda: blocked)
    real_stat = Path.stat

    def windows_stat(self: Path, *args, **kwargs):
        try:
            return real_stat(self, *args, **kwargs)
        except NotADirectoryError:
            raise FileNotFoundError(
                errno.ENOENT, "The system cannot find the path specified", str(self)
            ) from None

    monkeypatch.setattr(Path, "stat", windows_stat)

    with pytest.raises(ConfigFileError, match="is a file, not a folder"):
        load_config()


@pytest.mark.parametrize("port", [0, -1, 65536])
def test_the_oauth_redirect_port_must_be_a_real_port(tmp_path: Path, port: int) -> None:
    """Port 0 bound an ephemeral port while the redirect URI said 0, so
    sign-in timed out; past 65535, bind raised OverflowError, which the
    callback's port loop doesn't catch."""
    with pytest.raises(ValidationError):
        ShimConfig(allowed_root=tmp_path, oauth_redirect_port=port)


@pytest.mark.parametrize(
    "field", ["reconnect_base_delay", "reconnect_max_delay", "reconnect_jitter_seconds"]
)
def test_reconnect_delays_are_at_most_a_day(tmp_path: Path, field: str) -> None:
    """Any finite float passed. Near the largest, the delay plus its jitter
    overflowed to infinity, and asyncio.sleep(inf) never returns; well short
    of that, a delay of 1e300 seconds never ends either."""
    with pytest.raises(ValidationError):
        ShimConfig(allowed_root=tmp_path, **{field: 86_401})
    assert getattr(ShimConfig(allowed_root=tmp_path, **{field: 86_400}), field) == 86_400


@pytest.mark.parametrize("source", ["file", "environment"])
def test_a_home_relative_path_setting_is_expanded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    """A shell expands `~` on the command line, but from the config file or
    the environment `~/workspace` went through whole: the shim took a folder
    named `~` in its working directory (the home directory, under systemd)
    for the root, and made one."""
    import autogpt_local_executor.config as config_module

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    settings = {
        "allowed_root": "~/workspace",
        "audit_log_path": "~/logs/audit.log",
        "recording_buffer_dir": "~/buffer",
    }
    if source == "file":
        config_file = tmp_path / "config.toml"
        config_file.write_text(
            "".join(f'{key} = "{value}"\n' for key, value in settings.items()), encoding="utf-8"
        )
        config = load_config(config_path=config_file)
    else:
        monkeypatch.setattr(config_module, "_default_config_path", lambda: tmp_path / "absent")
        for key, value in settings.items():
            monkeypatch.setenv(f"AUTOGPT_SHIM_{key.upper()}", value)
        config = load_config()

    home = Path.home()
    assert config.allowed_root == home / "workspace"
    assert config.audit_log_path == home / "logs" / "audit.log"
    assert config.recording_buffer_dir == home / "buffer"


def test_a_home_that_cant_be_found_is_a_configuration_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`~name` for an account the machine doesn't have made pathlib raise
    RuntimeError, which pydantic lets through: `start` ended in a traceback
    and status 1, which launchd and systemd restart, rather than the
    configuration error's message and status."""
    from autogpt_local_executor import cli

    monkeypatch.delenv("AUTOGPT_SHIM_SUPERVISOR", raising=False)
    # Windows guesses another account's home from this one's, unless the
    # profile folder isn't named after the account.
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    monkeypatch.setenv("USERNAME", "someone-else")
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        'allowed_root = "~no-such-account-for-the-shim/workspace"\n', encoding="utf-8"
    )

    with pytest.raises(ValidationError, match="no home directory"):
        load_config(config_path=config_file)
    with pytest.raises(SystemExit) as exited:
        cli._config_or_exit(lambda: load_config(config_path=config_file))

    assert exited.value.code == 78
    assert "Configuration error (allowed_root)" in capsys.readouterr().err


def test_a_default_config_link_whose_target_is_gone_is_an_error(
    tmp_path: Path, monkeypatch
) -> None:
    """A config.toml left linking to a file that was moved or deleted read as
    a missing config, so the shim ran on the defaults, without the platform,
    paths and capabilities it set."""
    import autogpt_local_executor.config as config_module
    from autogpt_local_executor.config import ConfigFileError

    link = tmp_path / "config.toml"
    try:
        link.symlink_to(tmp_path / "moved.toml")
    except OSError:
        pytest.skip("this system can't make a symbolic link")
    monkeypatch.setattr(config_module, "_default_config_path", lambda: link)

    with pytest.raises(ConfigFileError, match="target doesn't exist"):
        load_config()
