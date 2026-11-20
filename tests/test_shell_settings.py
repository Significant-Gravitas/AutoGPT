"""A test doesn't see the settings exported in the shell that runs it.

ShimConfig reads every AUTOGPT_SHIM_* variable that names one of its
settings. A person who exports one to run their own shim
(AUTOGPT_SHIM_ENABLE_HARDWARE=true, AUTOGPT_SHIM_MAX_CONCURRENT=2) then had
tests fail that pass everywhere else: one that expects a default, or that a
setting is off, built its config and got theirs. Each test starts without
those variables now (tests/conftest.py), and one that means a config to
read a variable sets it.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from autogpt_local_executor.config import HOSTED_PLATFORM_URL, ShimConfig, environment_variable

# Every setting's variable, with a value none of them takes for a default
# and most can't read at all.
_EXPORTED = {environment_variable(field): "left-by-a-shell" for field in ShimConfig.model_fields}


@pytest.fixture(scope="module", autouse=True)
def _a_shell_that_exports_every_setting() -> Iterator[None]:
    """In the environment before any test here starts, as a shell's are
    before pytest does: a module's fixture is set up ahead of each test's."""
    before = {name: os.environ.get(name) for name in _EXPORTED}
    os.environ.update(_EXPORTED)
    try:
        yield
    finally:
        for name, value in before.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _config(tmp_path: Path) -> ShimConfig:
    return ShimConfig(
        allowed_root=tmp_path, audit_log_path=tmp_path / "audit.log", machine_id="test-machine"
    )


def test_no_settings_variable_reaches_a_test() -> None:
    assert len(_EXPORTED) == len(ShimConfig.model_fields) > 10
    assert [name for name in _EXPORTED if name in os.environ] == []


def test_a_config_a_test_builds_has_the_defaults(tmp_path: Path) -> None:
    config = _config(tmp_path)

    assert config.enable_hardware is False
    assert config.enable_shell is False
    assert config.max_concurrent == ShimConfig.model_fields["max_concurrent"].default
    assert config.platform_url == HOSTED_PLATFORM_URL


def test_a_test_that_sets_a_variable_has_it_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(environment_variable("max_concurrent"), "3")

    assert _config(tmp_path).max_concurrent == 3
