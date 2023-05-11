"""
Test cases for the plugins file that includes utility functions for loading plugins.
"""

import pytest

from autogpt.config.config import Config
from autogpt.plugins import denylist_allowlist_check


@pytest.fixture(scope="function")
def cfg_continuous_mode_true():
    config = Config()

    old_value = Config.continuous_mode if hasattr(Config, "continuous_mode") else False
    config.set_continuous_mode(True)
    yield config
    config.set_continuous_mode(old_value)


def test_denylist_allowlist_check_continuous_mode_no_prompt(cfg_continuous_mode_true):
    """
    Test if the denylist_allowlist_check() function skips user prompting when the continuous_mode attribute is True.

    Args:
        cfg_continuous_mode_true (pytest.fixture): returns a Config object with continuous_mode set to True
    """
    plugin_name = "test_plugin"
    cfg = Config()
    result = denylist_allowlist_check(plugin_name, cfg_continuous_mode_true)

    assert result is False
