"""
Test cases for the plugins file that includes utility functions for loading plugins.
"""

import pytest

from autogpt.config.config import Config
from autogpt.plugins import denylist_allowlist_check


@pytest.fixture
def cfg_continuous_mode_true():
    config = Config()
    config.continuous_mode = True
    return config


def test_denylist_allowlist_check_continuous_mode_no_prompt(cfg_continuous_mode_true):
    """
    Test if the denylist_allowlist_check() function skips user prompting when the continuous_mode attribute is True.

    Args:
        cfg_continuous_mode_true (pytest.fixture): returns a Config object with continuous_mode set to True
    """
    plugin_name = "test_plugin"
    result = denylist_allowlist_check(plugin_name, cfg_continuous_mode_true)

    assert result is False
