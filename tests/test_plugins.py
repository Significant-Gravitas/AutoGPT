"""
Test cases for the Config class, which handles the configuration settings
for the AI and ensures it behaves as a singleton.
"""

import pytest

from autogpt.config.config import Config
from autogpt.plugins import denylist_allowlist_check


@pytest.fixture
def cfg_continuous_mode_true():
    config = Config()
    config.continuous_mode = True
    return config


def test_denylist_allowlist_check_continuous_mode_true(cfg_continuous_mode_true):
    plugin_name = "test_plugin"
    result = denylist_allowlist_check(plugin_name, cfg_continuous_mode_true)

    assert result is False
