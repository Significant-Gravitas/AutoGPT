import pytest

from autogpt.config import Config
from autogpt.plugins import scan_plugins

PLUGINS_TEST_DIR = "tests/unit/data/test_plugins"
PLUGIN_TEST_OPENAI = "https://weathergpt.vercel.app/"


@pytest.fixture
def mock_config_denylist_allowlist_check():
    class MockConfig:
        """Mock config object for testing the denylist_allowlist_check function"""

        plugins_denylist = ["BadPlugin"]
        plugins_allowlist = ["GoodPlugin"]
        authorise_key = "y"
        exit_key = "n"

    return MockConfig()


@pytest.fixture
def config_with_plugins():
    """Mock config object for testing the scan_plugins function"""
    # Test that the function returns the correct number of plugins
    cfg = Config()
    cfg.plugins_dir = PLUGINS_TEST_DIR
    cfg.plugins_openai = ["https://weathergpt.vercel.app/"]
    return cfg


@pytest.fixture
def mock_config_openai_plugin():
    """Mock config object for testing the scan_plugins function"""

    class MockConfig:
        """Mock config object for testing the scan_plugins function"""

        plugins_dir = PLUGINS_TEST_DIR
        plugins_openai = [PLUGIN_TEST_OPENAI]
        plugins_denylist = ["AutoGPTPVicuna"]
        plugins_allowlist = [PLUGIN_TEST_OPENAI]

    return MockConfig()


def test_scan_plugins_openai(mock_config_openai_plugin):
    # Test that the function returns the correct number of plugins
    result = scan_plugins(mock_config_openai_plugin, debug=True)
    assert len(result) == 1


@pytest.fixture
def mock_config_generic_plugin():
    """Mock config object for testing the scan_plugins function"""

    # Test that the function returns the correct number of plugins
    class MockConfig:
        plugins_dir = PLUGINS_TEST_DIR
        plugins_openai = []
        plugins_denylist = []
        plugins_allowlist = ["AutoGPTPVicuna"]

    return MockConfig()


def test_scan_plugins_generic(mock_config_generic_plugin):
    # Test that the function returns the correct number of plugins
    result = scan_plugins(mock_config_generic_plugin, debug=True)
    assert len(result) == 1
