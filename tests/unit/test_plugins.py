import pytest

from autogpt.config import Config
from autogpt.plugins import (
    denylist_allowlist_check,
    inspect_zip_for_modules,
    scan_plugins,
)

PLUGINS_TEST_DIR = "tests/unit/data/test_plugins"
PLUGIN_TEST_ZIP_FILE = "Auto-GPT-Plugin-Test-master.zip"
PLUGIN_TEST_INIT_PY = "Auto-GPT-Plugin-Test-master/src/auto_gpt_vicuna/__init__.py"
PLUGIN_TEST_OPENAI = "https://weathergpt.vercel.app/"


def test_inspect_zip_for_modules():
    result = inspect_zip_for_modules(str(f"{PLUGINS_TEST_DIR}/{PLUGIN_TEST_ZIP_FILE}"))
    assert result == [PLUGIN_TEST_INIT_PY]


@pytest.fixture
def mock_config_denylist_allowlist_check():
    class MockConfig:
        """Mock config object for testing the denylist_allowlist_check function"""

        plugins_denylist = ["BadPlugin"]
        plugins_allowlist = ["GoodPlugin"]
        authorise_key = "y"
        exit_key = "n"

    return MockConfig()


def test_denylist_allowlist_check_denylist(
    mock_config_denylist_allowlist_check, monkeypatch
):
    # Test that the function returns False when the plugin is in the denylist
    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert not denylist_allowlist_check(
        "BadPlugin", mock_config_denylist_allowlist_check
    )


def test_denylist_allowlist_check_allowlist(
    mock_config_denylist_allowlist_check, monkeypatch
):
    # Test that the function returns True when the plugin is in the allowlist
    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert denylist_allowlist_check("GoodPlugin", mock_config_denylist_allowlist_check)


def test_denylist_allowlist_check_user_input_yes(
    mock_config_denylist_allowlist_check, monkeypatch
):
    # Test that the function returns True when the user inputs "y"
    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert denylist_allowlist_check(
        "UnknownPlugin", mock_config_denylist_allowlist_check
    )


def test_denylist_allowlist_check_user_input_no(
    mock_config_denylist_allowlist_check, monkeypatch
):
    # Test that the function returns False when the user inputs "n"
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert not denylist_allowlist_check(
        "UnknownPlugin", mock_config_denylist_allowlist_check
    )


def test_denylist_allowlist_check_user_input_invalid(
    mock_config_denylist_allowlist_check, monkeypatch
):
    # Test that the function returns False when the user inputs an invalid value
    monkeypatch.setattr("builtins.input", lambda _: "invalid")
    assert not denylist_allowlist_check(
        "UnknownPlugin", mock_config_denylist_allowlist_check
    )


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
