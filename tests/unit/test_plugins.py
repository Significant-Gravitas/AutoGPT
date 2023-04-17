import pytest
from pathlib import Path
from zipfile import ZipFile
from autogpt.plugins import inspect_zip_for_module, scan_plugins, load_plugins
from autogpt.config import Config

PLUGINS_TEST_DIR = "tests/unit/data/test_plugins/"
PLUGIN_TEST_ZIP_FILE = "Auto-GPT-Plugin-Test-master.zip"
PLUGIN_TEST_INIT_PY = "Auto-GPT-Plugin-Test-master/src/auto_gpt_plugin_template/__init__.py"


@pytest.fixture
def config_with_plugins():
    cfg = Config()
    cfg.plugins_dir = PLUGINS_TEST_DIR
    return cfg


def test_inspect_zip_for_module():
    result = inspect_zip_for_module(str(PLUGINS_TEST_DIR + PLUGIN_TEST_ZIP_FILE))
    assert result == PLUGIN_TEST_INIT_PY

def test_scan_plugins():
    result = scan_plugins(PLUGINS_TEST_DIR, debug=True)
    assert len(result) == 1
    assert result[0][0] == PLUGIN_TEST_INIT_PY


def test_load_plugins_blacklisted(config_with_plugins):
    config_with_plugins.plugins_blacklist = ['AbstractSingleton']
    result = load_plugins(cfg=config_with_plugins)
    assert len(result) == 0
