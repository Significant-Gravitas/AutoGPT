import os

import yaml

from autogpt.config.config import Config
from autogpt.plugins import inspect_zip_for_modules, scan_plugins
from autogpt.plugins.plugin_config import PluginConfig
from autogpt.plugins.plugins_config import PluginsConfig

PLUGINS_TEST_DIR = "tests/unit/data/test_plugins"
PLUGIN_TEST_ZIP_FILE = "Auto-GPT-Plugin-Test-master.zip"
PLUGIN_TEST_INIT_PY = "Auto-GPT-Plugin-Test-master/src/auto_gpt_vicuna/__init__.py"
PLUGIN_TEST_OPENAI = "https://weathergpt.vercel.app/"


def test_scan_plugins_openai(config: Config):
    config.plugins_openai = [PLUGIN_TEST_OPENAI]
    plugins_config = config.plugins_config
    plugins_config.plugins[PLUGIN_TEST_OPENAI] = PluginConfig(
        name=PLUGIN_TEST_OPENAI, enabled=True
    )

    # Test that the function returns the correct number of plugins
    result = scan_plugins(config, debug=True)
    assert len(result) == 1


def test_scan_plugins_generic(config: Config):
    # Test that the function returns the correct number of plugins
    plugins_config = config.plugins_config
    plugins_config.plugins["auto_gpt_guanaco"] = PluginConfig(
        name="auto_gpt_guanaco", enabled=True
    )
    plugins_config.plugins["AutoGPTPVicuna"] = PluginConfig(
        name="AutoGPTPVicuna", enabled=True
    )
    result = scan_plugins(config, debug=True)
    plugin_class_names = [plugin.__class__.__name__ for plugin in result]

    assert len(result) == 2
    assert "AutoGPTGuanaco" in plugin_class_names
    assert "AutoGPTPVicuna" in plugin_class_names


def test_scan_plugins_not_enabled(config: Config):
    # Test that the function returns the correct number of plugins
    plugins_config = config.plugins_config
    plugins_config.plugins["auto_gpt_guanaco"] = PluginConfig(
        name="auto_gpt_guanaco", enabled=True
    )
    plugins_config.plugins["auto_gpt_vicuna"] = PluginConfig(
        name="auto_gptp_vicuna", enabled=False
    )
    result = scan_plugins(config, debug=True)
    plugin_class_names = [plugin.__class__.__name__ for plugin in result]

    assert len(result) == 1
    assert "AutoGPTGuanaco" in plugin_class_names
    assert "AutoGPTPVicuna" not in plugin_class_names


def test_inspect_zip_for_modules():
    result = inspect_zip_for_modules(str(f"{PLUGINS_TEST_DIR}/{PLUGIN_TEST_ZIP_FILE}"))
    assert result == [PLUGIN_TEST_INIT_PY]


def test_create_base_config(config: Config):
    """Test the backwards-compatibility shim to convert old plugin allow/deny list to a config file"""
    config.plugins_allowlist = ["a", "b"]
    config.plugins_denylist = ["c", "d"]

    os.remove(config.plugins_config_file)
    plugins_config = PluginsConfig.load_config(
        plugins_config_file=config.workdir / config.plugins_config_file,
        plugins_denylist=config.plugins_denylist,
        plugins_allowlist=config.plugins_allowlist,
    )

    # Check the structure of the plugins config data
    assert len(plugins_config.plugins) == 4
    assert plugins_config.get("a").enabled
    assert plugins_config.get("b").enabled
    assert not plugins_config.get("c").enabled
    assert not plugins_config.get("d").enabled

    # Check the saved config file
    with open(config.plugins_config_file, "r") as saved_config_file:
        saved_config = yaml.load(saved_config_file, Loader=yaml.FullLoader)

    assert saved_config == {
        "a": {"enabled": True, "config": {}},
        "b": {"enabled": True, "config": {}},
        "c": {"enabled": False, "config": {}},
        "d": {"enabled": False, "config": {}},
    }


def test_load_config(config: Config):
    """Test that the plugin config is loaded correctly from the plugins_config.yaml file"""
    # Create a test config and write it to disk
    test_config = {
        "a": {"enabled": True, "config": {"api_key": "1234"}},
        "b": {"enabled": False, "config": {}},
    }
    with open(config.plugins_config_file, "w+") as f:
        f.write(yaml.dump(test_config))

    # Load the config from disk
    plugins_config = PluginsConfig.load_config(
        plugins_config_file=config.workdir / config.plugins_config_file,
        plugins_denylist=config.plugins_denylist,
        plugins_allowlist=config.plugins_allowlist,
    )

    # Check that the loaded config is equal to the test config
    assert len(plugins_config.plugins) == 2
    assert plugins_config.get("a").enabled
    assert plugins_config.get("a").config == {"api_key": "1234"}
    assert not plugins_config.get("b").enabled
    assert plugins_config.get("b").config == {}
