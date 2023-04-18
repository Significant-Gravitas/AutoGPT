import json
import os
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from unittest.mock import MagicMock, patch

import pytest

from autogpt.plugins import create_directory_if_not_exists, inspect_zip_for_module, load_plugins, \
    blacklist_whitelist_check, instantiate_openai_plugin_clients, scan_plugins, fetch_openai_plugins_manifest_and_spec, \
    initialize_openai_plugins


PLUGINS_TEST_DIR = "tests/unit/data/test_plugins"
PLUGIN_TEST_ZIP_FILE = "Auto-GPT-Plugin-Test-master.zip"
PLUGIN_TEST_INIT_PY = "Auto-GPT-Plugin-Test-master/src/auto_gpt_plugin_template/__init__.py"


def test_inspect_zip_for_module():
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "sample.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("test_module/__init__.py", "")

        result = plugins.inspect_zip_for_module(zip_path)
        assert result == "test_module/__init__.py"

        result = plugins.inspect_zip_for_module(zip_path, debug=True)
        assert result == "test_module/__init__.py"

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("not_a_module.py", "")

        result = plugins.inspect_zip_for_module(zip_path)
        assert result is None


def test_write_dict_to_json_file():
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as file:
        test_data = {"test_key": "test_value"}
        plugins.write_dict_to_json_file(test_data, file.name)

        file.seek(0)
        loaded_data = json.load(file)
        assert loaded_data == test_data


def test_create_directory_if_not_exists():
    with tempfile.TemporaryDirectory() as temp_dir:
        new_dir = os.path.join(temp_dir, "test_dir")
        assert not os.path.exists(new_dir)

        result = create_directory_if_not_exists(new_dir)
        assert result is True
        assert os.path.exists(new_dir)

        result = create_directory_if_not_exists(new_dir)
        assert result is True


@pytest.fixture
def config_mock():
    config = MagicMock()
    config.plugins_dir = "/plugins"
    config.plugins_openai = []
    return config


@patch("autogpt.plugins.write_dict_to_json_file")
@patch("requests.get")
@patch("autogpt.plugins.create_directory_if_not_exists")
def test_fetch_openai_plugins_manifest_and_spec(create_directory_mock, write_dict_mock, requests_get_mock, config_mock):
    requests_get_mock.side_effect = [
        MagicMock(status_code=200, json=lambda: {
            "schema_version": "v1",
            "api": {"type": "openapi", "url": "http://example.com/openapi.json"}
        }),
        MagicMock(status_code=404),
    ]

    config_mock.plugins_openai = ["http://example.com"]

    result = fetch_openai_plugins_manifest_and_spec(config_mock)
    assert len(result) == 1
    assert "http://example.com" in result
    assert "manifest" in result["http://example.com"]
    assert "openapi_spec" in result["http://example.com"]

    create_directory_mock.assert_called_once()
    write_dict_mock.assert_called_once()
    requests_get_mock.assert_has_calls([
        patch("requests.get", args=("http://example.com/.well-known/ai-plugin.json",)),
        patch("requests.get", args=("http://example.com/openapi.json",)),
    ])


# @patch("BaseOpenAIPlugin")
# def test_instantiate_openai_plugin_clients(base_openai_plugin_client_mock, config_mock):
#     manifests_specs_clients = {
#         "http://example.com": {
#             "manifest": {},
#             "openapi_spec": {},
#             "client": MagicMock(),
#         }
#     }
#
#     result = instantiate_openai_plugin_clients(manifests_specs_clients, config_mock)
#     assert len(result) == 1


def test_scan_plugins(config_mock):
    with patch("inspect_zip_for_module", return_value="test_module/__init__.py"):
        plugins = scan_plugins(config_mock)
        assert len(plugins) == 0


# @patch("BaseOpenAIPlugin")
# def test_initialize_openai_plugins(base_openai_plugin_client_mock, config_mock):
#     manifests_specs = {
#         "http://example.com": {
#             "manifest": {},
#             "openapi_spec": {},
#         }
#     }
#
#     with patch("Path.cwd") as cwd_mock:
#         cwd_mock.return_value = Path("/fake_cwd")
#         result = initialize_openai_plugins(manifests_specs, config_mock)
#         assert len(result) == 1
#         assert "http://example.com" in result
#         assert "client" in result["http://example.com"]


def test_blacklist_whitelist_check(config_mock):
    class Plugin1(MagicMock):
        __name__ = "Plugin1"

    class Plugin2(MagicMock):
        __name__ = "Plugin2"

    config_mock.plugins_blacklist = ["Plugin1"]
    config_mock.plugins_whitelist = ["Plugin2"]

    plugins = [Plugin1, Plugin2]
    result = blacklist_whitelist_check(plugins, config_mock)
    assert len(result) == 1
    assert isinstance(result[0], Plugin2)

    config_mock.plugins_blacklist = []
    config_mock.plugins_whitelist = []

    with patch("builtins.input", side_effect=["y", "n"]):
        result = blacklist_whitelist_check(plugins, config_mock)
        assert len(result) == 1
        assert isinstance(result[0], Plugin1)


@patch("autogpt.plugins.scan_plugins")
@patch("autogpt.plugins.blacklist_whitelist_check")
def test_load_plugins(blacklist_whitelist_check_mock, scan_plugins_mock, config_mock):
    load_plugins(cfg=config_mock, debug=True)

    scan_plugins_mock.assert_called_once_with(config_mock)
    blacklist_whitelist_check_mock.assert_called_once_with(scan_plugins_mock.return_value, config_mock)

def test_inspect_zip_for_module_no_init_py():
    with patch("zipfile.ZipFile") as zip_mock:
        zip_mock.return_value.__enter__.return_value.namelist.return_value = ["test_module/file1.py"]

        result = inspect_zip_for_module("test_module.zip")
        assert result is None


def test_create_directory_if_not_exists_error():
    with patch("os.makedirs") as makedirs_mock:
        makedirs_mock.side_effect = OSError("Error creating directory")

        result = create_directory_if_not_exists("non_existent_dir")
        assert result is False


def test_fetch_openai_plugins_manifest_and_spec_invalid_manifest():
    with patch("requests.get") as get_mock:
        get_mock.return_value.status_code = 200
        get_mock.return_value.json.return_value = {
            "schema_version": "v2",
            "api": {"type": "openapi"},
        }

        config = MagicMock()
        config.plugins_openai = ["http://example.com"]

        result = fetch_openai_plugins_manifest_and_spec(config)
        assert result == {}


# @patch("BaseOpenAIPlugin")
# def test_instantiate_openai_plugin_clients_invalid_input(base_openai_plugin_client_mock):
#     with pytest.raises(TypeError):
#         instantiate_openai_plugin_clients("invalid_input", MagicMock())


def test_scan_plugins_invalid_config():
    with pytest.raises(AttributeError):
        scan_plugins("invalid_config")


def test_blacklist_whitelist_check_invalid_plugins_input():
    with pytest.raises(TypeError):
        blacklist_whitelist_check("invalid_plugins_input", MagicMock())


def test_blacklist_whitelist_check_invalid_config_input():
    with pytest.raises(TypeError):
        blacklist_whitelist_check([], "invalid_config_input")


def test_load_plugins_invalid_config_input():
    with pytest.raises(TypeError):
        load_plugins("invalid_config_input")
