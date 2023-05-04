from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
import yaml

from autogpt.commands.openapi import (
    OPENAPI_COMMANDS_REGISTRY,
    AutoGPTOpenAPIConfig,
    get_openapi_specs_and_manifests,
    initialize_openapi_plugins,
    load_openapi_config,
)

PLUGINS_TEST_DIR = "tests/unit/data/test_plugins"
OPENAPI_COMMANDS_REGISTRY = OPENAPI_COMMANDS_REGISTRY
TEST_OPENAPIS_APIS = ["weathergpt"]


@pytest.fixture
def mock_config():
    class MockConfig:
        plugins_dir = PLUGINS_TEST_DIR
        openapi_apis = TEST_OPENAPIS_APIS

    return MockConfig()


def test_load_openapi_config(mock_config):
    # Mock the OPENAPI_COMMANDS_REGISTRY file
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".yaml", delete=False
    ) as temp_file:
        temp_file_path = Path(temp_file.name)
        example_data = {
            "weathergpt": {
                "url": "https://example.com",
                "openapi_spec": "spec.yaml",
                "openai_manifest": "manifest.yaml",
                "enabled": True,
                "auth": {"type": None},
            },
            "disabled_api": {
                "url": "https://disabled.com",
                "openapi_spec": "spec.yaml",
                "openai_manifest": "manifest.yaml",
                "enabled": False,
                "auth": {"type": None},
            },
        }
        yaml.dump(example_data, temp_file)
        temp_file.flush()

        # Test the load_openapi_config function
        result = load_openapi_config(
            mock_config, openapi_registry_path=str(temp_file_path)
        )

        temp_file_path.unlink()

        assert "weathergpt" in result
        assert isinstance(result["weathergpt"], AutoGPTOpenAPIConfig)
        assert result["weathergpt"].url == "https://example.com"
        assert "disabled_api" not in result


def test_load_openapi_config_missing_file(mock_config):
    # Test the load_openapi_config function with a non-existent file
    result = load_openapi_config(
        mock_config, openapi_registry_path="non_existent_file.yaml"
    )
    assert result == {}


@pytest.fixture
def weathergpt_openai_config():
    return {
        "weathergpt": AutoGPTOpenAPIConfig(
            url="https://weathergpt.vercel.app",
            openai_manifest="/.well-known/ai-plugin.json",
            enabled=True,
            auth={"type": None},
        )
    }


@pytest.fixture
def weatherapi_config():
    return {
        "weathergpt": AutoGPTOpenAPIConfig(
            url="https://weathergpt.vercel.app",
            openapi_spec="/openapi.json",
            enabled=True,
            auth={"type": None},
        )
    }


def test_get_openapi_manifests_and_specs_from_manifest(weathergpt_openai_config):
    specs_manifests = get_openapi_specs_and_manifests(
        weathergpt_openai_config, plugins_dir=PLUGINS_TEST_DIR
    )
    assert specs_manifests["weathergpt"]["manifest"]
    assert specs_manifests["weathergpt"]["openapi_spec"]


def test_get_openapi_manifests_and_specs_from_openapi_spec(weatherapi_config):
    specs_manifests = get_openapi_specs_and_manifests(
        weatherapi_config, plugins_dir=PLUGINS_TEST_DIR
    )
    assert specs_manifests["weathergpt"]["manifest"]
    assert specs_manifests["weathergpt"]["openapi_spec"]


def test_initialize_openapi_plugins(weathergpt_openai_config):
    specs_manifests = get_openapi_specs_and_manifests(
        weathergpt_openai_config, plugins_dir=PLUGINS_TEST_DIR
    )
    initalized_openapi_plugins = initialize_openapi_plugins(
        specs_manifests, weathergpt_openai_config, PLUGINS_TEST_DIR
    )

    assert initalized_openapi_plugins["weathergpt"]
    assert "modules" in initalized_openapi_plugins["weathergpt"].keys()
