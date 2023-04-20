from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
import yaml

from autogpt.commands.openapi import load_openapi_config, OPENAPI_COMMANDS_REGISTRY, OpenAPIConfig

PLUGINS_TEST_DIR = "tests/unit/data/test_plugins"
OPENAPI_COMMANDS_REGISTRY = OPENAPI_COMMANDS_REGISTRY
TEST_OPENAPIS_APIS = ['weathergpt']


@pytest.fixture
def mock_config():
    class MockConfig:
        plugins_dir = PLUGINS_TEST_DIR
        openapi_apis = TEST_OPENAPIS_APIS

    return MockConfig()


def test_load_openapi_config(mock_config):
    # Mock the OPENAPI_COMMANDS_REGISTRY file
    with NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".yaml", delete=False) as temp_file:
        temp_file_path = Path(temp_file.name)
        example_data = {
            "weathergpt": {
                "url": "https://example.com",
                "openapi_spec": "spec.yaml",
                "openai_manifest": "manifest.yaml",
                "enabled": True,
                "auth": "Bearer token"
            },
            "disabled_api": {
                "url": "https://disabled.com",
                "openapi_spec": "spec.yaml",
                "openai_manifest": "manifest.yaml",
                "enabled": False,
                "auth": "Bearer token"
            }
        }
        yaml.dump(example_data, temp_file)
        temp_file.flush()

        # Test the load_openapi_config function
        result = load_openapi_config(mock_config, openapi_registry_path=str(temp_file_path))

        temp_file_path.unlink()

        assert "weathergpt" in result
        assert isinstance(result["weathergpt"], OpenAPIConfig)
        assert result["weathergpt"].url == "https://example.com"
        assert "disabled_api" not in result

def test_load_openapi_config_missing_file(mock_config):
    # Test the load_openapi_config function with a non-existent file
    result = load_openapi_config(mock_config, openapi_registry_path="non_existent_file.yaml")

    assert result == {}