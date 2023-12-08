"""
Test cases for the config class, which handles the configuration settings
for the AI and ensures it behaves as a singleton.
"""
import os
from typing import Any
from unittest import mock
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from autogpt.app.configurator import GPT_3_MODEL, GPT_4_MODEL, apply_overrides_to_config
from autogpt.config import Config, ConfigBuilder


def test_initial_values(config: Config) -> None:
    """
    Test if the initial values of the config class attributes are set correctly.
    """
    assert config.continuous_mode is False
    assert config.tts_config.speak_mode is False
    assert config.fast_llm == "gpt-3.5-turbo-16k"
    assert config.smart_llm.startswith("gpt-4")


def test_set_continuous_mode(config: Config) -> None:
    """
    Test if the set_continuous_mode() method updates the continuous_mode attribute.
    """
    # Store continuous mode to reset it after the test
    continuous_mode = config.continuous_mode

    config.continuous_mode = True
    assert config.continuous_mode is True

    # Reset continuous mode
    config.continuous_mode = continuous_mode


def test_set_speak_mode(config: Config) -> None:
    """
    Test if the set_speak_mode() method updates the speak_mode attribute.
    """
    # Store speak mode to reset it after the test
    speak_mode = config.tts_config.speak_mode

    config.tts_config.speak_mode = True
    assert config.tts_config.speak_mode is True

    # Reset speak mode
    config.tts_config.speak_mode = speak_mode


def test_set_fast_llm(config: Config) -> None:
    """
    Test if the set_fast_llm() method updates the fast_llm attribute.
    """
    # Store model name to reset it after the test
    fast_llm = config.fast_llm

    config.fast_llm = "gpt-3.5-turbo-test"
    assert config.fast_llm == "gpt-3.5-turbo-test"

    # Reset model name
    config.fast_llm = fast_llm


def test_set_smart_llm(config: Config) -> None:
    """
    Test if the set_smart_llm() method updates the smart_llm attribute.
    """
    # Store model name to reset it after the test
    smart_llm = config.smart_llm

    config.smart_llm = "gpt-4-test"
    assert config.smart_llm == "gpt-4-test"

    # Reset model name
    config.smart_llm = smart_llm


@patch("openai.Model.list")
def test_smart_and_fast_llms_set_to_gpt4(mock_list_models: Any, config: Config) -> None:
    """
    Test if models update to gpt-3.5-turbo if gpt-4 is not available.
    """
    fast_llm = config.fast_llm
    smart_llm = config.smart_llm

    config.fast_llm = "gpt-4"
    config.smart_llm = "gpt-4"

    mock_list_models.return_value = {"data": [{"id": "gpt-3.5-turbo"}]}

    apply_overrides_to_config(
        config=config,
        gpt3only=False,
        gpt4only=False,
    )

    assert config.fast_llm == "gpt-3.5-turbo"
    assert config.smart_llm == "gpt-3.5-turbo"

    # Reset config
    config.fast_llm = fast_llm
    config.smart_llm = smart_llm


def test_missing_azure_config(config: Config) -> None:
    assert config.openai_credentials is not None

    config_file = config.app_data_dir / "azure_config.yaml"
    with pytest.raises(FileNotFoundError):
        config.openai_credentials.load_azure_config(config_file)

    config_file.write_text("")
    with pytest.raises(ValueError):
        config.openai_credentials.load_azure_config(config_file)

    assert config.openai_credentials.api_type != "azure"
    assert config.openai_credentials.api_version == ""
    assert config.openai_credentials.azure_model_to_deploy_id_map is None


def test_azure_config(config: Config) -> None:
    config_file = config.app_data_dir / "azure_config.yaml"
    config_file.write_text(
        f"""
azure_api_type: azure
azure_api_base: https://dummy.openai.azure.com
azure_api_version: 2023-06-01-preview
azure_model_map:
    {config.fast_llm}: FAST-LLM_ID
    {config.smart_llm}: SMART-LLM_ID
    {config.embedding_model}: embedding-deployment-id-for-azure
"""
    )

    os.environ["USE_AZURE"] = "True"
    os.environ["AZURE_CONFIG_FILE"] = str(config_file)
    config = ConfigBuilder.build_config_from_env(project_root=config.project_root)

    assert (credentials := config.openai_credentials) is not None
    assert credentials.api_type == "azure"
    assert credentials.api_base == SecretStr("https://dummy.openai.azure.com")
    assert credentials.api_version == "2023-06-01-preview"
    assert credentials.azure_model_to_deploy_id_map == {
        config.fast_llm: "FAST-LLM_ID",
        config.smart_llm: "SMART-LLM_ID",
        config.embedding_model: "embedding-deployment-id-for-azure",
    }

    fast_llm = config.fast_llm
    smart_llm = config.smart_llm
    assert (
        credentials.get_api_access_kwargs(config.fast_llm)["deployment_id"]
        == "FAST-LLM_ID"
    )
    assert (
        credentials.get_api_access_kwargs(config.smart_llm)["deployment_id"]
        == "SMART-LLM_ID"
    )

    # Emulate --gpt4only
    config.fast_llm = smart_llm
    assert (
        credentials.get_api_access_kwargs(config.fast_llm)["deployment_id"]
        == "SMART-LLM_ID"
    )
    assert (
        credentials.get_api_access_kwargs(config.smart_llm)["deployment_id"]
        == "SMART-LLM_ID"
    )

    # Emulate --gpt3only
    config.fast_llm = config.smart_llm = fast_llm
    assert (
        credentials.get_api_access_kwargs(config.fast_llm)["deployment_id"]
        == "FAST-LLM_ID"
    )
    assert (
        credentials.get_api_access_kwargs(config.smart_llm)["deployment_id"]
        == "FAST-LLM_ID"
    )

    del os.environ["USE_AZURE"]
    del os.environ["AZURE_CONFIG_FILE"]


def test_create_config_gpt4only(config: Config) -> None:
    with mock.patch("autogpt.llm.api_manager.ApiManager.get_models") as mock_get_models:
        mock_get_models.return_value = [{"id": GPT_4_MODEL}]
        apply_overrides_to_config(
            config=config,
            gpt4only=True,
        )
        assert config.fast_llm == GPT_4_MODEL
        assert config.smart_llm == GPT_4_MODEL


def test_create_config_gpt3only(config: Config) -> None:
    with mock.patch("autogpt.llm.api_manager.ApiManager.get_models") as mock_get_models:
        mock_get_models.return_value = [{"id": GPT_3_MODEL}]
        apply_overrides_to_config(
            config=config,
            gpt3only=True,
        )
        assert config.fast_llm == GPT_3_MODEL
        assert config.smart_llm == GPT_3_MODEL
