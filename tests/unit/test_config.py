"""
Test cases for the config class, which handles the configuration settings
for the AI and ensures it behaves as a singleton.
"""
import os
from unittest import mock
from unittest.mock import patch

import pytest

from autogpt.app.configurator import GPT_3_MODEL, GPT_4_MODEL, create_config
from autogpt.config import Config, ConfigBuilder
from autogpt.workspace.workspace import Workspace


def test_initial_values(config: Config):
    """
    Test if the initial values of the config class attributes are set correctly.
    """
    assert config.debug_mode == False
    assert config.continuous_mode == False
    assert config.speak_mode == False
    assert config.fast_llm == "gpt-3.5-turbo"
    assert config.smart_llm == "gpt-4"


def test_set_continuous_mode(config: Config):
    """
    Test if the set_continuous_mode() method updates the continuous_mode attribute.
    """
    # Store continuous mode to reset it after the test
    continuous_mode = config.continuous_mode

    config.continuous_mode = True
    assert config.continuous_mode == True

    # Reset continuous mode
    config.continuous_mode = continuous_mode


def test_set_speak_mode(config: Config):
    """
    Test if the set_speak_mode() method updates the speak_mode attribute.
    """
    # Store speak mode to reset it after the test
    speak_mode = config.speak_mode

    config.speak_mode = True
    assert config.speak_mode == True

    # Reset speak mode
    config.speak_mode = speak_mode


def test_set_fast_llm(config: Config):
    """
    Test if the set_fast_llm() method updates the fast_llm attribute.
    """
    # Store model name to reset it after the test
    fast_llm = config.fast_llm

    config.fast_llm = "gpt-3.5-turbo-test"
    assert config.fast_llm == "gpt-3.5-turbo-test"

    # Reset model name
    config.fast_llm = fast_llm


def test_set_smart_llm(config: Config):
    """
    Test if the set_smart_llm() method updates the smart_llm attribute.
    """
    # Store model name to reset it after the test
    smart_llm = config.smart_llm

    config.smart_llm = "gpt-4-test"
    assert config.smart_llm == "gpt-4-test"

    # Reset model name
    config.smart_llm = smart_llm


def test_set_debug_mode(config: Config):
    """
    Test if the set_debug_mode() method updates the debug_mode attribute.
    """
    # Store debug mode to reset it after the test
    debug_mode = config.debug_mode

    config.debug_mode = True
    assert config.debug_mode == True

    # Reset debug mode
    config.debug_mode = debug_mode


@patch("openai.Model.list")
def test_smart_and_fast_llms_set_to_gpt4(mock_list_models, config: Config):
    """
    Test if models update to gpt-3.5-turbo if both are set to gpt-4.
    """
    fast_llm = config.fast_llm
    smart_llm = config.smart_llm

    config.fast_llm = "gpt-4"
    config.smart_llm = "gpt-4"

    mock_list_models.return_value = {"data": [{"id": "gpt-3.5-turbo"}]}

    create_config(
        config=config,
        continuous=False,
        continuous_limit=False,
        ai_settings_file="",
        prompt_settings_file="",
        skip_reprompt=False,
        speak=False,
        debug=False,
        gpt3only=False,
        gpt4only=False,
        memory_type="",
        browser_name="",
        allow_downloads=False,
        skip_news=False,
    )

    assert config.fast_llm == "gpt-3.5-turbo"
    assert config.smart_llm == "gpt-3.5-turbo"

    # Reset config
    config.fast_llm = fast_llm
    config.smart_llm = smart_llm


def test_missing_azure_config(workspace: Workspace):
    config_file = workspace.get_path("azure_config.yaml")
    with pytest.raises(FileNotFoundError):
        ConfigBuilder.load_azure_config(str(config_file))

    config_file.write_text("")
    azure_config = ConfigBuilder.load_azure_config(str(config_file))

    assert azure_config["openai_api_type"] == "azure"
    assert azure_config["openai_api_base"] == ""
    assert azure_config["openai_api_version"] == "2023-03-15-preview"
    assert azure_config["azure_model_to_deployment_id_map"] == {}


def test_azure_config(config: Config, workspace: Workspace) -> None:
    config_file = workspace.get_path("azure_config.yaml")
    yaml_content = f"""
azure_api_type: azure
azure_api_base: https://dummy.openai.azure.com
azure_api_version: 2023-06-01-preview
azure_model_map:
    fast_llm_deployment_id: FAST-LLM_ID
    smart_llm_deployment_id: SMART-LLM_ID
    embedding_model_deployment_id: embedding-deployment-id-for-azure
"""
    config_file.write_text(yaml_content)

    os.environ["USE_AZURE"] = "True"
    os.environ["AZURE_CONFIG_FILE"] = str(config_file)
    config = ConfigBuilder.build_config_from_env(workspace.root.parent)

    assert config.openai_api_type == "azure"
    assert config.openai_api_base == "https://dummy.openai.azure.com"
    assert config.openai_api_version == "2023-06-01-preview"
    assert config.azure_model_to_deployment_id_map == {
        "fast_llm_deployment_id": "FAST-LLM_ID",
        "smart_llm_deployment_id": "SMART-LLM_ID",
        "embedding_model_deployment_id": "embedding-deployment-id-for-azure",
    }

    fast_llm = config.fast_llm
    smart_llm = config.smart_llm
    assert (
        config.get_azure_credentials(config.fast_llm)["deployment_id"] == "FAST-LLM_ID"
    )
    assert (
        config.get_azure_credentials(config.smart_llm)["deployment_id"]
        == "SMART-LLM_ID"
    )

    # Emulate --gpt4only
    config.fast_llm = smart_llm
    assert (
        config.get_azure_credentials(config.fast_llm)["deployment_id"] == "SMART-LLM_ID"
    )
    assert (
        config.get_azure_credentials(config.smart_llm)["deployment_id"]
        == "SMART-LLM_ID"
    )

    # Emulate --gpt3only
    config.fast_llm = config.smart_llm = fast_llm
    assert (
        config.get_azure_credentials(config.fast_llm)["deployment_id"] == "FAST-LLM_ID"
    )
    assert (
        config.get_azure_credentials(config.smart_llm)["deployment_id"] == "FAST-LLM_ID"
    )

    del os.environ["USE_AZURE"]
    del os.environ["AZURE_CONFIG_FILE"]


def test_create_config_gpt4only(config: Config) -> None:
    with mock.patch("autogpt.llm.api_manager.ApiManager.get_models") as mock_get_models:
        mock_get_models.return_value = [{"id": GPT_4_MODEL}]
        create_config(
            config=config,
            continuous=False,
            continuous_limit=None,
            ai_settings_file=None,
            prompt_settings_file=None,
            skip_reprompt=False,
            speak=False,
            debug=False,
            gpt3only=False,
            gpt4only=True,
            memory_type=None,
            browser_name=None,
            allow_downloads=False,
            skip_news=False,
        )
        assert config.fast_llm == GPT_4_MODEL
        assert config.smart_llm == GPT_4_MODEL


def test_create_config_gpt3only(config: Config) -> None:
    with mock.patch("autogpt.llm.api_manager.ApiManager.get_models") as mock_get_models:
        mock_get_models.return_value = [{"id": GPT_3_MODEL}]
        create_config(
            config=config,
            continuous=False,
            continuous_limit=None,
            ai_settings_file=None,
            prompt_settings_file=None,
            skip_reprompt=False,
            speak=False,
            debug=False,
            gpt3only=True,
            gpt4only=False,
            memory_type=None,
            browser_name=None,
            allow_downloads=False,
            skip_news=False,
        )
        assert config.fast_llm == GPT_3_MODEL
        assert config.smart_llm == GPT_3_MODEL
