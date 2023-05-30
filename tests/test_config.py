"""
Test cases for the Config class, which handles the configuration settings
for the AI and ensures it behaves as a singleton.
"""
from unittest import mock
from unittest.mock import patch

import pytest

from autogpt.config.config import Config
from autogpt.configurator import GPT_3_MODEL, GPT_4_MODEL, create_config
from autogpt.workspace.workspace import Workspace


def test_initial_values(config: Config):
    """
    Test if the initial values of the Config class attributes are set correctly.
    """
    assert config.debug_mode == False
    assert config.continuous_mode == False
    assert config.speak_mode == False
    assert config.fast_llm_model == "gpt-3.5-turbo"
    assert config.smart_llm_model == "gpt-4"
    assert config.fast_token_limit == 4000
    assert config.smart_token_limit == 8000


def test_set_continuous_mode(config: Config):
    """
    Test if the set_continuous_mode() method updates the continuous_mode attribute.
    """
    # Store continuous mode to reset it after the test
    continuous_mode = config.continuous_mode

    config.set_continuous_mode(True)
    assert config.continuous_mode == True

    # Reset continuous mode
    config.set_continuous_mode(continuous_mode)


def test_set_speak_mode(config: Config):
    """
    Test if the set_speak_mode() method updates the speak_mode attribute.
    """
    # Store speak mode to reset it after the test
    speak_mode = config.speak_mode

    config.set_speak_mode(True)
    assert config.speak_mode == True

    # Reset speak mode
    config.set_speak_mode(speak_mode)


def test_set_fast_llm_model(config: Config):
    """
    Test if the set_fast_llm_model() method updates the fast_llm_model attribute.
    """
    # Store model name to reset it after the test
    fast_llm_model = config.fast_llm_model

    config.set_fast_llm_model("gpt-3.5-turbo-test")
    assert config.fast_llm_model == "gpt-3.5-turbo-test"

    # Reset model name
    config.set_fast_llm_model(fast_llm_model)


def test_set_smart_llm_model(config: Config):
    """
    Test if the set_smart_llm_model() method updates the smart_llm_model attribute.
    """
    # Store model name to reset it after the test
    smart_llm_model = config.smart_llm_model

    config.set_smart_llm_model("gpt-4-test")
    assert config.smart_llm_model == "gpt-4-test"

    # Reset model name
    config.set_smart_llm_model(smart_llm_model)


def test_set_fast_token_limit(config: Config):
    """
    Test if the set_fast_token_limit() method updates the fast_token_limit attribute.
    """
    # Store token limit to reset it after the test
    fast_token_limit = config.fast_token_limit

    config.set_fast_token_limit(5000)
    assert config.fast_token_limit == 5000

    # Reset token limit
    config.set_fast_token_limit(fast_token_limit)


def test_set_smart_token_limit(config: Config):
    """
    Test if the set_smart_token_limit() method updates the smart_token_limit attribute.
    """
    # Store token limit to reset it after the test
    smart_token_limit = config.smart_token_limit

    config.set_smart_token_limit(9000)
    assert config.smart_token_limit == 9000

    # Reset token limit
    config.set_smart_token_limit(smart_token_limit)


def test_set_debug_mode(config: Config):
    """
    Test if the set_debug_mode() method updates the debug_mode attribute.
    """
    # Store debug mode to reset it after the test
    debug_mode = config.debug_mode

    config.set_debug_mode(True)
    assert config.debug_mode == True

    # Reset debug mode
    config.set_debug_mode(debug_mode)


@patch("openai.Model.list")
def test_smart_and_fast_llm_models_set_to_gpt4(mock_list_models, config: Config):
    """
    Test if models update to gpt-3.5-turbo if both are set to gpt-4.
    """
    fast_llm_model = config.fast_llm_model
    smart_llm_model = config.smart_llm_model

    config.fast_llm_model = "gpt-4"
    config.smart_llm_model = "gpt-4"

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

    assert config.fast_llm_model == "gpt-3.5-turbo"
    assert config.smart_llm_model == "gpt-3.5-turbo"

    # Reset config
    config.set_fast_llm_model(fast_llm_model)
    config.set_smart_llm_model(smart_llm_model)


def test_missing_azure_config(config: Config, workspace: Workspace):
    config_file = workspace.get_path("azure_config.yaml")
    with pytest.raises(FileNotFoundError):
        config.load_azure_config(str(config_file))

    config_file.write_text("")
    config.load_azure_config(str(config_file))

    assert config.openai_api_type == "azure"
    assert config.openai_api_base == ""
    assert config.openai_api_version == "2023-03-15-preview"
    assert config.azure_model_to_deployment_id_map == {}


def test_create_config_gpt4only(config: Config) -> None:
    fast_llm_model = config.fast_llm_model
    smart_llm_model = config.smart_llm_model
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
        assert config.fast_llm_model == GPT_4_MODEL
        assert config.smart_llm_model == GPT_4_MODEL

    # Reset config
    config.set_fast_llm_model(fast_llm_model)
    config.set_smart_llm_model(smart_llm_model)


def test_create_config_gpt3only(config: Config) -> None:
    fast_llm_model = config.fast_llm_model
    smart_llm_model = config.smart_llm_model
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
        assert config.fast_llm_model == GPT_3_MODEL
        assert config.smart_llm_model == GPT_3_MODEL

    # Reset config
    config.set_fast_llm_model(fast_llm_model)
    config.set_smart_llm_model(smart_llm_model)
