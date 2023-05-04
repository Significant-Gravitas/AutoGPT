"""
Test cases for the Config class, which handles the configuration settings
for the AI and ensures it behaves as a singleton.
"""

import os
from unittest.mock import MagicMock, mock_open, patch

import pytest

from autogpt.config import Config

# I think we can remove this test - it is not testing anything
# def test_initial_values(config):
#     """
#     Test if the initial values of the Config class attributes are set correctly.
#     """
#     assert config.debug_mode == False
#     assert config.continuous_mode == False
#     assert config.speak_mode == False
#     assert config.fast_llm_model == "gpt-3.5-turbo"
#     assert config.smart_llm_model == "gpt-4"
#     assert config.fast_token_limit == 4000
#     assert config.smart_token_limit == 8000


def test_set_continuous_mode(config):
    """
    Test if the set_continuous_mode() method updates the continuous_mode attribute.
    """
    # Store continuous mode to reset it after the test
    continuous_mode = config.continuous_mode

    config.set_continuous_mode(True)
    assert config.continuous_mode == True


def test_set_speak_mode(config):
    """
    Test if the set_speak_mode() method updates the speak_mode attribute.
    """
    # Store speak mode to reset it after the test
    speak_mode = config.speak_mode

    config.set_speak_mode(True)
    assert config.speak_mode == True


def test_set_fast_llm_model(config):
    """
    Test if the set_fast_llm_model() method updates the fast_llm_model attribute.
    """
    # Store model name to reset it after the test
    fast_llm_model = config.fast_llm_model

    config.set_fast_llm_model("gpt-3.5-turbo-test")
    assert config.fast_llm_model == "gpt-3.5-turbo-test"


def test_set_smart_llm_model(config):
    """
    Test if the set_smart_llm_model() method updates the smart_llm_model attribute.
    """
    # Store model name to reset it after the test
    smart_llm_model = config.smart_llm_model

    config.set_smart_llm_model("gpt-4-test")
    assert config.smart_llm_model == "gpt-4-test"


def test_set_fast_token_limit(config):
    """
    Test if the set_fast_token_limit() method updates the fast_token_limit attribute.
    """
    # Store token limit to reset it after the test
    fast_token_limit = config.fast_token_limit

    config.set_fast_token_limit(5000)
    assert config.fast_token_limit == 5000


def test_set_smart_token_limit(config):
    """
    Test if the set_smart_token_limit() method updates the smart_token_limit attribute.
    """
    # Store token limit to reset it after the test
    smart_token_limit = config.smart_token_limit

    config.set_smart_token_limit(9000)
    assert config.smart_token_limit == 9000


def test_set_debug_mode(config):
    """
    Test if the set_debug_mode() method updates the debug_mode attribute.
    """
    # Store debug mode to reset it after the test
    debug_mode = config.debug_mode

    config.set_debug_mode(True)
    assert config.debug_mode == True


# Generalized mock function
def mock_load_yaml_config(config, mock_data, config_file, *args, **kwargs):
    config.yaml_config = mock_data or {}


def mock_load_dotenv(*args, **kwargs):
    os.environ["MY_NEW_AUTOGPT_CONFIG"] = "dot_env_value"


# The new tests with custom mocks
@pytest.mark.parametrize(
    "config, expected_value",
    [
        (
            {
                "autogpt.config.Config.load_yaml_config": (
                    lambda config, config_file=None, *args, **kwargs: mock_load_yaml_config(
                        config,
                        {"my_new_autogpt_config": "some_value"},
                        config_file,
                        *args,
                        **kwargs,
                    )
                ),
            },
            {"my_new_autogpt_config": "some_value"},
        ),
        (
            {
                "autogpt.config.Config.load_yaml_config": (
                    lambda config, config_file=None, *args, **kwargs: mock_load_yaml_config(
                        config,
                        {"another_new_autogpt_config": "another_value"},
                        config_file,
                        *args,
                        **kwargs,
                    )
                ),
            },
            {"another_new_autogpt_config": "another_value"},
        ),
    ],
    indirect=["config"],
)
def test_yaml_config_mock_only(config, expected_value):
    # Mock the load_yaml_config function to directly set the yaml_config attribute
    config.load_yaml_config = lambda *args, **kwargs: mock_load_yaml_config(
        config, expected_value, *args, **kwargs
    )

    config.load_yaml_config(config_file="mock_command_config.yaml")

    for key, value in expected_value.items():
        assert config.yaml_config[key] == value


@pytest.mark.parametrize(
    "config, expected_value",
    [
        (
            {
                "autogpt.config.Config.load_yaml_config": (
                    lambda config, config_file=None, *args, **kwargs: mock_load_yaml_config(
                        config,
                        {"my_new_autogpt_config": "yaml_value"},
                        config_file,
                        *args,
                        **kwargs,
                    )
                ),
                "mock_load_dotenv": mock_load_dotenv,
                "env_vars": {"MY_NEW_AUTOGPT_CONFIG": "env_var_value"},
            },
            "env_var_value",
        ),
    ],
    indirect=["config"],
)
def test_precedence_order(config, expected_value):
    # Mock the load_yaml_config function to directly set the yaml_config attribute
    config.load_yaml_config = lambda *args, **kwargs: mock_load_yaml_config(
        config, {"my_new_autogpt_config": "yaml_value"}, *args, **kwargs
    )

    config.load_yaml_config(config_file="mock_command_config.yaml")
    assert config.get_env_or_yaml_config("MY_NEW_AUTOGPT_CONFIG") == expected_value


# TODO add test for command line options
