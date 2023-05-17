"""
Test cases for the Config class, which handles the configuration settings
for the AI and ensures it behaves as a singleton.
"""
from unittest.mock import patch

from openai import InvalidRequestError

from autogpt.llm.llm_utils import check_smart_llm_model_access


def test_initial_values(config):
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


def test_set_continuous_mode(config):
    """
    Test if the set_continuous_mode() method updates the continuous_mode attribute.
    """
    # Store continuous mode to reset it after the test
    continuous_mode = config.continuous_mode

    config.set_continuous_mode(True)
    assert config.continuous_mode == True

    # Reset continuous mode
    config.set_continuous_mode(continuous_mode)


def test_set_speak_mode(config):
    """
    Test if the set_speak_mode() method updates the speak_mode attribute.
    """
    # Store speak mode to reset it after the test
    speak_mode = config.speak_mode

    config.set_speak_mode(True)
    assert config.speak_mode == True

    # Reset speak mode
    config.set_speak_mode(speak_mode)


def test_set_fast_llm_model(config):
    """
    Test if the set_fast_llm_model() method updates the fast_llm_model attribute.
    """
    # Store model name to reset it after the test
    fast_llm_model = config.fast_llm_model

    config.set_fast_llm_model("gpt-3.5-turbo-test")
    assert config.fast_llm_model == "gpt-3.5-turbo-test"

    # Reset model name
    config.set_fast_llm_model(fast_llm_model)


def test_set_smart_llm_model(config):
    """
    Test if the set_smart_llm_model() method updates the smart_llm_model attribute.
    """
    # Store model name to reset it after the test
    smart_llm_model = config.smart_llm_model

    config.set_smart_llm_model("gpt-4-test")
    assert config.smart_llm_model == "gpt-4-test"

    # Reset model name
    config.set_smart_llm_model(smart_llm_model)


def test_set_fast_token_limit(config):
    """
    Test if the set_fast_token_limit() method updates the fast_token_limit attribute.
    """
    # Store token limit to reset it after the test
    fast_token_limit = config.fast_token_limit

    config.set_fast_token_limit(5000)
    assert config.fast_token_limit == 5000

    # Reset token limit
    config.set_fast_token_limit(fast_token_limit)


def test_set_smart_token_limit(config):
    """
    Test if the set_smart_token_limit() method updates the smart_token_limit attribute.
    """
    # Store token limit to reset it after the test
    smart_token_limit = config.smart_token_limit

    config.set_smart_token_limit(9000)
    assert config.smart_token_limit == 9000

    # Reset token limit
    config.set_smart_token_limit(smart_token_limit)


def test_set_debug_mode(config):
    """
    Test if the set_debug_mode() method updates the debug_mode attribute.
    """
    # Store debug mode to reset it after the test
    debug_mode = config.debug_mode

    config.set_debug_mode(True)
    assert config.debug_mode == True

    # Reset debug mode
    config.set_debug_mode(debug_mode)


def test_check_smart_llm_model_access(config):
    with patch("openai.ChatCompletion.create") as mock_create_chat_completion:
        # Test when no InvalidRequestError is raised
        smart_model = config.smart_llm_model
        check_smart_llm_model_access()  # should not change smart_llm_model
        assert config.smart_llm_model == smart_model

        # Test when InvalidRequestError is raised
        mock_create_chat_completion.side_effect = InvalidRequestError(
            "error message", "error param"
        )
        check_smart_llm_model_access()  # should set smart_llm_model to fast_llm_model
        assert config.smart_llm_model == config.fast_llm_model
