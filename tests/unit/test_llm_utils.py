from unittest.mock import patch

from autogpt.llm import llm_utils
from autogpt.llm.llm_utils import check_model


def test_chunked_tokens():
    text = "Auto-GPT is an experimental open-source application showcasing the capabilities of the GPT-4 language model"
    expected_output = [
        (
            13556,
            12279,
            2898,
            374,
            459,
            22772,
            1825,
            31874,
            3851,
            67908,
            279,
            17357,
            315,
            279,
            480,
            2898,
            12,
            19,
            4221,
            1646,
        )
    ]
    output = list(llm_utils.chunked_tokens(text, "cl100k_base", 8191))
    assert output == expected_output


def test_check_model(api_manager):
    """
    Test if check_model() returns original model when valid.
    Test if check_model() returns gpt-3.5-turbo when model is invalid.
    """
    with patch("openai.Model.list") as mock_list_models:
        # Test when correct model is returned
        mock_list_models.return_value = {"data": [{"id": "gpt-4"}]}
        result = check_model("gpt-4", "smart_llm_model")
        assert result == "gpt-4"

        # Reset api manager models
        api_manager.models = None

        # Test when incorrect model is returned
        mock_list_models.return_value = {"data": [{"id": "gpt-3.5-turbo"}]}
        result = check_model("gpt-4", "fast_llm_model")
        assert result == "gpt-3.5-turbo"

        # Reset api manager models
        api_manager.models = None
