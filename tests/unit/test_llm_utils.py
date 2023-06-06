from unittest.mock import patch

from autogpt.llm import utils as llm_utils


def test_check_model(api_manager):
    """
    Test if check_model() returns original model when valid.
    Test if check_model() returns gpt-3.5-turbo when model is invalid.
    """
    with patch("openai.Model.list") as mock_list_models:
        # Test when correct model is returned
        mock_list_models.return_value = {"data": [{"id": "gpt-4"}]}
        result = llm_utils.check_model("gpt-4", "smart_llm_model")
        assert result == "gpt-4"

        # Reset api manager models
        api_manager.models = None

        # Test when incorrect model is returned
        mock_list_models.return_value = {"data": [{"id": "gpt-3.5-turbo"}]}
        result = llm_utils.check_model("gpt-4", "fast_llm_model")
        assert result == "gpt-3.5-turbo"

        # Reset api manager models
        api_manager.models = None
