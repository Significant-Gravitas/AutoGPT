import builtins
from unittest.mock import patch


from autogpt.prompts.prompt import manage_ai_name, validate_input


class ConfigsMock:
    ai_name = ""


@patch.object(builtins, "input", side_effect=["-invalid", "--invalid", "valid"])
def test_validate_input(mock_input):
    assert validate_input("Enter text: ") == "Enter text: "
    assert mock_input.call_count == 3


@patch("autogpt.prompts.prompt.check_ai_name_exists", return_value=False)
@patch("autogpt.prompts.prompt.validate_input", side_effect=lambda x: x)
@patch.object(builtins, "input", return_value="test_name")
def test_manage_ai_name_create(mock_input, mock_validate_input, mock_check_name):
    configs = ConfigsMock()
    task = "create"
    result = manage_ai_name(configs, task)
    assert result == "test_name"
    assert configs.ai_name == "test_name"
    mock_input.assert_called_once_with("AI Name: ")
    mock_check_name.assert_called_once_with("test_name")


@patch("autogpt.prompts.prompt.check_ai_name_exists", return_value=False)
@patch("autogpt.prompts.prompt.validate_input", side_effect=lambda x: x)
@patch.object(builtins, "input", return_value="")
def test_manage_ai_name_edit_keep_current(
    mock_input, mock_validate_input, mock_check_name
):
    configs = ConfigsMock()
    configs.ai_name = "current_name"
    task = "edit"
    result = manage_ai_name(configs, task)
    assert result == "current_name"
    assert configs.ai_name == "current_name"
    mock_input.assert_called_once_with("AI Name: ")
    mock_check_name.assert_called_once_with("current_name")


@patch("autogpt.prompts.prompt.check_ai_name_exists", return_value=False)
@patch("autogpt.prompts.prompt.validate_input", side_effect=lambda x: x)
@patch.object(builtins, "input", return_value="new_name")
def test_manage_ai_name_edit_change_current(
    mock_input, mock_validate_input, mock_check_name
):
    configs = ConfigsMock()
    configs.ai_name = "current_name"
    task = "edit"
    result = manage_ai_name(configs, task)
    assert result == "new_name"
    assert configs.ai_name == "new_name"
    mock_input.assert_called_once_with("AI Name: ")
    mock_check_name.assert_called_once_with("new_name")
