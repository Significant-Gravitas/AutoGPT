import builtins
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from autogpt.prompts.prompt import (
    cfg,
    generate_unique_name,
    manage_ai_name,
    start_prompt,
    validate_input,
)


class ConfigsMock:
    ai_name = ""


@pytest.fixture(autouse=True)
def setup(tmp_path):
    cfg.ai_settings_filepath = tmp_path / "ai_settings.yaml"
    cfg.workspace_path = tmp_path / "auto_gpt_workspace"
    (cfg.workspace_path).mkdir(parents=True, exist_ok=True)
    yield

    if cfg.ai_settings_filepath.exists():
        cfg.ai_settings_filepath.unlink()


@patch.object(builtins, "input", side_effect=["-invalid", "--invalid", "valid"])
def test_validate_input(mock_input):
    assert validate_input("Enter text: ") == "Enter text: "
    assert mock_input.call_count == 3


def test_generate_unique_name():
    mock_ai_configs = {
        "config1": MagicMock(ai_name="base-1"),
        "config2": MagicMock(ai_name="base-2"),
        "config3": MagicMock(ai_name="nonrelevant-1"),
    }

    with patch(
        "autogpt.config.ai_config.AIConfig.load_all", return_value=mock_ai_configs
    ):
        name = generate_unique_name("base")
        assert name == "base-3"

    mock_ai_configs = {
        "config1": MagicMock(ai_name="nonrelevant-1"),
        "config2": MagicMock(ai_name="nonrelevant-2"),
        "config3": MagicMock(ai_name="nonrelevant-3"),
    }

    with patch(
        "autogpt.config.ai_config.AIConfig.load_all", return_value=mock_ai_configs
    ):
        name = generate_unique_name("base")
        assert name == "base-1"


@patch("autogpt.prompts.prompt.check_ai_name_exists", return_value=False)
@patch("autogpt.prompts.prompt.validate_input", side_effect=lambda x: x)
@patch("autogpt.utils.session.prompt", return_value="test_name")
def test_manage_ai_name_create(
    mock_session_prompt, mock_validate_input, mock_check_name
):
    configs = ConfigsMock()
    task = "create"
    result = manage_ai_name(configs, task)
    assert result == "test_name"
    assert configs.ai_name == "test_name"
    mock_check_name.assert_called_once_with("test_name")


@patch("autogpt.prompts.prompt.check_ai_name_exists", return_value=False)
@patch("autogpt.prompts.prompt.validate_input", side_effect=lambda x: x)
@patch("autogpt.utils.session.prompt", side_effect=["", "Y"])
def test_manage_ai_name_edit_keep_current(
    mock_session_prompt, mock_validate_input, mock_check_name
):
    configs = ConfigsMock()
    configs.ai_name = "current_name"
    task = "edit"
    result = manage_ai_name(configs, task)
    assert result == "current_name"
    assert configs.ai_name == "current_name"
    mock_check_name.assert_called_once_with("current_name")


@patch("autogpt.prompts.prompt.check_ai_name_exists", return_value=False)
@patch("autogpt.prompts.prompt.validate_input", side_effect=lambda x: x)
@patch("autogpt.utils.session.prompt", side_effect=["new_name"])
def test_manage_ai_name_edit_change_current(
    mock_session_prompt, mock_validate_input, mock_check_name
):
    configs = ConfigsMock()
    configs.ai_name = "current_name"
    task = "edit"
    result = manage_ai_name(configs, task)
    assert result == "new_name"
    assert configs.ai_name == "new_name"
    mock_check_name.assert_called_once_with("new_name")


def test_start_prompt(capsys):
    # Mock the necessary dependencies and set up the initial config
    config = Mock()
    config.ai_name = "Test AI"
    config.ai_role = "Test role"
    config.ai_goals = ["Goal 1", "Goal 2"]
    config.api_budget = 100.0
    config.plugins = [
        "Plugin 1",
        "Plugin 2",
    ]  # Provide a valid iterable value for plugins

    with patch("autogpt.prompts.prompt.logger.typewriter_log") as mock_typewriter_log:
        with patch("autogpt.prompts.prompt.ApiManager") as mock_api_manager:
            # Set up the mock return value for the total budget
            mock_api_manager.return_value.set_total_budget.return_value = None

            # Call the start_prompt function
            start_prompt(config)

            # Capture the printed output
            captured = capsys.readouterr()

            recorded_calls = mock_typewriter_log.call_args_list

            # Perform assertions on the recorded calls
            expected_call = call("Name:", "\x1b[32m", "Test AI", speak_text=False)
            assert len(recorded_calls) == 12
            assert any(expected_call == c for c in recorded_calls)
            assert (
                call("Role:", "\x1b[32m", "Test role", speak_text=False)
                in recorded_calls
            )
            assert (
                call("Budget:", "\x1b[32m", "$100.0", speak_text=False)
                in recorded_calls
            )

            # Perform assertions on the order of recorded calls
            assert recorded_calls.index(expected_call) < recorded_calls.index(
                call("Role:", "\x1b[32m", "Test role", speak_text=False)
            )


if __name__ == "__main__":
    pytest.main()
