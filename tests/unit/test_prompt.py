import builtins
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, Mock, call, patch

import pytest
from _pytest.capture import CaptureFixture

from autogpt.config.config import Config
from autogpt.prompts.prompt import (
    generate_unique_name,
    manage_ai_name,
    start_prompt,
    validate_input,
)

config = Config()


class ConfigsMock:
    ai_name = ""


@pytest.fixture(autouse=True)
def setup(tmp_path: Path) -> Generator[None, None, None]:
    config.ai_settings_filepath = tmp_path / "ai_settings.yaml"
    config.workspace_path = tmp_path / "auto_gpt_workspace"
    (config.workspace_path).mkdir(parents=True, exist_ok=True)
    yield

    if config.ai_settings_filepath.exists():
        config.ai_settings_filepath.unlink()


@patch.object(builtins, "input", side_effect=["-invalid", "--invalid", "valid"])
def test_validate_input(mock_input: MagicMock) -> None:
    assert validate_input("Enter text: ") == "valid"
    assert mock_input.call_count == 3


def test_generate_unique_name() -> None:
    mock_ai_configs = {
        "config1": MagicMock(ai_name="base-1"),
        "config2": MagicMock(ai_name="base-2"),
        "config3": MagicMock(ai_name="nonrelevant-1"),
    }

    with patch(
        "autogpt.config.ai_config.AIConfig.load_all", return_value=(mock_ai_configs, "")
    ):
        name = generate_unique_name(config, "base")
        assert name == "base-3"

    mock_ai_configs = {
        "config1": MagicMock(ai_name="nonrelevant-1"),
        "config2": MagicMock(ai_name="nonrelevant-2"),
        "config3": MagicMock(ai_name="nonrelevant-3"),
    }

    with patch(
        "autogpt.config.ai_config.AIConfig.load_all", return_value=(mock_ai_configs, "")
    ):
        name = generate_unique_name(config, "base")
        assert name == "base-1"


@patch("autogpt.prompts.prompt.check_ai_name_exists", return_value=False)
@patch("autogpt.prompts.prompt.validate_input", side_effect=lambda x: x)
@patch("autogpt.utils.session.prompt", return_value="test_name")
def test_manage_ai_name_create(
    mock_session_prompt: MagicMock,
    mock_validate_input: MagicMock,
    mock_check_name: MagicMock,
) -> None:
    configs = ConfigsMock()
    task = "create"
    result = manage_ai_name(config, configs, task)
    assert result == "test_name"
    assert configs.ai_name == "test_name"
    mock_check_name.assert_called_once_with(config, "test_name")


@patch("autogpt.prompts.prompt.check_ai_name_exists", return_value=False)
@patch("autogpt.prompts.prompt.validate_input", side_effect=lambda x: x)
@patch("autogpt.utils.session.prompt", side_effect=["", "Y"])
def test_manage_ai_name_edit_keep_current(
    mock_session_prompt: MagicMock,
    mock_validate_input: MagicMock,
    mock_check_name: MagicMock,
) -> None:
    configs = ConfigsMock()
    configs.ai_name = "current_name"
    task = "edit"
    result = manage_ai_name(config, configs, task)
    assert result == "current_name"
    assert configs.ai_name == "current_name"
    mock_check_name.assert_called_once_with(config, "current_name")


@patch("autogpt.prompts.prompt.check_ai_name_exists", return_value=False)
@patch("autogpt.prompts.prompt.validate_input", side_effect=lambda x: x)
@patch("autogpt.utils.session.prompt", side_effect=["new_name"])
def test_manage_ai_name_edit_change_current(
    mock_session_prompt: MagicMock,
    mock_validate_input: MagicMock,
    mock_check_name: MagicMock,
) -> None:
    configs = ConfigsMock()
    configs.ai_name = "current_name"
    task = "edit"
    result = manage_ai_name(config, configs, task)
    assert result == "new_name"
    assert configs.ai_name == "new_name"
    mock_check_name.assert_called_once_with(config, "new_name")


def test_start_prompt(capsys: CaptureFixture) -> None:
    # Mock the necessary dependencies and set up the initial config
    tmp_cfg = Mock()
    tmp_cfg.ai_name = "Test AI"
    tmp_cfg.ai_role = "Test role"
    tmp_cfg.ai_goals = ["Goal 1", "Goal 2"]
    tmp_cfg.api_budget = 100.0
    tmp_cfg.plugins = [
        "Plugin 1",
        "Plugin 2",
    ]  # Provide a valid iterable value for plugins

    with patch("autogpt.prompts.prompt.logger.typewriter_log") as mock_typewriter_log:
        with patch("autogpt.prompts.prompt.ApiManager") as mock_api_manager:
            # Set up the mock return value for the total budget
            mock_api_manager.return_value.set_total_budget.return_value = None

            # Call the start_prompt function
            start_prompt(config, tmp_cfg)

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
