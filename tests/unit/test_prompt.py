import builtins
from itertools import cycle
from unittest.mock import MagicMock, Mock, call, patch

import pytest
from colorama import Fore, Style

from autogpt.config.ai_config import AIConfig
from autogpt.prompts.prompt import (
    cfg,
    generate_unique_name,
    handle_config,
    handle_configs,
    manage_ai_name,
    start_prompt,
    validate_input,
    welcome_prompt,
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


@patch("autogpt.main.start_agent_directly")
@patch("autogpt.llm.api_manager.ApiManager")
@patch("autogpt.logs.logger")
def test_start_prompt(mock_logger, mock_api_manager, mock_start_agent_directly):
    with patch("autogpt.logs.logger") as mock_logger:
        config = Mock()
        config.ai_name = "test_name"
        config.ai_role = "test_role"
        config.ai_goals = ["goal1", "goal2"]
        config.api_budget = 100
        config.plugins = ["plugin1", "plugin2"]

        cfg = Mock()
        cfg.skip_reprompt = True
        cfg.restrict_to_workspace = True
        cfg.workspace_path = "workspace_path"

        mock_api_manager_instance = mock_api_manager.return_value

        start_prompt(config, cfg)

        # Check if the logger is called with the right arguments
        calls = [
            call("Name :", Fore.GREEN, config.ai_name),
            call("Role :", Fore.GREEN, config.ai_role),
            call("Goals:", Fore.GREEN, f"{config.ai_goals}"),
            call("Budget:", Fore.GREEN, f"${str(config.api_budget)}"),
            call(
                f"{Fore.YELLOW}NOTE:All files/directories created by this agent can be found inside its workspace at:{Style.RESET_ALL}"
            ),
            call(f"{Fore.YELLOW}-{Style.RESET_ALL}   {cfg.workspace_path}"),
            call(
                f"{Fore.LIGHTBLUE_EX}Auto-GPT has started with the following details:{Style.RESET_ALL}",
                speak_text=True,
            ),
            call("Name:", Fore.GREEN, config.ai_name, speak_text=False),
            call("Role:", Fore.GREEN, config.ai_role, speak_text=False),
            call("Goals:", Fore.GREEN, "", speak_text=False),
            call("-", Fore.GREEN, config.ai_goals[0], speak_text=False),
            call("-", Fore.GREEN, config.ai_goals[1], speak_text=False),
            call("Budget:", Fore.GREEN, f"${str(config.api_budget)}", speak_text=False),
            call("Plugins:", Fore.GREEN, "", speak_text=False),
            call("-", Fore.GREEN, config.plugins[0], speak_text=False),
            call("-", Fore.GREEN, config.plugins[1], speak_text=False),
        ]
        mock_logger.typewriter_log.assert_has_calls(calls, any_order=True)

        # Check that the API manager budget was set correctly
        mock_api_manager_instance.set_total_budget.assert_called_once_with(
            config.api_budget
        )

        # Check that the start_agent_directly was not called
        mock_start_agent_directly.assert_not_called()

        # Check with sad parameter
        start_prompt(config, cfg, sad=True)
        mock_start_agent_directly.assert_called_once_with(config, cfg)


@patch("autogpt.logs.logger.typewriter_log")
@patch("autogpt.utils.clean_input", side_effect=cycle(["", "--manual", "other"]))
@patch("autogpt.prompts.prompt.handle_config")
@patch("autogpt.setup.generate_aiconfig_automatic")
def test_welcome_prompt(mock_generate, mock_handle, mock_input, mock_log):
    # first test with an empty input
    welcome_prompt()
    mock_log.assert_called()  # We expect log to be called multiple times
    mock_handle.assert_called_once_with(None, "create")

    # reset mocks
    mock_handle.reset_mock()

    # second test with '--manual' input
    welcome_prompt()
    mock_handle.assert_called_once_with(None, "create")

    # reset mocks
    mock_handle.reset_mock()

    # third test with an exception when calling generate_aiconfig_automatic
    mock_generate.side_effect = Exception("Error")
    mock_input.side_effect = ["other"]  # add this line
    welcome_prompt()
    mock_handle.assert_called_once_with(None, "create")

    # reset mocks
    mock_generate.reset_mock()
    mock_handle.reset_mock()
    mock_input.side_effect = ["other"]  # add this line


@patch.object(builtins, "input", side_effect=["-invalid", "--invalid", "valid"])
def test_validate_input(mock_input):
    assert validate_input("Enter text: ") == "Enter text: "
    assert mock_input.call_count == 3


def test_generate_unique_name():
    mock_ai_configs = {  # Defining some mock configurations
        "config1": MagicMock(ai_name="base-1"),
        "config2": MagicMock(ai_name="base-2"),
        "config3": MagicMock(ai_name="nonrelevant-1"),
    }

    with patch(
        "autogpt.config.ai_config.AIConfig.load_all", return_value=mock_ai_configs
    ):
        name = generate_unique_name("base")
        assert name == "base-3"

    # Now we'll test for a name that doesn't exist yet
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


def test_handle_configs(monkeypatch):
    # Test case 1: Test with a valid config and task
    config = AIConfig(...)
    task = "edit"
    result = handle_configs(config, task)
    assert (
        len(result) == 5
    )  # Assert that all functions are called and returned a result

    # Test case 2: Test with None config
    config = None
    task = "edit"
    result = handle_configs(config, task)
    assert (
        len(result) == 5
    )  # Assert that all functions are called and returned a result

    # Test case 3: Test with empty config_functions list
    config = AIConfig(...)
    task = "edit"
    handle_configs.config_functions = []
    result = handle_configs(config, task)
    assert len(result) == 0  # Assert that no functions are called

    # Add more test cases to cover other scenarios


# Test handle_config function
def test_handle_config(monkeypatch):
    # Test case 1: Test with a valid config and task
    config = AIConfig(...)
    task = "edit"
    result = handle_config(config, task)
    assert isinstance(
        result, AIConfig
    )  # Assert that the result is an instance of AIConfig

    # Test case 2: Test with None config
    config = None
    task = "edit"
    result = handle_config(config, task)
    assert isinstance(
        result, AIConfig
    )  # Assert that the result is an instance of AIConfig

    # Test case 3: Test with task other than "edit" and empty config
    config = None
    task = "create"
    result = handle_config(config, task)
    assert isinstance(
        result, AIConfig
    )  # Assert that the result is an instance of AIConfig

    # Add more test cases to cover other scenarios


# Run the tests
if __name__ == "__main__":
    pytest.main()
