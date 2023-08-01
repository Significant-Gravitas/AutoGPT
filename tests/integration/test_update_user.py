from unittest.mock import MagicMock, patch

from colorama import Fore

from autogpt.app.main import update_user


def test_update_user_command_name_is_none() -> None:
    # Mock necessary objects
    config = MagicMock()
    ai_config = MagicMock()
    assistant_reply_dict = MagicMock()

    # Mock print_assistant_thoughts and logger.typewriter_log
    with patch(
        "autogpt.app.main.print_assistant_thoughts"
    ) as mock_print_assistant_thoughts, patch(
        "autogpt.app.main.logger.typewriter_log"
    ) as mock_logger_typewriter_log:
        # Test the update_user function with None command_name
        update_user(config, ai_config, None, None, assistant_reply_dict)

    # Check that print_assistant_thoughts was called once
    mock_print_assistant_thoughts.assert_called_once_with(
        ai_config.ai_name, assistant_reply_dict, config
    )

    # Check that logger.typewriter_log was called once with expected arguments
    mock_logger_typewriter_log.assert_called_once_with(
        "NO ACTION SELECTED: ",
        Fore.RED,
        f"The Agent failed to select an action.",
    )
