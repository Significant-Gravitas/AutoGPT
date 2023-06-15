from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from autogpt.config.ai_config import AIConfig
from autogpt.prompts.prompt import cfg, main_menu, welcome_prompt
from autogpt.setup import generate_aiconfig_automatic
from tests.utils import requires_api_key


@pytest.fixture(autouse=True)
def setup(tmp_path: Path) -> Any:
    cfg.ai_settings_filepath = tmp_path / "ai_settings.yaml"
    cfg.workspace_path = tmp_path / "auto_gpt_workspace"
    (cfg.workspace_path).mkdir(parents=True, exist_ok=True)
    cfg.plugins_allowlist = ["plugin1", "plugin2", "plugin3"]
    yield

    if cfg.ai_settings_filepath.exists():
        cfg.ai_settings_filepath.unlink()


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
def test_generate_aiconfig_automatic_default(patched_api_requestor: Mock) -> None:
    user_inputs = [""]

    with patch("autogpt.utils.session.prompt", side_effect=user_inputs):
        ai_config = welcome_prompt()

    assert isinstance(ai_config, AIConfig)
    assert ai_config.ai_name is not None
    assert ai_config.ai_role is not None
    assert 1 <= len(ai_config.ai_goals) <= 5


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
def test_generate_aiconfig_automatic_typical(patched_api_requestor: Mock) -> None:
    user_prompt = "Help me create a rock opera about cybernetic giraffes"
    ai_config = generate_aiconfig_automatic(user_prompt)

    assert isinstance(ai_config, AIConfig)
    assert ai_config.ai_name is not None
    assert ai_config.ai_role is not None
    assert 1 <= len(ai_config.ai_goals) <= 5


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
def test_generate_aiconfig_automatic_fallback(patched_api_requestor: Mock) -> None:
    user_inputs = [
        "T&GF£OIBECC()!*",
        "Chef-GPT",
        "an AI designed to browse bake a cake.",
        "2",
        "Purchase ingredients",
        "Bake a cake",
        "",
        "",
        "",
        "3.00",
        "r",
        "1",
    ]

    user_inputs_iterator = iter(user_inputs)

    def mock_input(_prompt: str) -> str:
        return next(user_inputs_iterator)

    with patch("builtins.input", new=mock_input), patch(
        "autogpt.utils.session.prompt", new=mock_input
    ):
        ai_config = main_menu()

    assert ai_config is not None, "ai_config is None"
    assert ai_config.ai_name == "Chef-GPT"
    assert ai_config.ai_role == "an AI designed to browse bake a cake."
    assert ai_config.ai_goals == [
        "Purchase ingredients",
        "Bake a cake",
    ]


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
def test_prompt_user_manual_mode_r_input(patched_api_requestor: Mock) -> None:
    user_inputs = [
        "--manual",
        "Chef-GPT",
        "an AI designed to browse bake a cake.",
        "2",
        "Purchase ingredients",
        "Bake a cake",
        "",
        "",
        "",
        "3.00",
        "r",
    ]

    user_inputs_iterator = iter(user_inputs)

    def mock_input(_prompt: str) -> str:
        return next(user_inputs_iterator)

    with patch("builtins.input", new=mock_input), patch(
        "autogpt.utils.session.prompt", new=mock_input
    ), patch("autogpt.prompts.prompt.main_menu", return_value=None) as mock_main_menu:
        ai_config = welcome_prompt()
        mock_main_menu.assert_called_once()

    assert ai_config is None


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
def test_prompt_user_manual_mode(patched_api_requestor: Mock) -> None:
    user_inputs = [
        "--manual",
        "Chef-GPT",
        "an AI designed to browse bake a cake.",
        "2",
        "Purchase ingredients",
        "Bake a cake",
        "",
        "",
        "",
        "3.00",
        "n",
    ]

    user_inputs_iterator = iter(user_inputs)

    def mock_prompt_input(_prompt: str) -> str:
        return next(user_inputs_iterator)

    with patch("autogpt.utils.session.prompt", new=mock_prompt_input), patch(
        "builtins.input", new=mock_prompt_input
    ):
        with patch("autogpt.logs.logger.typewriter_log") as mock_logger:
            ai_config = welcome_prompt()
            mock_logger.assert_called_with("Configuration saved.", "\x1b[32m")


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
def test_generate_aiconfig_delete_and_select() -> None:
    """Test delete configuration and select."""

    # Temporary path / file
    temp_config_file = cfg.ai_settings_filepath

    # Temporary config
    config_content = """configs:
    simple-GPT:
        ai_goals:
        -  save a text file test1.txt with the text "hello world"
        -  use list_files to confirm the file test1.txt does exist
        -  delete file test1.txt
        -  list_files the text file to confirm it does not exist
        -  shutdown
        ai_role: do a simple file task
        api_budget: 0.0
    normal-GPT:
        ai_goals:
        -  save a text file test2.txt with the text "hello space"
        -  use read_file to confirm the file test2.txt contains the words "hello space"
        -  rename file test2.txt to test-2.txt
        -  delete file test-2.txt
        -  shutdown
        ai_role: do a normal file task
        api_budget: 0.0
    """

    # Write to the temporary file
    with open(temp_config_file, "w") as temp_file:
        temp_file.write(config_content)

    # Sequence of user inputs:
    user_inputs = ["6", "", "1", "", "1", ""]

    user_inputs_iterator = iter(user_inputs)

    # Function to replace builtins.input and autogpt.utils.session.prompt
    def mock_input(_prompt: str) -> str:
        return next(user_inputs_iterator)

    # Patch functions to use the user_inputs list
    with patch("builtins.input", new=mock_input), patch(
        "autogpt.utils.session.prompt", new=mock_input
    ):
        ai_config = main_menu()

    assert ai_config is not None, "ai_config is None"
    assert ai_config.ai_name == "normal-GPT"
    assert ai_config.ai_role == "do a normal file task"
    assert ai_config.ai_goals == [
        'save a text file test2.txt with the text "hello space"',
        'use read_file to confirm the file test2.txt contains the words "hello space"',
        "rename file test2.txt to test-2.txt",
        "delete file test-2.txt",
        "shutdown",
    ]


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
def test_generate_aiconfig_change_and_select() -> None:
    """Test change configuration and select."""

    # Temporary path / file
    temp_config_file = cfg.ai_settings_filepath

    # Temporary config
    config_content = """configs:
    simple-GPT:
        ai_goals:
        -  save a text file test1.txt with the text "hello world"
        -  shutdown
        ai_role: do a simple file task
        api_budget: 0.0
    normal-GPT:
        ai_goals:
        -  save a text file test2.txt with the text "hello space".
        -  use read_file to confirm the file test2.txt contains the words "hello space".
        -  rename file test2.txt to test-4.txt.
        -  delete file test-2.txt.
        -  shutdown.
        ai_role: do a normal file task
        api_budget: 0.0
    """

    # Write to the temporary file
    with open(temp_config_file, "w") as temp_file:
        temp_file.write(config_content)

    # Sequence of user inputs:
    user_inputs = [
        "4",
        "1",
        "change-GPT",
        "do a changed file task",
        "2",
        "",
        "",
        "",
        "",
        "",
        "",
        "1",
    ]

    user_inputs_iterator = iter(user_inputs)

    # Function to replace builtins.input and autogpt.utils.session.prompt
    def mock_input(_prompt: str) -> str:
        return next(user_inputs_iterator)

    # Patch functions to use the user_inputs list
    with patch("builtins.input", new=mock_input), patch(
        "autogpt.utils.session.prompt", new=mock_input
    ):
        ai_config = main_menu()

    assert ai_config is not None, "ai_config is None"
    assert ai_config.ai_name == "change-GPT"
    assert ai_config.ai_role == "do a changed file task"
    assert ai_config.ai_goals == [
        'save a text file test1.txt with the text "hello world"',
        "shutdown",
    ]


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
def test_generate_aiconfig_change_add_more_goals() -> None:
    """Test change configuration and select."""

    # Temporary path / file
    temp_config_file = cfg.ai_settings_filepath

    # Temporary config
    config_content = """configs:
    simple-GPT:
        ai_goals:
        -  save a text file test1.txt with the text "hello world"
        -  shutdown
        ai_role: do a simple file task
        api_budget: 0.0
    normal-GPT:
        ai_goals:
        -  save a text file test2.txt with the text "hello space".
        -  use read_file to confirm the file test2.txt contains the words "hello space".
        -  rename file test2.txt to test-4.txt.
        -  delete file test-2.txt.
        -  shutdown.
        ai_role: do a normal file task
        api_budget: 0.0
    """

    # Write to the temporary file
    with open(temp_config_file, "w") as temp_file:
        temp_file.write(config_content)

    # Sequence of user inputs:
    user_inputs = [
        "4",
        "1",
        "change-GPT",
        "do a changed file task",
        "4",
        "",
        "",
        "new goal 3",
        "new goal 4",
        "",
        "",
        "",
        "",
        "1",
    ]

    user_inputs_iterator = iter(user_inputs)

    # Function to replace builtins.input and autogpt.utils.session.prompt
    def mock_input(_prompt: str) -> str:
        return next(user_inputs_iterator)

    # Patch functions to use the user_inputs list
    with patch("builtins.input", new=mock_input), patch(
        "autogpt.utils.session.prompt", new=mock_input
    ):
        ai_config = main_menu()

    assert ai_config is not None, "ai_config is None"
    assert ai_config.ai_name == "change-GPT"
    assert ai_config.ai_role == "do a changed file task"
    assert ai_config.ai_goals == [
        'save a text file test1.txt with the text "hello world"',
        "shutdown",
        "new goal 3",
        "new goal 4",
    ]


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
def test_generate_aiconfig_change_and_start() -> None:
    """Test change configuration and select."""

    # Temporary path / file
    temp_config_file = cfg.ai_settings_filepath

    # Temporary config
    config_content = """configs:
    simple-GPT:
        ai_goals:
        -  save a text file test1.txt with the text "hello world"
        -  shutdown
        ai_role: do a simple file task
        api_budget: 0.0
    normal-GPT:
        ai_goals:
        -  save a text file test2.txt with the text "hello space".
        -  use read_file to confirm the file test2.txt contains the words "hello space".
        -  rename file test2.txt to test-4.txt.
        -  delete file test-2.txt.
        -  shutdown.
        ai_role: do a normal file task
        api_budget: 0.0
    """

    # Write to the temporary file
    with open(temp_config_file, "w") as temp_file:
        temp_file.write(config_content)

    # Sequence of user inputs:
    user_inputs = [
        "4",
        "1",
        "change-GPT",
        "do a changed file task",
        "2",
        "",
        "",
        "",
        "",
        "",
        "",
        "r",
        "1",
    ]

    user_inputs_iterator = iter(user_inputs)

    # Function to replace builtins.input and autogpt.utils.session.prompt
    def mock_input(_prompt: str) -> str:
        return next(user_inputs_iterator)

    # Patch functions to use the user_inputs list
    with patch("builtins.input", new=mock_input), patch(
        "autogpt.utils.session.prompt", new=mock_input
    ):
        ai_config = main_menu()

    assert ai_config is not None, "ai_config is None"
    assert ai_config.ai_name == "change-GPT"
    assert ai_config.ai_role == "do a changed file task"
    assert ai_config.ai_goals == [
        'save a text file test1.txt with the text "hello world"',
        "shutdown",
    ]


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
def test_generate_aiconfig_create_and_select() -> None:
    """Test change configuration and select."""

    # Temporary path / file
    temp_config_file = cfg.ai_settings_filepath

    # Temporary config
    config_content = """configs:
    simple-GPT:
        ai_goals:
        -  save a text file test1.txt with the text "hello world"
        -  shutdown
        ai_role: do a simple file task
        api_budget: 0.0
    normal-GPT:
        ai_goals:
        -  save a text file test2.txt with the text "hello space".
        -  use read_file to confirm the file test2.txt contains the words "hello space".
        -  rename file test2.txt to test-4.txt.
        -  delete file test-2.txt.
        -  shutdown.
        ai_role: do a normal file task
        api_budget: 0.0
    """

    # Write to the temporary file
    with open(temp_config_file, "w") as temp_file:
        temp_file.write(config_content)

    # Sequence of user inputs:
    user_inputs = [
        "3",
        "new-GPT",
        "an agent that looks for the newest python code",
        "2",
        "go online and search for new python code",
        "grab it, save it in a file and shutdown",
        "",
        "",
        "",
        "",
        "3",
    ]

    user_inputs_iterator = iter(user_inputs)

    # Function to replace builtins.input and autogpt.utils.session.prompt
    def mock_input(_prompt: str) -> str:
        return next(user_inputs_iterator)

    # Patch functions to use the user_inputs list
    with patch("builtins.input", new=mock_input), patch(
        "autogpt.utils.session.prompt", new=mock_input
    ):
        ai_config = main_menu()

    assert ai_config is not None, "ai_config is None"
    assert ai_config.ai_name == "new-GPT"
    assert ai_config.ai_role == "an agent that looks for the newest python code"
    assert ai_config.ai_goals == [
        "go online and search for new python code",
        "grab it, save it in a file and shutdown",
    ]


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
def test_generate_aiconfig_edit_wrong_budget() -> None:
    """Test change configuration and select."""

    # Temporary path / file
    temp_config_file = cfg.ai_settings_filepath

    # Temporary config
    config_content = """configs:
    simple-GPT:
        ai_goals:
        -  save a text file test1.txt with the text "hello world"
        -  shutdown
        ai_role: do a simple file task
        api_budget: 0.0
    normal-GPT:
        ai_goals:
        -  save a text file test2.txt with the text "hello space".
        -  use read_file to confirm the file test2.txt contains the words "hello space".
        -  rename file test2.txt to test-4.txt.
        -  delete file test-2.txt.
        -  shutdown.
        ai_role: do a normal file task
        api_budget: 0.0
    """

    # Write to the temporary file
    with open(temp_config_file, "w") as temp_file:
        temp_file.write(config_content)

    # Sequence of user inputs:
    user_inputs = [
        "3",
        "new-GPT",
        "an agent that looks for the newest python code",
        "2",
        "go online and search for new python code",
        "grab it, save it in a file and shutdown",
        "",
        "",
        "",
        "a",
        "r",
        "1",
    ]

    user_inputs_iterator = iter(user_inputs)

    # Function to replace builtins.input and autogpt.utils.session.prompt
    def mock_input(_prompt: str) -> str:
        return next(user_inputs_iterator)

    # Patch functions to use the user_inputs list
    with patch("builtins.input", new=mock_input), patch(
        "autogpt.utils.session.prompt", new=mock_input
    ):
        ai_config = main_menu()

    assert ai_config is not None, "ai_config is None"
    assert ai_config.ai_name == "new-GPT"
    assert ai_config.ai_role == "an agent that looks for the newest python code"
    assert ai_config.ai_goals == [
        "go online and search for new python code",
        "grab it, save it in a file and shutdown",
    ]


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
def test_generate_aiconfig_view_config() -> None:
    """Test change configuration and select."""

    # Temporary path / file
    temp_config_file = cfg.ai_settings_filepath

    # Temporary config
    config_content = """configs:
    simple-GPT:
        ai_goals:
        -  save a text file test1.txt with the text "hello world"
        -  shutdown
        ai_role: do a simple file task
        api_budget: 0.0
    normal-GPT:
        ai_goals:
        -  save a text file test2.txt with the text "hello space".
        -  use read_file to confirm the file test2.txt contains the words "hello space".
        -  rename file test2.txt to test-4.txt.
        -  delete file test-2.txt.
        -  shutdown.
        ai_role: do a normal file task
        api_budget: 0.0
    """

    # Write to the temporary file
    with open(temp_config_file, "w") as temp_file:
        temp_file.write(config_content)

    # Sequence of user inputs:
    user_inputs = [
        "5",
        "1",
        "e",
        "",
        "",
        "2",
        "",
        "",
        "",
        "",
        "",
        "0",
        "r",
        "1",
    ]

    user_inputs_iterator = iter(user_inputs)

    # Function to replace builtins.input and autogpt.utils.session.prompt
    def mock_input(_prompt: str) -> str:
        return next(user_inputs_iterator)

    # Patch functions to use the user_inputs list
    with patch("builtins.input", new=mock_input), patch(
        "autogpt.utils.session.prompt", new=mock_input
    ):
        ai_config = main_menu()

    assert ai_config is not None, "ai_config is None"
    assert ai_config.ai_name == "normal-GPT"


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
def test_generate_aiconfig_empty_config() -> None:
    """Test change configuration and select."""

    # Temporary path / file
    temp_config_file = cfg.ai_settings_filepath

    # Temporary config
    config_content = """ """

    # Write to the temporary file
    with open(temp_config_file, "w") as temp_file:
        temp_file.write(config_content)

    # Sequence of user inputs:
    user_inputs = [
        "--manual",
        "chef-GPT",
        "an AI designed to browse bake a cake.",
        "2",
        "Purchase ingredients",
        "Bake a cake",
        "",
        "",
        "",
        "3.00",
        "n",
    ]

    user_inputs_iterator = iter(user_inputs)

    # Function to replace builtins.input and autogpt.utils.session.prompt
    def mock_input(_prompt: str) -> str:
        return next(user_inputs_iterator)

    # Patch functions to use the user_inputs list
    with patch("builtins.input", new=mock_input), patch(
        "autogpt.utils.session.prompt", new=mock_input
    ):
        ai_config = main_menu()

    assert ai_config is not None, "ai_config is None"
    assert ai_config.ai_name == "chef-GPT"


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
def test_generate_aiconfig_edit_num_goals_25() -> None:
    """Test change configuration and select."""

    # Temporary path / file
    temp_config_file = cfg.ai_settings_filepath

    # Temporary config
    config_content = """configs:
    simple-GPT:
        ai_goals:
        -  save a text file test1.txt with the text "hello world"
        -  shutdown
        ai_role: do a simple file task
        api_budget: 0.0
    normal-GPT:
        ai_goals:
        -  save a text file test2.txt with the text "hello space".
        -  use read_file to confirm the file test2.txt contains the words "hello space".
        -  rename file test2.txt to test-4.txt.
        -  delete file test-2.txt.
        -  shutdown.
        ai_role: do a normal file task
        api_budget: 0.0
    """

    # Write to the temporary file
    with open(temp_config_file, "w") as temp_file:
        temp_file.write(config_content)

    # Sequence of user inputs:
    user_inputs = [
        "3",
        "new-GPT",
        "an agent that looks for the newest python code",
        "25",
        "0",
        "go online and search for new python code",
        "grab it, save it in a file and shutdown",
        "",
        "",
        "",
        "",
        "a",
        "r",
        "1",
    ]

    user_inputs_iterator = iter(user_inputs)

    # Function to replace builtins.input and autogpt.utils.session.prompt
    def mock_input(_prompt: str) -> str:
        return next(user_inputs_iterator)

    # Patch functions to use the user_inputs list
    with patch("builtins.input", new=mock_input), patch(
        "autogpt.utils.session.prompt", new=mock_input
    ):
        ai_config = main_menu()

    assert ai_config is not None, "ai_config is None"
    assert ai_config.ai_name == "new-GPT"
    assert ai_config.ai_role == "an agent that looks for the newest python code"
    assert ai_config.ai_goals == [
        "go online and search for new python code",
        "grab it, save it in a file and shutdown",
    ]


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
def test_generate_aiconfig_edit_num_goals_none() -> None:
    """Test change configuration and select."""

    # Temporary path / file
    temp_config_file = cfg.ai_settings_filepath

    # Temporary config
    config_content = """configs:
    simple-GPT:
        ai_goals:
        -  save a text file test1.txt with the text "hello world"
        -  shutdown
        ai_role: do a simple file task
        api_budget: 0.0
    normal-GPT:
        ai_goals:
        -  save a text file test2.txt with the text "hello space".
        -  use read_file to confirm the file test2.txt contains the words "hello space".
        -  rename file test2.txt to test-4.txt.
        -  delete file test-2.txt.
        -  shutdown.
        ai_role: do a normal file task
        api_budget: 0.0
    """

    # Write to the temporary file
    with open(temp_config_file, "w") as temp_file:
        temp_file.write(config_content)

    # Sequence of user inputs:
    user_inputs = [
        "3",
        "new-GPT",
        "an agent that looks for the newest python code",
        "",
        "go online and search for new python code",
        "grab it, save it in a file and shutdown",
        "",
        "",
        "",
        "",
        "a",
        "r",
        "1",
    ]

    user_inputs_iterator = iter(user_inputs)

    # Function to replace builtins.input and autogpt.utils.session.prompt
    def mock_input(_prompt: str) -> str:
        return next(user_inputs_iterator)

    # Patch functions to use the user_inputs list
    with patch("builtins.input", new=mock_input), patch(
        "autogpt.utils.session.prompt", new=mock_input
    ):
        ai_config = main_menu()

    assert ai_config is not None, "ai_config is None"
    assert ai_config.ai_name == "new-GPT"
    assert ai_config.ai_role == "an agent that looks for the newest python code"
    assert ai_config.ai_goals == [
        "go online and search for new python code",
        "grab it, save it in a file and shutdown",
    ]


@pytest.mark.vcr
@requires_api_key("OPENAI_API_KEY")
def test_generate_aiconfig_edit_plugins() -> None:
    """Test change configuration and select."""

    # Temporary path / file
    temp_config_file = cfg.ai_settings_filepath

    # Temporary config
    config_content = """configs:
    simple-GPT:
        ai_goals:
        -  save a text file test1.txt with the text "hello world"
        -  shutdown
        ai_role: do a simple file task
        api_budget: 0.0
        plugins:
        - plugin1
        - plugin3
    normal-GPT:
        ai_goals:
        -  save a text file test2.txt with the text "hello space".
        -  use read_file to confirm the file test2.txt contains the words "hello space".
        -  rename file test2.txt to test-4.txt.
        -  delete file test-2.txt.
        -  shutdown.
        ai_role: do a normal file task
        api_budget: 0.0
    """

    # Write to the temporary file
    with open(temp_config_file, "w") as temp_file:
        temp_file.write(config_content)

    # Sequence of user inputs:
    user_inputs = [
        "4",
        "1",
        "plugin-GPT",
        "",
        "2",
        "",
        "",
        "d",
        "k",
        "a",
        "10.00",
        "r",
        "2",
    ]

    user_inputs_iterator = iter(user_inputs)

    # Function to replace builtins.input and autogpt.utils.session.prompt
    def mock_input(_prompt: str) -> str:
        return next(user_inputs_iterator)

    # Patch functions to use the user_inputs list
    with patch("builtins.input", new=mock_input), patch(
        "autogpt.utils.session.prompt", new=mock_input
    ):
        ai_config = main_menu()

    assert ai_config is not None, "ai_config is None"
    assert ai_config.ai_name == "plugin-GPT"
    assert ai_config.ai_role == "do a simple file task"
    assert ai_config.api_budget == 10.0
    assert ai_config.plugins == [
        "plugin3",
        "plugin2",
    ]
