import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import requests
from forge.json.parsing import extract_dict_from_json
from forge.utils.yaml_validator import validate_yaml_file
from git import InvalidGitRepositoryError

import autogpt.app.utils
from autogpt.app.utils import (
    get_bulletin_from_web,
    get_current_git_branch,
    get_latest_bulletin,
    set_env_config_value,
)
from tests.utils import skip_in_ci


@pytest.fixture
def valid_json_response() -> dict:
    return {
        "thoughts": {
            "text": "My task is complete. I will use the 'task_complete' command "
            "to shut down.",
            "reasoning": "I will use the 'task_complete' command because it allows me "
            "to shut down and signal that my task is complete.",
            "plan": "I will use the 'task_complete' command with the reason "
            "'Task complete: retrieved Tesla's revenue in 2022.' to shut down.",
            "criticism": "I need to ensure that I have completed all necessary tasks "
            "before shutting down.",
            "speak": "All done!",
        },
        "command": {
            "name": "task_complete",
            "args": {"reason": "Task complete: retrieved Tesla's revenue in 2022."},
        },
    }


@pytest.fixture
def invalid_json_response() -> dict:
    return {
        "thoughts": {
            "text": "My task is complete. I will use the 'task_complete' command "
            "to shut down.",
            "reasoning": "I will use the 'task_complete' command because it allows me "
            "to shut down and signal that my task is complete.",
            "plan": "I will use the 'task_complete' command with the reason "
            "'Task complete: retrieved Tesla's revenue in 2022.' to shut down.",
            "criticism": "I need to ensure that I have completed all necessary tasks "
            "before shutting down.",
            "speak": "",
        },
        "command": {"name": "", "args": {}},
    }


def test_validate_yaml_file_valid():
    with open("valid_test_file.yaml", "w") as f:
        f.write("setting: value")
    result, message = validate_yaml_file("valid_test_file.yaml")
    os.remove("valid_test_file.yaml")

    assert result is True
    assert "Successfully validated" in message


def test_validate_yaml_file_not_found():
    result, message = validate_yaml_file("non_existent_file.yaml")

    assert result is False
    assert "wasn't found" in message


def test_validate_yaml_file_invalid():
    with open("invalid_test_file.yaml", "w") as f:
        f.write(
            "settings:\n"
            "  first_setting: value\n"
            "  second_setting: value\n"
            "    nested_setting: value\n"
            "  third_setting: value\n"
            "unindented_setting: value"
        )
    result, message = validate_yaml_file("invalid_test_file.yaml")
    os.remove("invalid_test_file.yaml")
    print(result)
    print(message)
    assert result is False
    assert "There was an issue while trying to read" in message


@patch("requests.get")
def test_get_bulletin_from_web_success(mock_get):
    expected_content = "Test bulletin from web"

    mock_get.return_value.status_code = 200
    mock_get.return_value.text = expected_content
    bulletin = get_bulletin_from_web()

    assert expected_content in bulletin
    mock_get.assert_called_with(
        "https://raw.githubusercontent.com/Significant-Gravitas/AutoGPT/master/autogpts/autogpt/BULLETIN.md"  # noqa: E501
    )


@patch("requests.get")
def test_get_bulletin_from_web_failure(mock_get):
    mock_get.return_value.status_code = 404
    bulletin = get_bulletin_from_web()

    assert bulletin == ""


@patch("requests.get")
def test_get_bulletin_from_web_exception(mock_get):
    mock_get.side_effect = requests.exceptions.RequestException()
    bulletin = get_bulletin_from_web()

    assert bulletin == ""


def test_get_latest_bulletin_no_file():
    if os.path.exists("data/CURRENT_BULLETIN.md"):
        os.remove("data/CURRENT_BULLETIN.md")

    bulletin, is_new = get_latest_bulletin()
    assert is_new


def test_get_latest_bulletin_with_file():
    expected_content = "Test bulletin"
    with open("data/CURRENT_BULLETIN.md", "w", encoding="utf-8") as f:
        f.write(expected_content)

    with patch("autogpt.app.utils.get_bulletin_from_web", return_value=""):
        bulletin, is_new = get_latest_bulletin()
        assert expected_content in bulletin
        assert is_new is False

    os.remove("data/CURRENT_BULLETIN.md")


def test_get_latest_bulletin_with_new_bulletin():
    with open("data/CURRENT_BULLETIN.md", "w", encoding="utf-8") as f:
        f.write("Old bulletin")

    expected_content = "New bulletin from web"
    with patch(
        "autogpt.app.utils.get_bulletin_from_web", return_value=expected_content
    ):
        bulletin, is_new = get_latest_bulletin()
        assert "::NEW BULLETIN::" in bulletin
        assert expected_content in bulletin
        assert is_new

    os.remove("data/CURRENT_BULLETIN.md")


def test_get_latest_bulletin_new_bulletin_same_as_old_bulletin():
    expected_content = "Current bulletin"
    with open("data/CURRENT_BULLETIN.md", "w", encoding="utf-8") as f:
        f.write(expected_content)

    with patch(
        "autogpt.app.utils.get_bulletin_from_web", return_value=expected_content
    ):
        bulletin, is_new = get_latest_bulletin()
        assert expected_content in bulletin
        assert is_new is False

    os.remove("data/CURRENT_BULLETIN.md")


@skip_in_ci
def test_get_current_git_branch():
    branch_name = get_current_git_branch()
    assert branch_name != ""


@patch("autogpt.app.utils.Repo")
def test_get_current_git_branch_success(mock_repo):
    mock_repo.return_value.active_branch.name = "test-branch"
    branch_name = get_current_git_branch()

    assert branch_name == "test-branch"


@patch("autogpt.app.utils.Repo")
def test_get_current_git_branch_failure(mock_repo):
    mock_repo.side_effect = InvalidGitRepositoryError()
    branch_name = get_current_git_branch()

    assert branch_name == ""


def test_extract_json_from_response(valid_json_response: dict):
    emulated_response_from_openai = json.dumps(valid_json_response)
    assert extract_dict_from_json(emulated_response_from_openai) == valid_json_response


def test_extract_json_from_response_wrapped_in_code_block(valid_json_response: dict):
    emulated_response_from_openai = "```" + json.dumps(valid_json_response) + "```"
    assert extract_dict_from_json(emulated_response_from_openai) == valid_json_response


def test_extract_json_from_response_wrapped_in_code_block_with_language(
    valid_json_response: dict,
):
    emulated_response_from_openai = "```json" + json.dumps(valid_json_response) + "```"
    assert extract_dict_from_json(emulated_response_from_openai) == valid_json_response


def test_extract_json_from_response_json_contained_in_string(valid_json_response: dict):
    emulated_response_from_openai = (
        "sentence1" + json.dumps(valid_json_response) + "sentence2"
    )
    assert extract_dict_from_json(emulated_response_from_openai) == valid_json_response


@pytest.fixture
def mock_env_file_path(tmp_path):
    return tmp_path / ".env"


env_file_initial_content = """
# This is a comment
EXISTING_KEY=EXISTING_VALUE

## This is also a comment
# DISABLED_KEY=DISABLED_VALUE

# Another comment
UNUSED_KEY=UNUSED_VALUE
"""


@pytest.fixture
def mock_env_file(mock_env_file_path: Path, monkeypatch: pytest.MonkeyPatch):
    mock_env_file_path.write_text(env_file_initial_content)
    monkeypatch.setattr(autogpt.app.utils, "ENV_FILE_PATH", mock_env_file_path)
    return mock_env_file_path


@pytest.fixture
def mock_environ(monkeypatch: pytest.MonkeyPatch):
    env = {}
    monkeypatch.setattr(os, "environ", env)
    return env


def test_set_env_config_value_updates_existing_key(
    mock_env_file: Path, mock_environ: dict
):
    # Before updating, ensure the original content is as expected
    with mock_env_file.open("r") as file:
        assert file.readlines() == env_file_initial_content.splitlines(True)

    set_env_config_value("EXISTING_KEY", "NEW_VALUE")
    with mock_env_file.open("r") as file:
        content = file.readlines()

    # Ensure only the relevant line is altered
    expected_content_lines = [
        "\n",
        "# This is a comment\n",
        "EXISTING_KEY=NEW_VALUE\n",  # existing key + new value
        "\n",
        "## This is also a comment\n",
        "# DISABLED_KEY=DISABLED_VALUE\n",
        "\n",
        "# Another comment\n",
        "UNUSED_KEY=UNUSED_VALUE\n",
    ]
    assert content == expected_content_lines
    assert mock_environ["EXISTING_KEY"] == "NEW_VALUE"


def test_set_env_config_value_uncomments_and_updates_disabled_key(
    mock_env_file: Path, mock_environ: dict
):
    # Before adding, ensure the original content is as expected
    with mock_env_file.open("r") as file:
        assert file.readlines() == env_file_initial_content.splitlines(True)

    set_env_config_value("DISABLED_KEY", "ENABLED_NEW_VALUE")
    with mock_env_file.open("r") as file:
        content = file.readlines()

    # Ensure only the relevant line is altered
    expected_content_lines = [
        "\n",
        "# This is a comment\n",
        "EXISTING_KEY=EXISTING_VALUE\n",
        "\n",
        "## This is also a comment\n",
        "DISABLED_KEY=ENABLED_NEW_VALUE\n",  # disabled -> enabled + new value
        "\n",
        "# Another comment\n",
        "UNUSED_KEY=UNUSED_VALUE\n",
    ]
    assert content == expected_content_lines
    assert mock_environ["DISABLED_KEY"] == "ENABLED_NEW_VALUE"


def test_set_env_config_value_adds_new_key(mock_env_file: Path, mock_environ: dict):
    # Before adding, ensure the original content is as expected
    with mock_env_file.open("r") as file:
        assert file.readlines() == env_file_initial_content.splitlines(True)

    set_env_config_value("NEW_KEY", "NEW_VALUE")
    with mock_env_file.open("r") as file:
        content = file.readlines()

    # Ensure the new key-value pair is added without altering the rest
    expected_content_lines = [
        "\n",
        "# This is a comment\n",
        "EXISTING_KEY=EXISTING_VALUE\n",
        "\n",
        "## This is also a comment\n",
        "# DISABLED_KEY=DISABLED_VALUE\n",
        "\n",
        "# Another comment\n",
        "UNUSED_KEY=UNUSED_VALUE\n",
        "NEW_KEY=NEW_VALUE\n",  # New key-value pair added at the end
    ]
    assert content == expected_content_lines
    assert mock_environ["NEW_KEY"] == "NEW_VALUE"
