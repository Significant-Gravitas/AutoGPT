import os
from unittest.mock import Mock, patch

import pytest
import requests
from colorama import Fore
from git import Repo

from autogpt.utils import (
    clean_input,
    get_bulletin_from_web,
    get_current_git_branch,
    get_latest_bulletin,
    readable_file_size,
    remove_color_codes,
    validate_yaml_file,
)
from tests.utils import skip_in_ci


def test_validate_yaml_file_valid():
    with open("valid_test_file.yaml", "w") as f:
        f.write("setting: value")
    result, message = validate_yaml_file("valid_test_file.yaml")
    os.remove("valid_test_file.yaml")

    assert result == True
    assert "Successfully validated" in message


def test_validate_yaml_file_not_found():
    result, message = validate_yaml_file("non_existent_file.yaml")

    assert result == False
    assert "wasn't found" in message


def test_validate_yaml_file_invalid():
    with open("invalid_test_file.yaml", "w") as f:
        f.write(
            "settings:\n  first_setting: value\n  second_setting: value\n    nested_setting: value\n  third_setting: value\nunindented_setting: value"
        )
    result, message = validate_yaml_file("invalid_test_file.yaml")
    os.remove("invalid_test_file.yaml")
    print(result)
    print(message)
    assert result == False
    assert "There was an issue while trying to read" in message


def test_readable_file_size():
    size_in_bytes = 1024 * 1024 * 3.5  # 3.5 MB
    readable_size = readable_file_size(size_in_bytes)

    assert readable_size == "3.50 MB"


@patch("requests.get")
def test_get_bulletin_from_web_success(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.text = "Test bulletin"
    bulletin = get_bulletin_from_web()

    assert bulletin == "Test bulletin"


@patch("requests.get")
def test_get_bulletin_from_web_failure(mock_get):
    mock_get.return_value.status_code = 404
    bulletin = get_bulletin_from_web()
    print(bulletin)
    assert bulletin == ""


@skip_in_ci
def test_get_current_git_branch():
    branch_name = get_current_git_branch()

    # Assuming that the branch name will be non-empty if the function is working correctly.
    assert branch_name != ""


def test_get_latest_bulletin_no_file():
    if os.path.exists("CURRENT_BULLETIN.md"):
        os.remove("CURRENT_BULLETIN.md")

    with patch("autogpt.utils.get_bulletin_from_web", return_value=""):
        bulletin = get_latest_bulletin()
        assert bulletin == ""


def test_get_latest_bulletin_with_file():
    with open("CURRENT_BULLETIN.md", "w", encoding="utf-8") as f:
        f.write("Test bulletin")

    with patch("autogpt.utils.get_bulletin_from_web", return_value=""):
        bulletin = get_latest_bulletin()
        assert bulletin == "Test bulletin"

    os.remove("CURRENT_BULLETIN.md")


def test_get_latest_bulletin_with_new_bulletin():
    with open("CURRENT_BULLETIN.md", "w", encoding="utf-8") as f:
        f.write("Old bulletin")

    with patch("autogpt.utils.get_bulletin_from_web", return_value="New bulletin"):
        bulletin = get_latest_bulletin()
        assert "New bulletin" in bulletin

    os.remove("CURRENT_BULLETIN.md")


@patch("requests.get")
def test_get_bulletin_from_web_success(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.text = "Test bulletin"
    bulletin = get_bulletin_from_web()

    assert bulletin == "Test bulletin"
    mock_get.assert_called_with(
        "https://raw.githubusercontent.com/Significant-Gravitas/Auto-GPT/master/BULLETIN.md"
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


@patch("autogpt.utils.Repo")
def test_get_current_git_branch_success(mock_repo):
    mock_repo.return_value.active_branch.name = "test-branch"
    branch_name = get_current_git_branch()

    assert branch_name == "test-branch"


@patch("autogpt.utils.Repo")
def test_get_current_git_branch_failure(mock_repo):
    mock_repo.side_effect = Exception()
    branch_name = get_current_git_branch()

    assert branch_name == ""


def test_get_latest_bulletin_no_file():
    if os.path.exists("CURRENT_BULLETIN.md"):
        os.remove("CURRENT_BULLETIN.md")

    with patch("autogpt.utils.get_bulletin_from_web", return_value=""):
        bulletin = get_latest_bulletin()
        assert bulletin == ""


def test_get_latest_bulletin_with_file():
    with open("CURRENT_BULLETIN.md", "w", encoding="utf-8") as f:
        f.write("Test bulletin")

    with patch("autogpt.utils.get_bulletin_from_web", return_value=""):
        bulletin = get_latest_bulletin()
        assert bulletin == "Test bulletin"

    os.remove("CURRENT_BULLETIN.md")


def test_get_latest_bulletin_with_new_bulletin():
    with open("CURRENT_BULLETIN.md", "w", encoding="utf-8") as f:
        f.write("Old bulletin")

    with patch("autogpt.utils.get_bulletin_from_web", return_value="New bulletin"):
        bulletin = get_latest_bulletin()
        assert f" {Fore.RED}::UPDATED:: {Fore.CYAN}New bulletin{Fore.RESET}" in bulletin

    os.remove("CURRENT_BULLETIN.md")


def test_get_latest_bulletin_new_bulletin_same_as_old_bulletin():
    with open("CURRENT_BULLETIN.md", "w", encoding="utf-8") as f:
        f.write("Test bulletin")

    with patch("autogpt.utils.get_bulletin_from_web", return_value="Test bulletin"):
        bulletin = get_latest_bulletin()
        assert bulletin == "Test bulletin"

    os.remove("CURRENT_BULLETIN.md")


def test_remove_color_codes():
    assert (
        remove_color_codes(
            "COMMAND = \x1b[36mbrowse_website\x1b[0m  ARGUMENTS = \x1b[36m{'url': 'https://www.google.com', "
            "'question': 'What is the capital of France?'}\x1b[0m"
        )
        == "COMMAND = browse_website  ARGUMENTS = {'url': 'https://www.google.com', 'question': 'What is the capital "
        "of France?'}"
    )
    assert (
        remove_color_codes(
            "{'Schaue dir meine Projekte auf github () an, als auch meine Webseiten': "
            "'https://github.com/Significant-Gravitas/Auto-GPT, https://discord.gg/autogpt und "
            "https://twitter.com/SigGravitas'}"
        )
        == "{'Schaue dir meine Projekte auf github () an, als auch meine Webseiten': "
        "'https://github.com/Significant-Gravitas/Auto-GPT, https://discord.gg/autogpt und "
        "https://twitter.com/SigGravitas'}"
    )
    assert remove_color_codes("") == ""
    assert remove_color_codes("hello") == "hello"
    assert remove_color_codes("hello\x1B[31m world") == "hello world"
    assert remove_color_codes("\x1B[36mHello,\x1B[32m World!") == "Hello, World!"
    assert (
        remove_color_codes("\x1B[1m\x1B[31mError:\x1B[0m\x1B[31m file not found")
        == "Error: file not found"
    )
    assert remove_color_codes({"I": "am a dict"}) == f"{{'I': 'am a dict'}}"


if __name__ == "__main__":
    pytest.main()
