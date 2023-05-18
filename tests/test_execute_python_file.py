import os
from unittest import mock

import pytest

from autogpt.commands.execute_code import execute_python_file

@mock.patch("subprocess.run")
def test_execute_python_file_with_valid_file(mock_run):
    # Test executing a Python file with a valid file
    filename = "tests/scripts/example.py"

    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = "Hello, world!"

    result = execute_python_file(filename)

    assert result == "Hello, world!"

    os.remove(filename)


@mock.patch("subprocess.run")
def test_execute_python_file_with_valid_file_and_args(mock_run):
    # Test executing a Python file with a valid file and args
    filename = "tests/scripts/exampleargs.py"
    args = "Hello world"

    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = "Hello world"

    result = execute_python_file(filename, args)

    assert result == "Hello world"

    os.remove(filename)


def test_execute_python_file_with_invalid_file():
    # Test executing a Python file with an invalid file
    filename = "tests/scripts/invalid.txt"
    
    result = execute_python_file(filename)

    assert result == "Error: Invalid file type. Only .py files are allowed."


def test_execute_python_file_with_nonexistent_file():
    # Test executing a Python file with a nonexistent file
    filename = "nonexistent.py"

    result = execute_python_file(filename)

    assert result == f"Error: File '{filename}' does not exist."
