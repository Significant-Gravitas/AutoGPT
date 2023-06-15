import os
import shutil
import sys
from pathlib import Path
from typing import Callable

import pytest

from autogpt.models.command import Command
from autogpt.models.command_argument import CommandArgument
from autogpt.models.command_registry import CommandRegistry


@pytest.fixture
def default_arguments():
    return [
        CommandArgument(name="arg1", description="arg 1", type="int", required=True),
        CommandArgument(name="arg2", description="arg 2", type="str", required=True),
    ]


@pytest.fixture
def example_command_method():
    def command_method(arg1: int, arg2: str) -> str:
        """Example function for testing the Command class."""
        # This function is static because it is not used by any other test cases.
        return f"{arg1} - {arg2}"

    return command_method


def test_command_creation(example_command_method: Callable, default_arguments):
    """Test that a Command object can be created with the correct attributes."""
    cmd = Command(
        name="example",
        description="Example command",
        method=example_command_method,
        arguments=default_arguments,
    )

    assert cmd.name == "example"
    assert cmd.description == "Example command"
    assert cmd.method == example_command_method

    assert len(cmd.arguments) == 2
    assert cmd.arguments[0].name == "arg1"
    assert cmd.arguments[1].name == "arg2"


def test_command_call(example_command_method: Callable, default_arguments: dict):
    """Test that Command(*args) calls and returns the result of method(*args)."""
    # Create a Command object with the example_command_method.
    cmd = Command(
        name="example",
        description="Example command",
        method=example_command_method,
        arguments=default_arguments,
    )
    result = cmd(arg1=1, arg2="test")
    assert result == "1 - test"


def test_command_call_with_invalid_arguments(
    example_command_method: Callable, default_arguments: dict
):
    """Test that calling a Command object with invalid arguments raises a TypeError."""
    cmd = Command(
        name="example",
        description="Example command",
        method=example_command_method,
        arguments=default_arguments,
    )
    with pytest.raises(TypeError):
        cmd(arg1="invalid", does_not_exist="test")


def test_command_custom_arguments(
    example_command_method: Callable, default_arguments: dict
):
    custom_arguments = "custom_arg1: int, custom_arg2: str"
    cmd = Command(
        name="example",
        description="Example command",
        method=example_command_method,
        arguments=custom_arguments,
    )

    assert cmd.arguments == custom_arguments


def test_register_command(example_command_method: Callable, default_arguments: dict):
    """Test that a command can be registered to the registry."""
    registry = CommandRegistry()
    cmd = Command(
        name="example",
        description="Example command",
        method=example_command_method,
        arguments=default_arguments,
    )

    registry.register(cmd)

    assert cmd.name in registry.commands
    assert registry.commands[cmd.name] == cmd


def test_unregister_command(example_command_method: Callable, default_arguments: dict):
    """Test that a command can be unregistered from the registry."""
    registry = CommandRegistry()
    cmd = Command(
        name="example",
        description="Example command",
        method=example_command_method,
        arguments=default_arguments,
    )

    registry.register(cmd)
    registry.unregister(cmd.name)

    assert cmd.name not in registry.commands


def test_get_command(example_command_method: Callable, default_arguments: dict):
    """Test that a command can be retrieved from the registry."""
    registry = CommandRegistry()
    cmd = Command(
        name="example",
        description="Example command",
        method=example_command_method,
        arguments=default_arguments,
    )

    registry.register(cmd)
    retrieved_cmd = registry.get_command(cmd.name)

    assert retrieved_cmd == cmd


def test_get_nonexistent_command(
    example_command_method: Callable, default_arguments: dict
):
    """Test that attempting to get a nonexistent command raises a KeyError."""
    registry = CommandRegistry()

    with pytest.raises(KeyError):
        registry.get_command("nonexistent_command")


def test_call_command(example_command_method: Callable, default_arguments: dict):
    """Test that a command can be called through the registry."""
    registry = CommandRegistry()
    cmd = Command(
        name="example",
        description="Example command",
        method=example_command_method,
        arguments=default_arguments,
    )

    registry.register(cmd)
    result = registry.call("example", arg1=1, arg2="test")

    assert result == "1 - test"


def test_call_nonexistent_command(
    example_command_method: Callable, default_arguments: dict
):
    """Test that attempting to call a nonexistent command raises a KeyError."""
    registry = CommandRegistry()

    with pytest.raises(KeyError):
        registry.call("nonexistent_command", arg1=1, arg2="test")


def test_import_mock_commands_module(
    example_command_method: Callable, default_arguments: dict
):
    """Test that the registry can import a module with mock command plugins."""
    registry = CommandRegistry()
    mock_commands_module = "tests.mocks.mock_commands"

    registry.import_commands(mock_commands_module)

    assert "function_based" in registry.commands
    assert registry.commands["function_based"].name == "function_based"
    assert (
        registry.commands["function_based"].description == "Function-based test command"
    )


def test_import_temp_command_file_module(tmp_path):
    """
    Test that the registry can import a command plugins module from a temp file.
    Args:
        tmp_path (pathlib.Path): Path to a temporary directory.
    """
    registry = CommandRegistry()

    # Create a temp command file
    src = Path(os.getcwd()) / "tests/mocks/mock_commands.py"
    temp_commands_file = tmp_path / "mock_commands.py"
    shutil.copyfile(src, temp_commands_file)

    # Add the temp directory to sys.path to make the module importable
    sys.path.append(str(tmp_path))

    temp_commands_module = "mock_commands"
    registry.import_commands(temp_commands_module)

    # Remove the temp directory from sys.path
    sys.path.remove(str(tmp_path))

    assert "function_based" in registry.commands
    assert registry.commands["function_based"].name == "function_based"
    assert (
        registry.commands["function_based"].description == "Function-based test command"
    )
