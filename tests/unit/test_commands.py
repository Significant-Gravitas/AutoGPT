import os
import shutil
import sys
from pathlib import Path

import pytest

from autogpt.models.command import Command, CommandParameter
from autogpt.models.command_registry import CommandRegistry

PARAMETERS = [
    CommandParameter("arg1", "int", description="Argument 1", required=True),
    CommandParameter("arg2", "str", description="Argument 2", required=False),
]


def example_command_method(arg1: int, arg2: str) -> str:
    """Example function for testing the Command class."""
    # This function is static because it is not used by any other test cases.
    return f"{arg1} - {arg2}"


def test_command_creation():
    """Test that a Command object can be created with the correct attributes."""
    cmd = Command(
        name="example",
        description="Example command",
        method=example_command_method,
        parameters=PARAMETERS,
    )

    assert cmd.name == "example"
    assert cmd.description == "Example command"
    assert cmd.method == example_command_method
    assert (
        str(cmd) == "example: Example command, params: (arg1: int, arg2: Optional[str])"
    )


@pytest.fixture
def example_command():
    yield Command(
        name="example",
        description="Example command",
        method=example_command_method,
        parameters=PARAMETERS,
    )


def test_command_call(example_command: Command):
    """Test that Command(*args) calls and returns the result of method(*args)."""
    result = example_command(arg1=1, arg2="test")
    assert result == "1 - test"


def test_command_call_with_invalid_arguments(example_command: Command):
    """Test that calling a Command object with invalid arguments raises a TypeError."""
    with pytest.raises(TypeError):
        example_command(arg1="invalid", does_not_exist="test")


def test_register_command(example_command: Command):
    """Test that a command can be registered to the registry."""
    registry = CommandRegistry()

    registry.register(example_command)

    assert registry.get_command(example_command.name) == example_command
    assert len(registry.commands) == 1


def test_unregister_command(example_command: Command):
    """Test that a command can be unregistered from the registry."""
    registry = CommandRegistry()

    registry.register(example_command)
    registry.unregister(example_command)

    assert len(registry.commands) == 0
    assert example_command.name not in registry


@pytest.fixture
def example_command_with_aliases(example_command: Command):
    example_command.aliases = ["example_alias", "example_alias_2"]
    return example_command


def test_register_command_aliases(example_command_with_aliases: Command):
    """Test that a command can be registered to the registry."""
    registry = CommandRegistry()
    command = example_command_with_aliases

    registry.register(command)

    assert command.name in registry
    assert registry.get_command(command.name) == command
    for alias in command.aliases:
        assert registry.get_command(alias) == command
    assert len(registry.commands) == 1


def test_unregister_command_aliases(example_command_with_aliases: Command):
    """Test that a command can be unregistered from the registry."""
    registry = CommandRegistry()
    command = example_command_with_aliases

    registry.register(command)
    registry.unregister(command)

    assert len(registry.commands) == 0
    assert command.name not in registry
    for alias in command.aliases:
        assert alias not in registry


def test_command_in_registry(example_command_with_aliases: Command):
    """Test that `command_name in registry` works."""
    registry = CommandRegistry()
    command = example_command_with_aliases

    assert command.name not in registry
    assert "nonexistent_command" not in registry

    registry.register(command)

    assert command.name in registry
    assert "nonexistent_command" not in registry
    for alias in command.aliases:
        assert alias in registry


def test_get_command(example_command: Command):
    """Test that a command can be retrieved from the registry."""
    registry = CommandRegistry()

    registry.register(example_command)
    retrieved_cmd = registry.get_command(example_command.name)

    assert retrieved_cmd == example_command


def test_get_nonexistent_command():
    """Test that attempting to get a nonexistent command raises a KeyError."""
    registry = CommandRegistry()

    assert registry.get_command("nonexistent_command") is None
    assert "nonexistent_command" not in registry


def test_call_command():
    """Test that a command can be called through the registry."""
    registry = CommandRegistry()
    cmd = Command(
        name="example",
        description="Example command",
        method=example_command_method,
        parameters=PARAMETERS,
    )

    registry.register(cmd)
    result = registry.call("example", arg1=1, arg2="test")

    assert result == "1 - test"


def test_call_nonexistent_command():
    """Test that attempting to call a nonexistent command raises a KeyError."""
    registry = CommandRegistry()

    with pytest.raises(KeyError):
        registry.call("nonexistent_command", arg1=1, arg2="test")


def test_get_command_prompt():
    """Test that the command prompt is correctly formatted."""
    registry = CommandRegistry()
    cmd = Command(
        name="example",
        description="Example command",
        method=example_command_method,
        parameters=PARAMETERS,
    )

    registry.register(cmd)
    command_prompt = registry.command_prompt()

    assert f"(arg1: int, arg2: Optional[str])" in command_prompt


def test_import_mock_commands_module():
    """Test that the registry can import a module with mock command plugins."""
    registry = CommandRegistry()
    mock_commands_module = "tests.mocks.mock_commands"

    registry.import_commands(mock_commands_module)

    assert "function_based" in registry
    assert registry.commands["function_based"].name == "function_based"
    assert (
        registry.commands["function_based"].description == "Function-based test command"
    )


def test_import_temp_command_file_module(tmp_path: Path):
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

    assert "function_based" in registry
    assert registry.commands["function_based"].name == "function_based"
    assert (
        registry.commands["function_based"].description == "Function-based test command"
    )
