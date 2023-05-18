import os
import shutil
import sys
from pathlib import Path

import pytest

from autogpt.commands.command import Command, CommandRegistry


class TestCommand:
    """Test cases for the Command class."""

    @staticmethod
    def example_command_method(arg1: int, arg2: str) -> str:
        """Example function for testing the Command class."""
        # This function is static because it is not used by any other test cases.
        return f"{arg1} - {arg2}"

    def test_command_creation(self):
        """Test that a Command object can be created with the correct attributes."""
        cmd = Command(
            name="example",
            description="Example command",
            method=self.example_command_method,
        )

        assert cmd.name == "example"
        assert cmd.description == "Example command"
        assert cmd.method == self.example_command_method
        assert cmd.signature == "(arg1: int, arg2: str) -> str"

    def test_command_call(self):
        """Test that Command(*args) calls and returns the result of method(*args)."""
        # Create a Command object with the example_command_method.
        cmd = Command(
            name="example",
            description="Example command",
            method=self.example_command_method,
        )
        result = cmd(arg1=1, arg2="test")
        assert result == "1 - test"

    def test_command_call_with_invalid_arguments(self):
        """Test that calling a Command object with invalid arguments raises a TypeError."""
        cmd = Command(
            name="example",
            description="Example command",
            method=self.example_command_method,
        )
        with pytest.raises(TypeError):
            cmd(arg1="invalid", does_not_exist="test")

    def test_command_default_signature(self):
        """Test that the default signature is generated correctly."""
        cmd = Command(
            name="example",
            description="Example command",
            method=self.example_command_method,
        )
        assert cmd.signature == "(arg1: int, arg2: str) -> str"

    def test_command_custom_signature(self):
        custom_signature = "custom_arg1: int, custom_arg2: str"
        cmd = Command(
            name="example",
            description="Example command",
            method=self.example_command_method,
            signature=custom_signature,
        )

        assert cmd.signature == custom_signature


class TestCommandRegistry:
    @staticmethod
    def example_command_method(arg1: int, arg2: str) -> str:
        return f"{arg1} - {arg2}"

    def test_register_command(self):
        """Test that a command can be registered to the registry."""
        registry = CommandRegistry()
        cmd = Command(
            name="example",
            description="Example command",
            method=self.example_command_method,
        )

        registry.register(cmd)

        assert cmd.name in registry.commands
        assert registry.commands[cmd.name] == cmd

    def test_unregister_command(self):
        """Test that a command can be unregistered from the registry."""
        registry = CommandRegistry()
        cmd = Command(
            name="example",
            description="Example command",
            method=self.example_command_method,
        )

        registry.register(cmd)
        registry.unregister(cmd.name)

        assert cmd.name not in registry.commands

    def test_get_command(self):
        """Test that a command can be retrieved from the registry."""
        registry = CommandRegistry()
        cmd = Command(
            name="example",
            description="Example command",
            method=self.example_command_method,
        )

        registry.register(cmd)
        retrieved_cmd = registry.get_command(cmd.name)

        assert retrieved_cmd == cmd

    def test_get_nonexistent_command(self):
        """Test that attempting to get a nonexistent command raises a KeyError."""
        registry = CommandRegistry()

        with pytest.raises(KeyError):
            registry.get_command("nonexistent_command")

    def test_call_command(self):
        """Test that a command can be called through the registry."""
        registry = CommandRegistry()
        cmd = Command(
            name="example",
            description="Example command",
            method=self.example_command_method,
        )

        registry.register(cmd)
        result = registry.call("example", arg1=1, arg2="test")

        assert result == "1 - test"

    def test_call_nonexistent_command(self):
        """Test that attempting to call a nonexistent command raises a KeyError."""
        registry = CommandRegistry()

        with pytest.raises(KeyError):
            registry.call("nonexistent_command", arg1=1, arg2="test")

    def test_get_command_prompt(self):
        """Test that the command prompt is correctly formatted."""
        registry = CommandRegistry()
        cmd = Command(
            name="example",
            description="Example command",
            method=self.example_command_method,
        )

        registry.register(cmd)
        command_prompt = registry.command_prompt()

        assert f"(arg1: int, arg2: str)" in command_prompt

    def test_import_mock_commands_module(self):
        """Test that the registry can import a module with mock command plugins."""
        registry = CommandRegistry()
        mock_commands_module = "tests.mocks.mock_commands"

        registry.import_commands(mock_commands_module)

        assert "function_based" in registry.commands
        assert registry.commands["function_based"].name == "function_based"
        assert (
            registry.commands["function_based"].description
            == "Function-based test command"
        )

    def test_import_temp_command_file_module(self, tmp_path):
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
            registry.commands["function_based"].description
            == "Function-based test command"
        )
