import shutil
from pathlib import Path

import pytest
from auto_gpt.commands import Command, CommandRegistry


class TestCommand:
    @staticmethod
    def example_function(arg1: int, arg2: str) -> str:
        return f"{arg1} - {arg2}"

    def test_command_creation(self):
        cmd = Command(name="example", description="Example command", method=self.example_function)

        assert cmd.name == "example"
        assert cmd.description == "Example command"
        assert cmd.method == self.example_function
        assert cmd.signature == "(arg1: int, arg2: str) -> str"

    def test_command_call(self):
        cmd = Command(name="example", description="Example command", method=self.example_function)

        result = cmd(arg1=1, arg2="test")
        assert result == "1 - test"

    def test_command_call_with_invalid_arguments(self):
        cmd = Command(name="example", description="Example command", method=self.example_function)

        with pytest.raises(TypeError):
            cmd(arg1="invalid", does_not_exist="test")

    def test_command_default_signature(self):
        cmd = Command(name="example", description="Example command", method=self.example_function)

        assert cmd.signature == "(arg1: int, arg2: str) -> str"

    def test_command_custom_signature(self):
        custom_signature = "custom_arg1: int, custom_arg2: str"
        cmd = Command(name="example", description="Example command", method=self.example_function, signature=custom_signature)

        assert cmd.signature == custom_signature



class TestCommandRegistry:
    @staticmethod
    def example_function(arg1: int, arg2: str) -> str:
        return f"{arg1} - {arg2}"

    def test_register_command(self):
        """Test that a command can be registered to the registry."""
        registry = CommandRegistry()
        cmd = Command(name="example", description="Example command", method=self.example_function)

        registry.register(cmd)

        assert cmd.name in registry.commands
        assert registry.commands[cmd.name] == cmd

    def test_unregister_command(self):
        """Test that a command can be unregistered from the registry."""
        registry = CommandRegistry()
        cmd = Command(name="example", description="Example command", method=self.example_function)

        registry.register(cmd)
        registry.unregister(cmd.name)

        assert cmd.name not in registry.commands

    def test_get_command(self):
        """Test that a command can be retrieved from the registry."""
        registry = CommandRegistry()
        cmd = Command(name="example", description="Example command", method=self.example_function)

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
        cmd = Command(name="example", description="Example command", method=self.example_function)

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
        cmd = Command(name="example", description="Example command", method=self.example_function)

        registry.register(cmd)
        command_prompt = registry.command_prompt()

        assert f"(arg1: int, arg2: str)" in command_prompt

    def test_scan_directory_for_mock_commands(self):
        """Test that the registry can scan a directory for mocks command plugins."""
        registry = CommandRegistry()
        mock_commands_dir = Path("/app/auto_gpt/tests/mocks")
        import os

        print(os.getcwd())
        registry.scan_directory_for_plugins(mock_commands_dir)

        assert "function_based" in registry.commands
        assert registry.commands["function_based"].name == "function_based"
        assert registry.commands["function_based"].description == "Function-based test command"

    def test_scan_directory_for_temp_command_file(self, tmp_path):
        """Test that the registry can scan a directory for command plugins in a temp file."""
        registry = CommandRegistry()

        # Create a temp command file
        src = Path("/app/auto_gpt/tests/mocks/mock_commands.py")
        temp_commands_file = tmp_path / "mock_commands.py"
        shutil.copyfile(src, temp_commands_file)

        registry.scan_directory_for_plugins(tmp_path)
        print(registry.commands)

        assert "function_based" in registry.commands
        assert registry.commands["function_based"].name == "function_based"
        assert registry.commands["function_based"].description == "Function-based test command"
