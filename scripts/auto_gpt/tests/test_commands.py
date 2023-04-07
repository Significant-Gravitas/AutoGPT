from pathlib import Path

import pytest
from commands import Command, CommandRegistry


class TestCommand:
    @staticmethod
    def example_function(arg1: int, arg2: str) -> str:
        return f"{arg1} - {arg2}"

    def test_command_creation(self):
        cmd = Command(name="example", description="Example command", method=self.example_function)

        assert cmd.name == "example"
        assert cmd.description == "Example command"
        assert cmd.method == self.example_function
        assert cmd.signature == "arg1: int, arg2: str"

    def test_command_call(self):
        cmd = Command(name="example", description="Example command", method=self.example_function)

        result = cmd(arg1=1, arg2="test")
        assert result == "1 - test"

    def test_command_call_with_invalid_arguments(self):
        cmd = Command(name="example", description="Example command", method=self.example_function)

        with pytest.raises(TypeError):
            cmd(arg1="invalid", arg2="test")

    def test_command_default_signature(self):
        cmd = Command(name="example", description="Example command", method=self.example_function)

        assert cmd.signature == "arg1: int, arg2: str"

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

        assert cmd.name in registry._commands
        assert registry._commands[cmd.name] == cmd

    def test_unregister_command(self):
        """Test that a command can be unregistered from the registry."""
        registry = CommandRegistry()
        cmd = Command(name="example", description="Example command", method=self.example_function)

        registry.register(cmd)
        registry.unregister(cmd.name)

        assert cmd.name not in registry._commands

    def test_get_command(self):
        """Test that a command can be retrieved from the registry."""
        registry = CommandRegistry()
        cmd = Command(name="example", description="Example command", method=self.example_function)

        registry.register(cmd)
        retrieved_cmd = registry.get(cmd.name)

        assert retrieved_cmd == cmd

    def test_get_nonexistent_command(self):
        """Test that attempting to get a nonexistent command raises a KeyError."""
        registry = CommandRegistry()

        with pytest.raises(KeyError):
            registry.get("nonexistent_command")

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

    def test_get_command_list(self):
        """Test that a list of registered commands can be retrieved."""
        registry = CommandRegistry()
        cmd1 = Command(name="example1", description="Example command 1", method=self.example_function)
        cmd2 = Command(name="example2", description="Example command 2", method=self.example_function)

        registry.register(cmd1)
        registry.register(cmd2)
        command_list = registry.get_command_list()

        assert len(command_list) == 2
        assert cmd1.name in command_list
        assert cmd2.name in command_list

    def test_get_command_prompt(self):
        """Test that the command prompt is correctly formatted."""
        registry = CommandRegistry()
        cmd = Command(name="example", description="Example command", method=self.example_function)

        registry.register(cmd)
        command_prompt = registry.get_command_prompt()

        assert f"{cmd.name}: {cmd.description}, args: {cmd.signature}" in command_prompt

    def test_scan_directory_for_mock_commands(self):
        """Test that the registry can scan a directory for mock command plugins."""
        registry = CommandRegistry()
        mock_commands_dir = Path("auto_gpt/tests/mocks")

        registry.scan_directory_for_plugins(mock_commands_dir)

        assert "mock_class_based" in registry._commands
        assert registry._commands["mock_class_based"].name == "mock_class_based"
        assert registry._commands["mock_class_based"].description == "Mock class-based command"

        assert "mock_function_based" in registry._commands
        assert registry._commands["mock_function_based"].name == "mock_function_based"
        assert registry._commands["mock_function_based"].description == "Mock function-based command"

    def test_scan_directory_for_temp_command_file(self, tmp_path):
        """Test that the registry can scan a directory for command plugins in a temp file."""
        registry = CommandRegistry()
        temp_commands_dir = tmp_path / "temp_commands"
        temp_commands_dir.mkdir()

        # Create a temp command file
        temp_commands_file = temp_commands_dir / "temp_commands.py"
        temp_commands_content = (
            "from commands import Command, command\n\n"
            "class TempCommand(Command):\n"
            "    def __init__(self):\n"
            "        super().__init__(name='temp_class_based', description='Temp class-based command')\n\n"
            "    def __call__(self, arg1: int, arg2: str) -> str:\n"
            "        return f'{arg1} - {arg2}'\n\n"
            "@command('temp_function_based', 'Temp function-based command')\n"
            "def temp_function_based(arg1: int, arg2: str) -> str:\n"
            "    return f'{arg1} - {arg2}'\n"
        )

        with open(temp_commands_file, "w") as f:
            f.write(temp_commands_content)

        registry.scan_directory_for_plugins(temp_commands_dir)

        assert "temp_class_based" in registry._commands
        assert registry._commands["temp_class_based"].name == "temp_class_based"
        assert registry._commands["temp_class_based"].description == "Temp class-based command"

        assert "temp_function_based" in registry._commands
        assert registry._commands["temp_function_based"].name == "temp_function_based"
        assert registry._commands["temp_function_based"].description == "Temp function-based command"
