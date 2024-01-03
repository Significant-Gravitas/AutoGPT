from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING
import asyncio
import pytest

if TYPE_CHECKING:
    from AFAAS.interfaces.agent.main import BaseAgent

from AFAAS.lib.utils.json_schema import JSONSchema
from AFAAS.core.tools.tools import Tool
from AFAAS.interfaces.tools.tool_parameters import ToolParameter
from AFAAS.core.tools.simple import SimpleToolRegistry

PARAMETERS = [
    ToolParameter(
        "arg1",
        spec=JSONSchema(
            type=JSONSchema.Type.INTEGER,
            description="Argument 1",
            required=True,
        ),
    ),
    ToolParameter(
        "arg2",
        spec=JSONSchema(
            type=JSONSchema.Type.STRING,
            description="Argument 2",
            required=False,
        ),
    ),
]


def example_tool_exec_function(arg1: int, arg2: str, agent: BaseAgent) -> str:
    """Example function for testing the Command class."""
    # This function is static because it is not used by any other test cases.
    return f"{arg1} - {arg2}"


def test_tool_creation():
    """Test that a Command object can be created with the correct attributes."""
    cmd = Tool(
        name="example",
        description="Example command",
        exec_function=example_tool_exec_function,
        parameters=PARAMETERS,
        success_check_callback=Tool.default_success_check_callback
    )

    assert cmd.name == "example"
    assert cmd.description == "Example command"
    assert cmd.method == example_tool_exec_function
    assert (
        str(cmd)
        == "example: Example command. Params: (arg1: integer, arg2: Optional[string])"
    )


@pytest.fixture
def example_tool():
    yield Tool(
        name="example",
        description="Example command",
        exec_function=example_tool_exec_function,
        parameters=PARAMETERS,
        success_check_callback=Tool.default_success_check_callback
    )


def test_tool_call(example_tool: Tool, agent: BaseAgent):
    """Test that Tool(*args) calls and returns the result of exec_function(*args)."""
    result = example_tool(arg1=1, arg2="test", agent=agent)
    assert result == "1 - test"


def test_tool_call_with_invalid_arguments(example_tool: Tool, agent: BaseAgent):
    """Test that calling a Command object with invalid arguments raises a TypeError."""
    with pytest.raises(TypeError):
        example_tool(arg1="invalid", does_not_exist="test", agent=agent)


def test_register_tool(example_tool: Tool, empty_tool_registry: SimpleToolRegistry):
    """Test that a command can be registered to the empty_tool_registry."""

    empty_tool_registry.register(example_tool)

    assert empty_tool_registry.get_tool(example_tool.name) == example_tool
    assert len(empty_tool_registry.tools) == 1


def test_unregister_tool(example_tool: Tool, empty_tool_registry: SimpleToolRegistry):
    """Test that a command can be unregistered from the empty_tool_registry."""

    empty_tool_registry.register(example_tool)
    empty_tool_registry.unregister(example_tool)

    assert len(empty_tool_registry.tools) == 0
    assert example_tool.name not in empty_tool_registry


@pytest.fixture
def example_tool_with_aliases(example_tool: Tool):
    example_tool.aliases = ["example_alias", "example_alias_2"]
    return example_tool


def test_register_tool_aliases(example_tool_with_aliases: Tool, empty_tool_registry: SimpleToolRegistry):
    """Test that a command can be registered to the empty_tool_registry."""
    command = example_tool_with_aliases

    empty_tool_registry.register(command)

    assert command.name in empty_tool_registry
    assert empty_tool_registry.get_tool(command.name) == command
    for alias in command.aliases:
        assert empty_tool_registry.get_tool(alias) == command
    assert len(empty_tool_registry.tools) == 1


def test_unregister_tool_aliases(example_tool_with_aliases: Tool, empty_tool_registry: SimpleToolRegistry):
    """Test that a command can be unregistered from the empty_tool_registry."""
    command = example_tool_with_aliases

    empty_tool_registry.register(command)
    empty_tool_registry.unregister(command)

    assert len(empty_tool_registry.tools) == 0
    assert command.name not in empty_tool_registry
    for alias in command.aliases:
        assert alias not in empty_tool_registry


def test_tool_in_registry(example_tool_with_aliases: Tool, empty_tool_registry: SimpleToolRegistry):
    """Test that `command_name in registry` works."""
    command = example_tool_with_aliases

    assert command.name not in empty_tool_registry
    assert "nonexistent_command" not in empty_tool_registry

    empty_tool_registry.register(command)

    assert command.name in empty_tool_registry
    assert "nonexistent_command" not in empty_tool_registry
    for alias in command.aliases:
        assert alias in empty_tool_registry


def test_get_tool(example_tool: Tool, empty_tool_registry: SimpleToolRegistry):
    """Test that a command can be retrieved from the empty_tool_registry."""

    empty_tool_registry.register(example_tool)
    retrieved_cmd = empty_tool_registry.get_tool(example_tool.name)

    assert retrieved_cmd == example_tool

#FIXME:
# def test_get_nonexistent_tool( empty_tool_registry: SimpleToolRegistry):
#     """Test that attempting to get a nonexistent command raises a KeyError."""

#     assert empty_tool_registry.get_tool("nonexistent_command") is None
#     assert "nonexistent_command" not in empty_tool_registry


@pytest.mark.asyncio  # This decorator is necessary for running async tests with pytest
async def test_call_tool(agent: BaseAgent, empty_tool_registry: SimpleToolRegistry):
    """Test that a command can be called through the empty_tool_registry."""
    cmd = Tool(
        name="example",
        description="Example command",
        exec_function=example_tool_exec_function,
        parameters=PARAMETERS,
        success_check_callback=Tool.default_success_check_callback
    )

    empty_tool_registry.register(cmd)
    result = await empty_tool_registry.call("example", arg1=1, arg2="test", agent=agent)

    assert result == "1 - test"

# FIXME:
# @pytest.mark.asyncio  # This decorator is necessary for running async tests with pytest
# async def test_call_nonexistent_tool(agent: BaseAgent, empty_tool_registry: SimpleToolRegistry):
#     """Test that attempting to call a nonexistent command raises a KeyError."""

#     with pytest.raises(KeyError):
#         await empty_tool_registry.call("nonexistent_command", arg1=1, arg2="test", agent=task_ready_no_predecessors_or_subtasks.agent)

#FIXME:
# def test_import_mock_commands_module( empty_tool_registry: SimpleToolRegistry):
#     """Test that the registry can import a module with mock command plugins."""
#     mock_commands_module = "tests.mocks.mock_commands"

#     empty_tool_registry.import_tool_module(mock_commands_module)

#     assert "function_based_cmd" in empty_tool_registry
#     assert empty_tool_registry.tools["function_based_cmd"].name == "function_based_cmd"
#     assert (
#         empty_tool_registry.tools["function_based_cmd"].description
#         == "Function-based test command"
#     )

# FIXME:
# def test_import_temp_tool_file_module(tmp_path: Path, empty_tool_registry: SimpleToolRegistry):
#     """
#     Test that the registry can import a command plugins module from a temp file.
#     Args:
#         tmp_path (pathlib.Path): Path to a temporary directory.
#     """

#     # Create a temp command file
#     src = Path(os.getcwd()) / "tests/mocks/mock_commands.py"
#     temp_commands_file = tmp_path / "mock_commands.py"
#     shutil.copyfile(src, temp_commands_file)

#     # Add the temp directory to sys.path to make the module importable
#     sys.path.append(str(tmp_path))

#     temp_commands_module = "mock_commands"
#     empty_tool_registry.import_tool_module(temp_commands_module)

#     # Remove the temp directory from sys.path
#     sys.path.remove(str(tmp_path))

#     assert "function_based_cmd" in empty_tool_registry
#     assert empty_tool_registry.tools["function_based_cmd"].name == "function_based_cmd"
#     assert (
#         empty_tool_registry.tools["function_based_cmd"].description
#         == "Function-based test command"
#     )
