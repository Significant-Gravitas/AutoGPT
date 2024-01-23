from __future__ import annotations
import pytest
from AFAAS.core.tools.tool import Tool
from AFAAS.plugins.tools.langchain_google_places import GooglePlacesTool
from unittest.mock import Mock, patch
from tests.dataset.plan_familly_dinner import (
    Task,
    _plan_familly_dinner,
    default_task,
    plan_familly_dinner_with_tasks_saved_in_db,
    plan_step_0,
    task_ready_no_predecessors_or_subtasks,
)

def test_tool_creation_with_valid_langchain_tool():

    langchain_tool = GooglePlacesTool()
    tool = Tool.generate_from_langchain_tool(langchain_tool)
    assert tool.name == langchain_tool.name
    assert tool.description == langchain_tool.description
    assert tool.exec_function is not None
    assert tool.parameters is not None
    assert tool.categories == ["undefined"]

def test_tool_creation_with_valid_langchain_tool(empty_tool_registry):
    langchain_tool = GooglePlacesTool()
    tool = Tool.generate_from_langchain_tool(langchain_tool)
    empty_tool_registry.register(tool)

    registered_tool = empty_tool_registry.get_tool(langchain_tool.name)
    assert registered_tool.name == langchain_tool.name
    assert registered_tool.description == langchain_tool.description

@pytest.mark.asyncio
async def test_argument_conversion(empty_tool_registry , default_task: Task):
    pytest.skip("Requires Google Maps API Key")
    def mock_arg_converter(kwargs, agent):
        return {k: v.upper() if isinstance(v, str) else v for k, v in kwargs.items()}

    langchain_tool = GooglePlacesTool()
    tool = Tool.generate_from_langchain_tool(langchain_tool, arg_converter=mock_arg_converter)
    empty_tool_registry.register(tool)

    tool = empty_tool_registry.get_tool(tool.name)
    result = await tool(query="lowercase", task = default_task,  agent=default_task.agent)
    assert result['query'] == "LOWERCASE"

def test_custom_success_check_callback(empty_tool_registry):
    def custom_success_check(result):
        return "custom_check_passed" in result

    langchain_tool = GooglePlacesTool()
    tool = Tool.generate_from_langchain_tool(langchain_tool, success_check_callback=custom_success_check)
    empty_tool_registry.register(tool)

    registered_tool = empty_tool_registry.get_tool(langchain_tool.name)
    assert registered_tool.success_check_callback is custom_success_check

def test_default_success_check_callback(empty_tool_registry):
    langchain_tool = GooglePlacesTool()
    tool = Tool.generate_from_langchain_tool(langchain_tool)
    empty_tool_registry.register(tool)

    registered_tool = empty_tool_registry.get_tool(langchain_tool.name)
    assert registered_tool.success_check_callback == Tool.default_success_check_callback

