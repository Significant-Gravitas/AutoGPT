from __future__ import annotations

import copy
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Assuming the not_implemented_tool function is defined in a module named `tools`
from AFAAS.core.tools.builtins.search_info import AbstractChatModelResponse, search_info
from AFAAS.core.tools.tool import Tool
from AFAAS.interfaces.task.meta import AFAASModel, TaskStatusList
from AFAAS.interfaces.task.stack import TaskStack
from AFAAS.lib.task.plan import Plan
from AFAAS.lib.task.task import BaseAgent, Task

from .dataset.agent_planner import agent_dataset
from .dataset.example_tool_exec_function import (
    PARAMETERS,
    example_tool,
    example_tool_exec_function,
)
from .dataset.plan_familly_dinner import (
    _plan_familly_dinner,
    _plan_step_3,
    _plan_step_10,
    _plan_step_11,
    _plan_step_12,
    _plan_step_13,
    _plan_step_14,
    _plan_step_15,
    _plan_step_16,
    _plan_step_17,
    _plan_step_18,
    _plan_step_19,
    _plan_step_20,
    _plan_step_21,
    _plan_step_22,
    _plan_step_23,
    _plan_step_24,
    default_task,
    plan_familly_dinner_with_tasks_saved_in_db,
    plan_step_0,
    plan_step_1,
    plan_step_2,
    plan_step_3,
    plan_step_4,
    plan_step_5,
    plan_step_6,
    plan_step_7,
    plan_step_8,
    plan_step_9,
    plan_step_10,
    plan_step_11,
    plan_step_12,
    plan_step_13,
    plan_step_14,
    task_awaiting_preparation,
    task_ready_no_predecessors_or_subtasks,
    task_with_mixed_predecessors,
    task_with_ongoing_subtasks,
    task_with_unmet_predecessors,
)


@pytest.mark.asyncio
async def test_default_post_prossessing_retry(
    example_tool: Tool, default_task: Task, mocker
):
    # Mock dependencies
    mock_tool_output = {"result": "success"}
    default_task.memorize_output = AsyncMock()

    mock_search_result = MagicMock(spec=AbstractChatModelResponse)
    mock_parameters = {param.name: MagicMock() for param in example_tool.parameters}
    mock_assistant_response = MagicMock()
    mock_search_result.parsed_result = [
        {
            "command_name": example_tool.name,
            "command_args": {
                "text_output": "Processed output",
                "text_output_as_uml": "UML output",
            },
            "mock_assistant_response": {},
        }
    ]
    default_task.agent.execute_strategy = AsyncMock(return_value=mock_search_result)

    example_tool.success_check_callback = AsyncMock(return_value=False)
    default_task.retry = AsyncMock()
    await default_task.process_tool_output(example_tool, mock_tool_output)
    default_task.memorize_output.assert_called_once()
    default_task.retry.assert_called_once()
    assert (
        default_task.task_text_output == "Processed output"
    ), "The task text output should be updated with the processed output"
    assert (
        default_task.task_text_output_as_uml == "UML output"
    ), "The task UML output should be updated accordingly"

    example_tool.success_check_callback = AsyncMock(return_value=True)
    default_task.retry = AsyncMock()
    await default_task.process_tool_output(example_tool, mock_tool_output)
    default_task.retry.assert_not_called()


@pytest.mark.asyncio
async def test_tool_execution_summarry(example_tool: Tool, default_task: Task, mocker):
    pytest.skip("This test is not yet implemented")
    # Mock dependencies
    mock_tool_output = {"result": "success"}
    mock_strategy_result = mocker.MagicMock()
    mock_strategy_result.parsed_result = [
        {
            "command_args": {
                "text_output": "Processed output",
                "text_output_as_uml": "UML output",
            }
        }
    ]
    mocker.patch.object(
        default_task.agent,
        "execute_strategy",
        new_callable=AsyncMock(mock_strategy_result),
    )

    # Execute the summary function
    await example_tool.default_tool_execution_summarry(default_task, mock_tool_output)

    # Assertions to verify correct behavior
    assert (
        default_task.task_text_output == "Processed output"
    ), "The task text output should be updated with the processed output"
    assert (
        default_task.task_text_output_as_uml == "UML output"
    ), "The task UML output should be updated accordingly"
