import pytest

from AFAAS.core.tools.tool import Tool
from AFAAS.core.tools.tool_registry import DefaultToolRegistry
from AFAAS.interfaces.agent.main import BaseAgent
from AFAAS.interfaces.tools.tool_parameters import ToolParameter
from AFAAS.lib.utils.json_schema import JSONSchema

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


@pytest.fixture
def example_tool():
    yield Tool(
        name="example",
        description="Example command",
        exec_function=example_tool_exec_function,
        parameters=PARAMETERS,
        success_check_callback=Tool.default_tool_success_check_callback,
        make_summarry_function=Tool.default_tool_execution_summarry,
        categories=["undefined"],
    )
