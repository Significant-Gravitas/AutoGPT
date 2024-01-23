from __future__ import annotations

import sys

from llama_hub.tools.google_calendar import GoogleCalendarToolSpec
from llama_index.bridge.langchain import Tool as LlamaTool
from llama_index.tools.function_tool import FunctionTool

from AFAAS.core.tools.tool import Tool
from AFAAS.core.tools.tool_decorator import tool_from_langchain
from AFAAS.lib.sdk.logger import AFAASLogger

LOG = AFAASLogger(name=__name__)


def create_dynamic_function(tool_func, name):
    def dynamic_func(*args, **kwargs):
        return tool_func(*args, **kwargs)

    dynamic_func.__name__ = name
    return dynamic_func


tool_spec = GoogleCalendarToolSpec()
tool_list: list[FunctionTool] = tool_spec.to_tool_list()
ignore_list = ["get_date"]
for llamatool in tool_list:
    langchaintool: LlamaTool = llamatool.to_langchain_tool()

    # Create a unique class name using the name of the langchaintool
    class_name = f"Adapted{langchaintool.name.capitalize()}Tool"
    function_name = langchaintool.name.lower()
    if function_name in ignore_list:
        continue

    # Dynamically create a new class with a unique name
    # AdaptedToolClass = type(class_name, (langchaintool.__class__,), {})

    # Apply the tool_from_langchain decorator to the dynamically created class
    # AdaptedToolClass = tool_from_langchain()(AdaptedToolClass)
    generated_tool = Tool.generate_from_langchain_tool(
        langchain_tool=langchaintool, categories=["google", "calendar"]
    )

    # Create a dynamic function
    dynamic_function = create_dynamic_function(
        generated_tool, function_name
    )  # Add the dynamic function to the current module
    setattr(sys.modules[__name__], function_name, dynamic_function)

    LOG.debug(f"Tool : {function_name} created")

# load_data: Load the upcoming events from your calendar
# create_event: Creates a new Google Calendar event
# get_date: Utility for the Agent to get todays date
