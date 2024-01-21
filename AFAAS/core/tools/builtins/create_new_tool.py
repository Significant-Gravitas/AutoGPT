from __future__ import annotations

import importlib
import os

from AFAAS.core.tools.tool_decorator import tool


@tool(
    name="create_function",
    description="This function can create a Python function based on a specific JSON object that describes the desired functionality, including the required imports.",
    parameters={
        "type": "object",
        "properties": {
            "function_description": {
                "type": "object",
                "description": "This parameter receives a JSON object that describes the function you want to create. It must respect the following JSON Schema:",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the function to be created",
                    },
                    "description": {
                        "type": "string",
                        "description": "A brief description of what the function does",
                    },
                    "packages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "A list of required package to be imported",
                    },
                    "imports": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "A list of required Python modules to import (ex : from package import module)",
                    },
                    "new_config_keys": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "key": {
                                    "type": "string",
                                    "description": "The key name to be added; in the format TOOL_NAMEOFUNCTION_NAMEOFKEY",
                                },
                                "description": {
                                    "type": "string",
                                    "description": "Description of the new  key",
                                },
                            },
                            "required": ["key", "description"],
                        },
                        "description": "A list of envirnmental variable that must be accessible via os.ENVIRON, ideal to store API_KEY, configuration & credentials. ",
                    },
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["object"]},
                            "properties": {
                                "type": "object",
                                "description": "A description of the parameters and their types",
                            },
                            "required": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "A list of required parameters",
                            },
                        },
                        "required": ["type", "properties"],
                    },
                },
                "required": ["name", "parameters"],
            }
        },
        "required": ["function_description"],
    },
)
def create_new_tool(self, function_description):
    # Import statements
    imports = "\n".join(
        f"import {module}" for module in function_description["imports"]
    )

    # Adding new config keys
    new_config_keys = "\n    ".join(
        f"# {key['description']}\n    config['{key['key']}'] = None"
        for key in function_description.get("new_config_keys", [])
    )

    # Function parameters
    parameters = ", ".join(function_description["parameters"]["required"])

    # Function body
    body = f"""
def {function_description['name']}({parameters}):
    {new_config_keys}
    """

    # Combining everything
    code = f"""
{imports}

{body}
    """

    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Path for the new Python file
    file_path = os.path.join(current_directory, function_description["name"] + ".py")

    # Write the Python code to the file
    with open(file_path, "w") as file:
        file.write(code)

    # Load the module
    spec = importlib.util.spec_from_file_location(
        function_description["name"], file_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return code
