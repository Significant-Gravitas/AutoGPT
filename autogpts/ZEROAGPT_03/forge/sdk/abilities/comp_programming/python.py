"""
Ability for running Python code
"""
from typing import Dict
import io
from contextlib import redirect_stdout

from ..registry import ability

@ability(
    name="run_python_file",
    description="run a python file",
    parameters=[
        {
            "name": "file_name",
            "description": "Name of the file",
            "type": "string",
            "required": True
        }
    ],
    output_type="dict"
)

async def run_python_file(agent, task_id: str, file_name: str) -> Dict:
    """
    run_python_file
    Uses the UNSAFE exec method after reading file from local workspace
    Look for safer method
    """
    open_file = agent.workspace.read(task_id, file_name)
    
    response = {
        "output": None,
        "errors": None
    }

    out_io = io.StringIO()

    try:
        with redirect_stdout(out_io):
            exec(open_file)
        response["output"] = out_io.getvalue()
    except Exception as e:
        response["errors"] = str(e)
    
    return response
