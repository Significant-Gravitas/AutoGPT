"""
Ability for running Python code
"""
from typing import Dict
import subprocess
import json

from forge.sdk.memory.memstore_tools import add_memory

from ...forge_log import ForgeLogger
from ..registry import ability

logger = ForgeLogger(__name__)

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

    get_cwd = agent.workspace.get_cwd_path(task_id)

    return_dict = {
        "return_code": -1,
        "stdout": "",
        "stderr": ""
    }

    command = f"python {file_name}"

    try:
        req = subprocess.run(command,
            shell=True,
            capture_output=True,
            cwd=get_cwd
        )

        return_dict["return_code"] = req.returncode
        return_dict["stdout"] = req.stdout.decode()
        return_dict["stderr"] = req.stderr.decode()

        return_json = json.dumps(return_dict)
        add_memory(task_id, return_json, "run_bash_command")

        return return_json
    except json.JSONDecodeError as e:
        logger.error(f"JSON dumps failed for return_dict in run_bash_command: {e}")
        add_memory(task_id, str(return_dict), "run_bash_command")
    except Exception as e:
        logger.error(f"subprocess call failed: {e}")
    

    return str(return_dict)
