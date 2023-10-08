"""
Calculator for AI
"""

import subprocess
import json

from forge.sdk.memory.memstore_tools import add_ability_memory

from ..forge_log import ForgeLogger
from .registry import ability

logger = ForgeLogger(__name__)

@ability(
    name="math_run_expression",
    description="Eval and return answer to mathematical expression using Python",
    parameters=[
        {
            "name": "math_expression",
            "description": "Mathematical expression",
            "type": "string",
            "required": True,
        }
    ],
    output_type="str",
)
async def math_run_expression(agent, task_id: str, math_expression: str) -> str:
    return_dict = {
        "return_code": -1,
        "stdout": "",
        "stderr": ""
    }

    command = f"python -c 'print({math_expression})'"
    try:
        req = subprocess.run(command,
            shell=True,
            capture_output=True
        )

        return_dict["return_code"] = req.returncode
        return_dict["stdout"] = req.stdout.decode()
        return_dict["stderr"] = req.stderr.decode()
    except Exception as err:
        logger.error(f"subprocess call failed: {err}")
        raise err
    
    try:
        return_json = json.dumps(return_dict)
    except json.JSONDecodeError as err:
        logger.error(f"JSON dumps failed: {err}")
        raise err
    
    add_ability_memory(task_id, return_json, "run_python_file")
    return return_json