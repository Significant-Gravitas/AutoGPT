"""
Bash terminal abilities
"""
import subprocess
import json

from forge.sdk.memory.memstore_tools import add_memory

from ...forge_log import ForgeLogger
from ..registry import ability

logger = ForgeLogger(__name__)

@ability(
    name="run_bash_command",
    description="Run safe bash scripts or linux binaries in bash",
    parameters=[
        {
            "name": "command",
            "description": "Bash command to run",
            "type": "string",
            "required": True
        }
    ],
    output_type="str"
)
async def run_bash_command(agent, task_id: str, command: str) -> str:
    """
    run command with subprocess
    """

    req = subprocess.run(command, shell=True, capture_output=True)

    return_dict = {
        "return_code": req.returncode,
        "stdout": req.stdout,
        "stderr": req.stderr
    }

    try:
        return_json = json.dumps(return_dict)
        add_memory(task_id, return_json, "run_bash_command")

        return return_json
    except Exception as err:
        logger.error(f"JSON dumps failed for return_dict in run_bash_command: {err}")
        add_memory(task_id, str(return_dict), "run_bash_command")

        return str(return_dict)