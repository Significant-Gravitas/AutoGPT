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
    get_cwd = agent.workspace.get_cwd_path(task_id)

    return_dict = {
        "return_code": -1,
        "stdout": "",
        "stderr": ""
    }

    try:
        req = subprocess.run(command,
            shell=True,
            capture_output=True,
            cwd=get_cwd
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
        logger.error(f"JSON dumps failed in run_bash_command: {err}")
        raise err

    await add_memory(task_id, return_json, "run_bash_command")
    return return_json