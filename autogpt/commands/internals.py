import os
import platform
import subprocess
import sys
import json
from autogpt.commands.command import command

from autogpt.config import Config
from autogpt.logs import logger


CFG = Config()


@command(
    "get_system_info",
    "Get System Information",
    '"arguments": "<args>"'
)
def get_system_info(arguments: str) -> str:
    """
    A function that returns various Python script internals and environment details.

    Returns:
        A string representation of the dictionary containing the Python version, platform,
        operating system, current working directory, shell information, shell version,
        and additional environment information.
    """

    script_info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'operating_system': platform.system(),
        'cwd': os.getcwd(),
        'shell': determine_shell(),
        'shell_version': determine_shell_version(),
        'processor': platform.processor(),
        'architecture': platform.machine(),
        'hostname': platform.node(),
    }
    return json.dumps(script_info)

def determine_shell() -> str:
    """
    A function that determines the available shell based on the operating system.

    Returns:
        The name of the available shell or 'Unknown' if it cannot be determined.
    """

    if os.name == 'posix':
        return os.getenv('SHELL', 'Unknown')
    elif os.name == 'nt':
        return os.getenv('COMSPEC', 'Unknown')
    else:
        return 'Unknown'

def determine_shell_version() -> str:
    """
    A function that determines the version of the available shell.

    Returns:
        The version of the shell or 'Unknown' if it cannot be determined.
    """

    shell = determine_shell()
    if shell != 'Unknown':
        try:
            output = subprocess.check_output([shell, '--version'], stderr=subprocess.DEVNULL)
            return output.decode().strip()
        except (subprocess.CalledProcessError, OSError):
            pass
    return 'Unknown'

