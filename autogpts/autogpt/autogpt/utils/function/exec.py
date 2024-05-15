import asyncio
import enum
import logging
import os
import subprocess
import tempfile
from asyncio.subprocess import Process
from pathlib import Path

logger = logging.getLogger(__name__)


class OutputType(enum.Enum):
    STD_OUT = "stdout"
    STD_ERR = "stderr"
    BOTH = "both"


class ExecError(Exception):
    content: str | None

    def __init__(self, error: str, content: str | None = None):
        super().__init__(error)
        self.content = content


async def exec_external_on_contents(
    command_arguments: list[str],
    file_contents,
    suffix: str = ".py",
    output_type: OutputType = OutputType.BOTH,
    raise_file_contents_on_error: bool = False,
) -> str:
    """
    Execute an external tool with the provided command arguments and file contents
    :param command_arguments: The command arguments to execute
    :param file_contents: The file contents to execute the command on
    :param suffix: The suffix of the temporary file. Default is ".py"
    :return: The file contents after the command has been executed

    Note: The file contents are written to a temporary file and the command is executed
    on that file. The command arguments should be a list of strings, where the first
    element is the command to execute and the rest of the elements are the arguments to
    the command. There is no need to provide the file path as an argument, as it will
    be appended to the command arguments.

    Example:
    exec_external(["ruff", "check"], "print('Hello World')")
    will run the command "ruff check <temp_file_path>" with the file contents
    "print('Hello World')" and return the file contents after the command
    has been executed.

    """
    errors = ""
    if len(command_arguments) == 0:
        raise ExecError("No command arguments provided")

    # Run ruff to validate the code
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(file_contents.encode("utf-8"))
        temp_file.flush()

        command_arguments.append(str(temp_file_path))

        # Run Ruff on the temporary file
        try:
            r: Process = await asyncio.create_subprocess_exec(
                *command_arguments,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            result = await r.communicate()
            stdout, stderr = result[0].decode("utf-8"), result[1].decode("utf-8")
            logger.debug(f"Output: {stdout}")
            if temp_file_path in stdout:
                stdout = stdout  # .replace(temp_file.name, "/generated_file")
                logger.debug(f"Errors: {stderr}")
                if output_type == OutputType.STD_OUT:
                    errors = stdout
                elif output_type == OutputType.STD_ERR:
                    errors = stderr
                else:
                    errors = stdout + "\n" + stderr

            with open(temp_file_path, "r") as f:
                file_contents = f.read()
        finally:
            # Ensure the temporary file is deleted
            os.remove(temp_file_path)

    if not errors:
        return file_contents

    if raise_file_contents_on_error:
        raise ExecError(errors, file_contents)

    raise ExecError(errors)


FOLDER_NAME = "agpt-static-code-analysis"
PROJECT_PARENT_DIR = Path(__file__).resolve().parent.parent.parent / f".{FOLDER_NAME}"
PROJECT_TEMP_DIR = Path(tempfile.gettempdir()) / FOLDER_NAME
DEFAULT_DEPS = ["pyright", "pydantic", "virtualenv-clone"]


def is_env_exists(path: Path):
    return (
        (path / "venv/bin/python").exists()
        and (path / "venv/bin/pip").exists()
        and (path / "venv/bin/virtualenv-clone").exists()
        and (path / "venv/bin/pyright").exists()
    )


async def setup_if_required(
    cwd: Path = PROJECT_PARENT_DIR, copy_from_parent: bool = True
) -> Path:
    """
    Set-up the virtual environment if it does not exist
    This setup is executed expectedly once per application run
    Args:
        cwd (Path): The current working directory
        copy_from_parent (bool): 
          Whether to copy the virtual environment from PROJECT_PARENT_DIR
    Returns:
        Path: The path to the virtual environment
    """
    if not cwd.exists():
        cwd.mkdir(parents=True, exist_ok=True)

    path = cwd / "venv/bin"
    if is_env_exists(cwd):
        return path

    if copy_from_parent and cwd != PROJECT_PARENT_DIR:
        if (cwd / "venv").exists():
            await execute_command(["rm", "-rf", str(cwd / "venv")], cwd, None)
        await execute_command(
            ["virtualenv-clone", str(PROJECT_PARENT_DIR / "venv"), str(cwd / "venv")],
            cwd,
            await setup_if_required(PROJECT_PARENT_DIR),
        )
        return path

    # Create a virtual environment
    output = await execute_command(["python", "-m", "venv", "venv"], cwd, None)
    logger.info(f"[Setup] Created virtual environment: {output}")

    # Install dependencies
    output = await execute_command(["pip", "install", "-I"] + DEFAULT_DEPS, cwd, path)
    logger.info(f"[Setup] Installed {DEFAULT_DEPS}: {output}")

    output = await execute_command(["pyright"], cwd, path, raise_on_error=False)
    logger.info(f"[Setup] Set up pyright: {output}")

    return path


async def execute_command(
    command: list[str],
    cwd: str | Path | None,
    python_path: str | Path | None = None,
    raise_on_error: bool = True,
) -> str:
    """
    Execute a command in the shell
    Args:
        command (list[str]): The command to execute
        cwd (str | Path): The current working directory
        python_path (str | Path): The python executable path
        raise_on_error (bool): Whether to raise an error if the command fails
    Returns:
        str: The output of the command
    """
    # Set the python path by replacing the env 'PATH' with the provided python path
    venv = os.environ.copy()
    if python_path:
        # PATH prioritize first occurrence of python_path, so we need to prepend.
        venv["PATH"] = f"{python_path}:{venv['PATH']}"
    r = await asyncio.create_subprocess_exec(
        *command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(cwd),
        env=venv,
    )
    stdout, stderr = await r.communicate()
    if r.returncode == 0:
        return (stdout or stderr).decode("utf-8")

    if raise_on_error:
        raise Exception((stderr or stdout).decode("utf-8"))
    else:
        return (stderr or stdout).decode("utf-8")
