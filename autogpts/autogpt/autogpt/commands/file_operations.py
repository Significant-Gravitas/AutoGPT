"""Commands to perform operations on files"""

from __future__ import annotations

import hashlib
import inspect
import logging
import os
import os.path
from enum import Enum
from pathlib import Path
from typing import Awaitable, Callable, Iterator, Literal

from autogpt.agents.agent import Agent
from autogpt.agents.utils.exceptions import (
    CommandExecutionError,
    DuplicateOperationError,
    InvalidArgumentError,
)
from autogpt.command_decorator import command
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.memory.vector import MemoryItemFactory, VectorMemory

from .decorators import sanitize_path_arg
from .file_operations_utils import decode_textual_file

COMMAND_CATEGORY = "file_operations"
COMMAND_CATEGORY_TITLE = "File Operations"


from .file_context import open_file, open_folder  # NOQA

logger = logging.getLogger(__name__)


class Operations(Enum):
    WRITE = "write"
    APPEND = "append"
    DELETE = "delete"


def text_checksum(text: str) -> str:
    """Get the hex checksum for the given text."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def operations_from_log(
    logs: list[str],
) -> Iterator[
    tuple[Operations, str, str] | tuple[Literal[Operations.DELETE], str, None]
]:
    """
    Parse logs and return a tuple containing the log entries

    Yields:
        Operations: The operation recorded in the log entry
        str: The file path affected
        str | None: The checksum of the file if written, None if deleted
    """
    for line in logs:
        line = line.replace("File Operation Logger", "").strip()
        if not line:
            continue
        operation, tail = line.split(": ", maxsplit=1)
        operation = Operations[operation.strip()]
        if operation in (Operations.WRITE, Operations.APPEND):
            try:
                path, checksum = (x.strip() for x in tail.rsplit(" #", maxsplit=1))
            except ValueError:
                logger.warn(f"File log entry lacks checksum: '{line}'")
                path, checksum = tail.strip(), None
            yield (operation, path, checksum)
        elif operation == Operations.DELETE:
            yield (Operations.DELETE, tail.strip(), None)


def file_operations_state(logs: list[str]) -> dict[str, str]:
    """Iterates over the operations and returns the expected state.

    Constructs a dictionary that maps each written or appended file to its checksum.
    Deleted files are removed from the dictionary.

    Returns:
        A dictionary mapping file paths to their checksums.

    Raises:
        FileNotFoundError: If file_manager.file_ops_log_path is not found.
        ValueError: If the log file content is not in the expected format.
    """
    state = {}
    for operation, path, checksum in operations_from_log(logs):
        if operation in (Operations.WRITE, Operations.APPEND):
            state[path] = checksum
        elif operation == Operations.DELETE:
            del state[path]
    return state


@sanitize_path_arg("file_path", make_relative=True)
def is_duplicate_operation(
    operation: Operations, file_path: Path, agent: Agent, checksum: str | None = None
) -> bool:
    """Check if the operation has already been performed

    Args:
        operation: The operation to check for
        file_path: The name of the file to check for
        agent: The agent
        checksum: The checksum of the contents to be written

    Returns:
        True if the operation has already been performed on the file
    """
    state = file_operations_state(agent.get_file_operation_lines())
    if operation == Operations.DELETE and file_path.as_posix() not in state:
        return True
    if operation == Operations.WRITE and state.get(file_path.as_posix()) == checksum:
        return True
    return False


@sanitize_path_arg("file_path", make_relative=True)
async def log_operation(
    operation: Operations,
    file_path: str | Path,
    agent: Agent,
    checksum: str | None = None,
) -> None:
    """Log the file operation to the file_logger.log

    Args:
        operation: The operation to log
        file_path: The name of the file the operation was performed on
        checksum: The checksum of the contents to be written
    """
    log_entry = (
        f"{operation.value}: "
        f"{file_path.as_posix() if isinstance(file_path, Path) else file_path}"
    )
    if checksum is not None:
        log_entry += f" #{checksum}"
    logger.debug(f"Logging file operation: {log_entry}")
    await agent.log_file_operation(log_entry)


@command(
    "read_file",
    "Read an existing file",
    {
        "filename": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The path of the file to read",
            required=True,
        )
    },
)
def read_file(filename: str | Path, agent: Agent) -> str:
    """Read a file and return the contents

    Args:
        filename (Path): The name of the file to read

    Returns:
        str: The contents of the file
    """
    file = agent.workspace.open_file(filename, binary=True)
    content = decode_textual_file(file, os.path.splitext(filename)[1], logger)

    # # TODO: invalidate/update memory when file is edited
    # file_memory = MemoryItem.from_text_file(content, str(filename), agent.config)
    # if len(file_memory.chunks) > 1:
    #     return file_memory.summary

    return content


def ingest_file(
    filename: str,
    memory: VectorMemory,
) -> None:
    """
    Ingest a file by reading its content, splitting it into chunks with a specified
    maximum length and overlap, and adding the chunks to the memory storage.

    Args:
        filename: The name of the file to ingest
        memory: An object with an add() method to store the chunks in memory
    """
    try:
        logger.info(f"Ingesting file {filename}")
        content = read_file(filename)

        # TODO: differentiate between different types of files
        file_memory = MemoryItemFactory.from_text_file(content, filename)
        logger.debug(f"Created memory: {file_memory.dump(True)}")
        memory.add(file_memory)

        logger.info(f"Ingested {len(file_memory.e_chunks)} chunks from {filename}")
    except Exception as err:
        logger.warning(f"Error while ingesting file '{filename}': {err}")


@command(
    "write_file",
    "Write a file, creating it if necessary.",
    {
        "filename": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The name of the file to write to",
            required=True,
        ),
        "contents": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The contents to write to the file",
            required=True,
        ),
        "if_exists": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="What to do if the file already exists",
            enum=["overwrite", "prepend", "append", "skip", "fail"],
            required=True,
        ),
    },
    aliases=["create_file"],
)
async def write_to_file(
    filename: str | Path, contents: str, if_exists: str, agent: Agent
) -> str:
    """Write contents to a file

    Args:
        filename (Path): The name of the file to write to
        contents (str): The contents to write to the file
        if_exists (str): One of 'overwrite', 'prepend', 'append', 'skip' or 'fail'

    Returns:
        str: A message indicating success or failure
    """
    ACTIONS: dict[
        str, Callable[[str | Path, str], str | None | Exception | Awaitable[None]]
    ] = {
        "overwrite": lambda p, _txt: agent.workspace.write_file(p, _txt),
        "prepend": lambda p, _txt: (
            agent.workspace.write_file(p, _txt + agent.workspace.read_file(p))
        ),
        "append": lambda p, _txt: (
            agent.workspace.write_file(p, agent.workspace.read_file(p) + _txt)
        ),
        "skip": (
            lambda p, _txt: "File exists, skipping."
            if agent.workspace.exists(p)
            else agent.workspace.write_file(p, _txt)
        ),
        "fail": (
            lambda p, _txt: CommandExecutionError("File exists")
            if agent.workspace.exists(p)
            else agent.workspace.write_file(p, _txt)
        ),
    }
    if if_exists not in ACTIONS:
        raise InvalidArgumentError("Invalid value for 'if_exists'")

    checksum = text_checksum(contents)
    if is_duplicate_operation(Operations.WRITE, Path(filename), agent, checksum):
        raise DuplicateOperationError(f"File {filename} has already been updated.")

    if directory := os.path.dirname(filename):
        agent.workspace.make_dir(directory)

    result = ACTIONS[if_exists](filename, contents)
    if inspect.isawaitable(result):
        await result

    if isinstance(result, str):  # If the result is a string, return it
        return result
    if isinstance(result, Exception):
        raise result

    await log_operation(Operations.WRITE, filename, agent, checksum)
    return f"File {filename} has been written successfully."


@command(
    "list_folder",
    "List the items in a folder",
    {
        "folder": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The folder to list files in",
            required=True,
        )
    },
)
def list_folder(folder: str | Path, agent: Agent) -> list[str]:
    """Lists files in a folder recursively

    Args:
        folder (Path): The folder to search in

    Returns:
        list[str]: A list of files found in the folder
    """
    return [str(p) for p in agent.workspace.list_files(folder)]
