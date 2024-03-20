"""Commands to perform operations on files"""

from __future__ import annotations

import hashlib
import logging
import os
import os.path
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Literal

from autogpt.agents.agent import Agent
from autogpt.agents.base import BaseAgent, CommandArgs
from autogpt.command_decorator import command
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.memory.vector import MemoryItemFactory, VectorMemory
from autogpt.models.command import ValidityResult

from .decorators import sanitize_path_arg
from .file_operations_utils import decode_textual_file

COMMAND_CATEGORY = "file_operations"
COMMAND_CATEGORY_TITLE = "File Operations"


from .file_context import open_file, open_folder  # NOQA

logger = logging.getLogger(__name__)


def text_checksum(text: str) -> str:
    """Get the hex checksum for the given text."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _is_write_valid(agent: Agent, arguments: CommandArgs) -> ValidityResult:
    if not agent.workspace.exists(arguments["filename"]):
        return ValidityResult(True)

    if agent.workspace.read_file(arguments["filename"]):
        if text_checksum(arguments["contents"]) == text_checksum(
            agent.workspace.read_file(arguments["filename"])
        ):
            return ValidityResult(
                False, "Trying to write the same content to the same file."
            )

    return ValidityResult(True)


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
    "Write a file, creating it if necessary. If the file exists, it is overwritten.",
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
    },
    aliases=["create_file"],
    is_valid=_is_write_valid,
)
async def write_to_file(filename: str | Path, contents: str, agent: Agent) -> str:
    """Write contents to a file

    Args:
        filename (Path): The name of the file to write to
        contents (str): The contents to write to the file

    Returns:
        str: A message indicating success or failure
    """
    if directory := os.path.dirname(filename):
        agent.workspace.make_dir(directory)
    await agent.workspace.write_file(filename, contents)
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
