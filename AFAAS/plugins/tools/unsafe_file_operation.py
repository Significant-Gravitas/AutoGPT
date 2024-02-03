"""Tools to perform operations on files"""

from __future__ import annotations

import os
import os.path
import shutil
from pathlib import Path

from langchain.vectorstores import VectorStore
from langchain_community.tools.file_management.file_search import FileSearchTool

from AFAAS.core.tools.builtins.file_operations_helpers import (
    is_duplicate_operation,
    log_operation,
    text_checksum,
)
from AFAAS.core.tools.builtins.file_operations_utils import (  # FIXME: replace with Langchain
    decode_textual_file,
)
from AFAAS.core.tools.tool import Tool
from AFAAS.core.tools.tool_decorator import SAFE_MODE, tool
from AFAAS.interfaces.agent.main import BaseAgent
from AFAAS.lib.sdk.errors import DuplicateOperationError
from AFAAS.lib.sdk.logger import AFAASLogger
from AFAAS.lib.task.task import Task
from AFAAS.lib.utils.json_schema import JSONSchema


@tool(
    name="copy_file",
    description="Copy a file from one location to another",
    parameters={
        "source_path": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The path of the source file",
            required=True,
        ),
        "destination_path": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The path of the destination file",
            required=True,
        ),
    },
)
def copy_file(
    source_path: Path, destination_path: Path, task: Task, agent: BaseAgent
) -> str:
    """Copy a file from source to destination

    Args:
        source_path (Path): The path of the source file
        destination_path (Path): The path of the destination file

    Returns:
        str: A message indicating success or failure
    """
    try:
        shutil.copyfile(source_path, destination_path)
        return f"File {source_path} has been copied to {destination_path} successfully."
    except Exception as e:
        return f"Error occurred while copying file: {e}"


@tool(
    name="move_file",
    description="Move a file from one location to another",
    parameters={
        "source_path": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The path of the source file",
            required=True,
        ),
        "destination_path": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The path of the destination file",
            required=True,
        ),
    },
)
def move_file(
    source_path: Path, destination_path: Path, task: Task, agent: BaseAgent
) -> str:
    """Move a file from source to destination

    Args:
        source_path (Path): The path of the source file
        destination_path (Path): The path of the destination file

    Returns:
        str: A message indicating success or failure
    """
    try:
        shutil.move(str(source_path), str(destination_path))
        return f"File {source_path} has been moved to {destination_path} successfully."
    except Exception as e:
        return f"Error occurred while moving file: {e}"


@tool(
    name="delete_file",
    description="Delete a file",
    parameters={
        "file_path": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The path of the file to be deleted",
            required=True,
        ),
    },
)
def delete_file(file_path: Path, task: Task, agent: BaseAgent) -> str:
    """Delete a file

    Args:
        file_path (Path): The path of the file to be deleted

    Returns:
        str: A message indicating success or failure
    """
    try:
        if file_path.is_dir():
            shutil.rmtree(file_path)
        else:
            os.remove(file_path)
        return f"File {file_path} has been deleted successfully."
    except Exception as e:
        return f"Error occurred while deleting file: {e}"


@tool(
    name="create_link",
    description="Create a hard link to a file",
    parameters={
        "source_path": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The path of the source file",
            required=True,
        ),
        "link_path": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The path for the new hard link",
            required=True,
        ),
    },
)
def create_link(
    source_path: Path, link_path: Path, task: Task, agent: BaseAgent
) -> str:
    """Create a hard link for a file

    Args:
        source_path (Path): The path of the source file
        link_path (Path): The path for the new hard link

    Returns:
        str: A message indicating success or failure
    """
    try:
        os.link(source_path, link_path)
        return f"Hard link created for {source_path} at {link_path}."
    except Exception as e:
        return f"Error occurred while creating hard link: {e}"


@tool(
    name="create_symlink",
    description="Create a symbolic link to a file or directory",
    parameters={
        "source_path": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The path of the source file or directory",
            required=True,
        ),
        "link_path": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The path for the new symbolic link",
            required=True,
        ),
    },
)
def create_symlink(
    source_path: Path, link_path: Path, task: Task, agent: BaseAgent
) -> str:
    """Create a symbolic link for a file or directory

    Args:
        source_path (Path): The path of the source file or directory
        link_path (Path): The path for the new symbolic link

    Returns:
        str: A message indicating success or failure
    """
    try:
        os.symlink(source_path, link_path)
        return f"Symbolic link created for {source_path} at {link_path}."
    except Exception as e:
        return f"Error occurred while creating symbolic link: {e}"


@tool(
    name="synchronize_directories",
    description="Synchronize the contents of two directories",
    parameters={
        "source_directory": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The source directory",
            required=True,
        ),
        "destination_directory": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The destination directory",
            required=True,
        ),
    },
)
def synchronize_directories(
    source_directory: Path, destination_directory: Path, task: Task, agent: BaseAgent
) -> str:
    """Synchronize two directories

    Args:
        source_directory (Path): The source directory
        destination_directory (Path): The destination directory

    Returns:
        str: A message indicating success or failure
    """
    try:
        from dirsync import sync

        sync(source_directory, destination_directory, "sync", verbose=True)
        return f"Directories {source_directory} and {destination_directory} have been synchronized."
    except Exception as e:
        return f"Error occurred while synchronizing directories: {e}"


@tool(
    name="change_permissions",
    description="Change the permissions of a file or directory",
    parameters={
        "path": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The path of the file or directory",
            required=True,
        ),
        "mode": JSONSchema(
            type=JSONSchema.Type.INTEGER,
            description="The new permission mode (e.g., 0o755)",
            required=True,
        ),
    },
)
def change_permissions(path: Path, mode: int, task: Task, agent: BaseAgent) -> str:
    """Change the permissions of a file or directory

    Args:
        path (Path): The path of the file or directory
        mode (int): The new permission mode (e.g., 0o755)

    Returns:
        str: A message indicating success or failure
    """
    try:
        os.chmod(path, mode)
        return f"Permissions of {path} changed to {oct(mode)}."
    except Exception as e:
        return f"Error occurred while changing permissions: {e}"


@tool(
    name="change_ownership",
    description="Change the owner and group of a file or directory",
    parameters={
        "path": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The path of the file or directory",
            required=True,
        ),
        "uid": JSONSchema(
            type=JSONSchema.Type.INTEGER,
            description="The user ID of the new owner",
            required=True,
        ),
        "gid": JSONSchema(
            type=JSONSchema.Type.INTEGER,
            description="The group ID of the new group",
            required=True,
        ),
    },
)
def change_ownership(
    path: Path, uid: int, gid: int, task: Task, agent: BaseAgent
) -> str:
    """Change the owner and group of a file or directory

    Args:
        path (Path): The path of the file or directory
        uid (int): The user ID of the new owner
        gid (int): The group ID of the new group

    Returns:
        str: A message indicating success or failure
    """
    try:
        os.chown(path, uid, gid)
        return f"Ownership of {path} changed to user ID {uid} and group ID {gid}."
    except Exception as e:
        return f"Error occurred while changing ownership: {e}"
