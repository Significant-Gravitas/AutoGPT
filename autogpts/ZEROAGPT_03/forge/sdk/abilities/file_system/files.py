from typing import List, Match
import re
import os

from forge.sdk.memory.memstore_tools import add_memory

from ...forge_log import ForgeLogger
from ..registry import ability

logger = ForgeLogger(__name__)

@ability(
    name="list_files",
    description="List files in a directory",
    parameters=[
        {
            "name": "path",
            "description": "Path to the directory",
            "type": "string",
            "required": True,
        }
    ],
    output_type="list[str]",
)
async def list_files(agent, task_id: str, path: str) -> List[str]:
    """
    List files in a workspace directory
    """
    return agent.workspace.list(task_id=task_id, path=path)

@ability(
    name="write_source_code",
    description="Write programming language source code to a file",
    parameters=[
        {
            "name": "file_name",
            "description": "Name of the file",
            "type": "string",
            "required": True,
        },
        {
            "name": "code",
            "description": "Code to write to the file",
            "type": "string",
            "required": True,
        },
    ],
    output_type="None",
)
async def write_source_code(agent, task_id: str, file_name: str, code: str) -> None:
    """
    Write source code as string/text to a file
    """

    # clean extra escape slashes
    code = code.replace('\\\\', '\\')
    
    # clean \n being written as text and not a new line
    code = code.replace('\\n', '\n')

    agent.workspace.write_str(task_id=task_id, path=file_name, data=code)
    
    await agent.db.create_artifact(
        task_id=task_id,
        file_name=file_name.split("/")[-1],
        relative_path=file_name,
        agent_created=True,
    )

    add_memory(task_id, code, "write_source_code")


@ability(
    name="write_file",
    description="Write data to a file",
    parameters=[
        {
            "name": "file_name",
            "description": "Name of the file",
            "type": "string",
            "required": True,
        },
        {
            "name": "data",
            "description": "Data to write to the file",
            "type": "bytes",
            "required": True,
        },
    ],
    output_type="None",
)
async def write_file(agent, task_id: str, file_name: str, data: bytes) -> None:
    """
    Write data to a file
    """
    if isinstance(data, str):
        # ai adding too many escape back slashes
        data = data.encode()

    agent.workspace.write(task_id=task_id, path=file_name, data=data)
    
    await agent.db.create_artifact(
        task_id=task_id,
        file_name=file_name.split("/")[-1],
        relative_path=file_name,
        agent_created=True,
    )

    add_memory(task_id, str(data), "write_file")

@ability(
    name="read_file",
    description="Read data from a file",
    parameters=[
        {
            "name": "file_name",
            "description": "Name of the file",
            "type": "string",
            "required": True,
        },
    ],
    output_type="bytes",
)
async def read_file(agent, task_id: str, file_name: str) -> bytes:
    """
    Read data from a file
    """
    read_file = agent.workspace.read(task_id=task_id, path=file_name)

    add_memory(task_id, str(read_file), "read_file")
    
    return read_file

@ability(
    name="search_file",
    description="Search data from a file using regex",
    parameters=[
        {
            "name": "regex",
            "description": "regular expression for searching file",
            "type": "string",
            "required": True
        },
        {
            "name": "file_name",
            "description": "Name of file",
            "type": "string",
            "required": True,
        }
    ],
    output_type="list[match]"
)
async def search_file(agent, task_id: str, file_name: str, regex: str) -> List[Match]:
    """
    Search file using regex
    """
    open_file = agent.workspace.read(task_id=task_id, path=file_name)

    search_rgx = re.findall(rf"{regex}", open_file.decode())

    return search_rgx

