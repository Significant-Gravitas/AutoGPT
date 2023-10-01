from typing import List, Match
import re
import os

from forge.sdk.memory.memstore import ChromaMemStore

from ...forge_log import ForgeLogger
from ..registry import ability

logger = ForgeLogger(__name__)

def add_memory(task_id: str, document: str, ability_name: str) -> None:
    logger.info(f"🧠 Adding ability '{ability_name}' memory for task {task_id}")
    chromadb_path = f"{os.getenv('AGENT_WORKSPACE')}/{task_id}"
    memory = ChromaMemStore(chromadb_path)
    memory.add(
        task_id=task_id,
        document=document,
        metadatas={"function": ability_name}
    )

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
    name="write_file",
    description="Write data to a file",
    parameters=[
        {
            "name": "file_path",
            "description": "Path to the file including file name",
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
async def write_file(agent, task_id: str, file_path: str, data: bytes) -> None:
    """
    Write data to a file
    """
    if isinstance(data, str):
        data = data.encode()

    agent.workspace.write(task_id=task_id, path=file_path, data=data)
    
    await agent.db.create_artifact(
        task_id=task_id,
        file_name=file_path.split("/")[-1],
        relative_path=file_path,
        agent_created=True,
    )

    add_memory(task_id, str(data), "write_file")

@ability(
    name="read_file",
    description="Read data from a file",
    parameters=[
        {
            "name": "file_path",
            "description": "Path to the file including file name",
            "type": "string",
            "required": True,
        },
    ],
    output_type="bytes",
)
async def read_file(agent, task_id: str, file_path: str) -> bytes:
    """
    Read data from a file
    """
    return agent.workspace.read(task_id=task_id, path=file_path)

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
            "name": "file_path",
            "description": "Path to the file including file name",
            "type": "string",
            "required": True,
        }
    ],
    output_type="list[match]"
)
async def search_file(agent, task_id: str, file_path: str, regex: str) -> List[Match]:
    """
    Search file using regex
    """
    open_file = agent.workspace.read(task_id=task_id, path=file_path)

    search_rgx = re.findall(rf"{regex}", open_file.decode())

    return search_rgx

