from typing import List

from ..registry import ability
from ... import ForgeLogger

LOG = ForgeLogger(__name__)

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
    output_type="list[string]",
)
async def list_files(agent, task_id: str, path: str) -> List[str]:
    """
    List files in a workspace directory
    """
    files = agent.workspace.list(task_id=task_id, path=path)
    LOG.debug(f"List {len(files)} files in {path}")
    return files


@ability(
    name="write_file",
    description="Write data to a file",
    parameters=[
        {
            "name": "file_path",
            "description": "Path to the file",
            "type": "string",
            "required": True,
        },
        {
            "name": "data",
            "description": "Data to write to the file",
            "type": "string",
            "required": True,
        },
    ],
    output_type="string",
)
async def write_file(agent, task_id: str, file_path: str, data: bytes) -> str:
    """
    Write data to a file
    """
    if isinstance(data, str):
        data = data.encode()

    agent.workspace.write(task_id=task_id, path=file_path, data=data)
    artifact = await agent.db.create_artifact(
        task_id=task_id,
        file_name=file_path.split("/")[-1],
        relative_path=file_path,
        agent_created=True,
    )

    LOG.debug(f"Wrote data to file {file_path} and created artifact {artifact.artifact_id}")

    return f"Wrote data to file {file_path}"


@ability(
    name="read_file",
    description="Read data from a file",
    parameters=[
        {
            "name": "file_path",
            "description": "Path to the file",
            "type": "string",
            "required": True,
        },
    ],
    output_type="string",
)
async def read_file(agent, task_id: str, file_path: str) -> str:
    """
    Read data from a file
    """
    data = agent.workspace.read(task_id=task_id, path=file_path)
    if data is None:
        LOG.info(f"No file found on path {file_path}")
        return f"No file found on path {file_path}"

    if isinstance(data, bytes):
        data = data.decode("utf-8")

    LOG.debug(f"Read file {file_path}")
    return data
