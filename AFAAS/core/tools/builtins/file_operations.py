"""Tools to perform operations on files"""

from __future__ import annotations

import os
import os.path
from pathlib import Path

from langchain_community.tools.file_management.file_search import FileSearchTool
from langchain_core.vectorstores import VectorStore
from AFAAS.core.tools.builtins.file_operations_helpers import is_duplicate_operation, log_operation, text_checksum
from AFAAS.core.tools.builtins.file_operations_utils import decode_textual_file #FIXME: replace with Langchain
from AFAAS.core.tools.tool_decorator import tool
from AFAAS.core.tools.tools import Tool
from AFAAS.interfaces.agent.main import BaseAgent
from AFAAS.lib.sdk.errors import DuplicateOperationError
from AFAAS.lib.sdk.logger import AFAASLogger
from AFAAS.lib.task.task import Task
from AFAAS.lib.utils.json_schema import JSONSchema

TOOL_CATEGORY = "file_operations"
TOOL_CATEGORY_TITLE = "File Operations"


LOG = AFAASLogger(name=__name__)

@tool(
    name="read_file",
    description="Read an existing file",
    parameters={
        "filename": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The path of the file to read",
            required=True,
        )
    },
)
def read_file(filename:  str | Path, task: Task, agent: BaseAgent) -> str:
    """Read a file and return the contents

    Args:
        filename (Path): The name of the file to read

    Returns:
        str: The contents of the file
    """
    file = agent.workspace.open_file(filename, binary=True)
    content = decode_textual_file(file, os.path.splitext(filename)[1])

    return content



@tool(
    name="write_file",
    description="Write a file, creating it if necessary. If the file exists, it is overwritten.",
    parameters={
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
    aliases=["write_file", "create_file"],
)
async def write_to_file(
    filename: Path, contents: str, task: Task, agent: BaseAgent
) -> str:
    """Write contents to a file

    Args:
        filename (Path): The name of the file to write to
        contents (str): The contents to write to the file

    Returns:
        str: A message indicating success or failure
    """
    checksum = text_checksum(contents)
    if is_duplicate_operation(operation="write", file_path=Path(filename), agent=agent, checksum=checksum):
        raise DuplicateOperationError(f"File {filename} has already been updated.")

    if directory := os.path.dirname(filename):
        agent.workspace.get_path(directory).mkdir(exist_ok=True)
    await agent.workspace.write_file(filename, contents)
    log_operation("write", filename, agent, checksum)

    from AFAAS.lib.sdk.artifacts import Artifact
    artifact = Artifact(
        agent_id=agent.agent_id,
        user_id=agent.user_id,
        source="AGENT",
        relative_path=str(filename.parent),
        file_name=str(filename.name),
        mime_type="text/plain",
        license=None,
        checksum=checksum,
    )

    #cf : ingest_file
    # FIXME:v0.1.0 if file exists, delete it first    
    #await agent.vectorstore.adelete(id=str(filename))

    await agent.vectorstore.aadd_texts(texts=[contents],
                                 metadatas=[{"id": str(artifact.artifact_id),
                                            "agent_id": str(artifact.agent_id),
                                            "user_id": str(artifact.user_id),
                                            "relative_path": str(artifact.relative_path),
                                            "file_name": str(artifact.file_name),
                                            "mime_type": str(artifact.mime_type)}
                                            ],
    )                         
    #  ids=[str(filename)],
    #  lang="en",
    #  title=str(filename),
    #  description="",
    #  tags=[],
    #  metadata={},
    #  source="",
    #  author="",
    #  date="",
    #  license="",
    #  url="",
    #  chunk_size=100,
    #  chunk_overlap=0,
    #  chunking_strategy="fixed",
    #  chunking_strategy_args={},
    #  chunking_strategy_kwargs={},

    # Save the artifact metadata in the database
    if await artifact.create_in_db(agent = agent) :
        return f"File {filename} has been written successfully."
    else :
        return f"Ooops ! Something went wrong when writing file {filename}."


@tool(
    name="list_folder",
    description="List the items in a folder",
    parameters={
        "folder": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The folder to list files in",
            required=True,
        )
    },
)
def list_folder(folder: Path, task: Task, agent: BaseAgent) -> list[str]:
    """Lists files in a folder recursively

    Args:
        folder (Path): The folder to search in

    Returns:
        list[str]: A list of files found in the folder
    """
    return [str(p) for p in agent.workspace.list(folder)]



def file_search_args(input_args: dict[str, any], agent: BaseAgent):
    # Force only searching in the workspace root
    input_args["dir_path"] = str(agent.workspace.get_path(input_args["dir_path"]))

    return input_args


file_search = Tool.generate_from_langchain_tool(
    tool=FileSearchTool(), arg_converter=file_search_args
)
