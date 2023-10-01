"""
web content abilities
"""
import requests
import os
from bs4 import BeautifulSoup

from forge.sdk.memory.memstore import ChromaMemStore

from .registry import ability

def add_memory(task_id: str, document: str, ability_name: str) -> None:
    chromadb_path = f"{os.getenv('AGENT_WORKSPACE')}/{task_id}"
    memory = ChromaMemStore(chromadb_path)
    memory.add(
        task_id=task_id,
        document=document,
        metadatas={"function": ability_name}
    )

@ability(
    name="html_to_file",
    description="get html from website and output to file",
    parameters=[
        {
            "name": "url",
            "description": "Website's url",
            "type": "string",
            "required": True,
        },
        {
            "name": "file_path",
            "description": "Path to the file",
            "type": "string",
            "required": True,
        },
    ],
    output_type="None",
)
async def html_to_file(agent, task_id: str, url: str, file_path: str) -> None:
    """
    html_to_file

    takes a string URL and returns HTML
    then writes HTML to file
    """
    try:
        req = requests.get(url)
        data = req.text.encode()

        agent.workspace.write(task_id=task_id, path=file_path, data=data)

        await agent.db.create_artifact(
            task_id=task_id,
            file_name=file_path.split("/")[-1],
            relative_path=file_path,
            agent_created=True,
        )

        add_memory(task_id, data, "html_to_file")
    except Exception as e:
        raise e


@ability(
    name="html_to_text_file",
    description="get html from website, convert it to text and output to file",
    parameters=[
        {
            "name": "url",
            "description": "Website's url",
            "type": "string",
            "required": True,
        },
        {
            "name": "file_path",
            "description": "Path to the file",
            "type": "string",
            "required": True,
        },
    ],
    output_type="None",
)
async def html_to_text_file(agent, task_id: str, url: str, file_path: str) -> None:
    """
    html_to_text_file

    takes a string URL and returns HTML
    then removes html and writes text to file
    """
    try:
        req = requests.get(url)

        html_soap = BeautifulSoup(req.text, "html.parser")

        agent.workspace.write(
            task_id=task_id, path=file_path, data=html_soap.get_text().encode()
        )

        await agent.db.create_artifact(
            task_id=task_id,
            file_name=file_path.split("/")[-1],
            relative_path=file_path,
            agent_created=True,
        )

        add_memory(task_id, html_soap.get_text(), "html_to_text_file")
    except Exception as e:
        raise e

