"""
web content abilities
"""
import requests
import os
from bs4 import BeautifulSoup

from forge.sdk.memory.memstore_tools import add_memory

from ..forge_log import ForgeLogger
from .registry import ability

logger = ForgeLogger(__name__)

# @ability(
#     name="html_to_file",
#     description="get html from website and output to file",
#     parameters=[
#         {
#             "name": "url",
#             "description": "Website's url",
#             "type": "string",
#             "required": True,
#         },
#         {
#             "name": "file_path",
#             "description": "Path to the file",
#             "type": "string",
#             "required": True,
#         },
#     ],
#     output_type="None",
# )
# async def html_to_file(agent, task_id: str, url: str, file_path: str) -> None:
#     """
#     html_to_file

#     takes a string URL and returns HTML
#     then writes HTML to file
#     """
#     try:
#         req = requests.get(url)
#         data = req.text.encode()

#         agent.workspace.write(task_id=task_id, path=file_path, data=data)

#         await agent.db.create_artifact(
#             task_id=task_id,
#             file_name=file_path.split("/")[-1],
#             relative_path=file_path,
#             agent_created=True,
#         )

#         add_memory(task_id, data, "html_to_file")
#     except Exception as e:
#         logger.error(f"html_to_file failed: {e}")


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
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'}

        req = requests.get(
            url=url,
            headers=headers
        )

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
        logger.error(f"html_to_text_file failed: {e}")

