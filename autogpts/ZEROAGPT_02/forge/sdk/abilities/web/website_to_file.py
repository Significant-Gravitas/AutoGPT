"""
Export website to a file
"""
import requests

from ..registry import ability

@ability(
    name="website_to_file",
    description="get website content and output to file",
    parameters= [
        {
            "name": "url",
            "description": "Website's url",
            "type": "string",
            "required": True
        },
        {
            "name": "file_path",
            "description": "Path to the file",
            "type": "string",
            "required": True,
        },
    ],
    output_type="str"
)

async def website_to_file(
    agent,
    task_id: str,
    url: str,
    file_path: str
) -> None:
    """
    website_to_file

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
    except Exception as e:
        raise e
