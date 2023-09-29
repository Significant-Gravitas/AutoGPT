"""
Ability to read a website
"""
import requests

from ..registry import ability

@ability(
    name="read_website",
    description="read website content",
    parameters= [
        {
            "name": "url",
            "description": "Website's url",
            "type": "string",
            "required": True
        }
    ],
    output_type="str"
)

async def read_website(
    agent,
    task_id: str,
    url: str
) -> str:
    """
    read_website

    takes a string URL and returns HTML
    """
    try:
        req = requests.get(url)
        return req.text
    except Exception as e:
        raise e