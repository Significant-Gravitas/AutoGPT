from __future__ import annotations
from ..registry import action
from forge.sdk import ForgeLogger, PromptEngine
import requests
import os

LOG = ForgeLogger(__name__)
API_KEY = os.getenv("ERASERIO_API_KEY")


@action(
    name="gen-diagram-eraserio",
    description="Generate a code diagram using eraser.io",
    parameters=[
        {
            "name": "specification",
            "description": "Code specification",
            "type": "string",
            "required": True,
        },
        {
            "name": "code",
            "description": "Code generated from the specification",
            "type": "string",
            "required": False
        }
    ],
    output_type="str"
)
async def generate_architecture_diagram(agent, task_id: str, specification: str, code: str) -> str:
    prompt_engine = PromptEngine("gpt-3.5-turbo")
    diagram_prompt = prompt_engine.load_prompt(
        "diagram-prompt", specification=specification, code=code)

    url = "https://app.eraser.io/api/render/prompt"

    payload = {
        "text": diagram_prompt,
        "diagramType": "sequence-diagram",
        "background": True,
        "theme": "light",
        "scale": "1",
        "returnFile": True
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        result = response.json()
        LOG.info(f"Diagram generated successfully: {result['fileUrl']}")
        return result['fileUrl']
    else:
        LOG.error(f"Error generating diagram: {response.text}")
        return "Failed to generate diagram."


async def generate_use_case_diagram(code):
    pass
