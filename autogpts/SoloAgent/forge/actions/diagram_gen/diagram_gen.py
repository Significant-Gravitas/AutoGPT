from __future__ import annotations
from ..registry import action
from forge.sdk import ForgeLogger, PromptEngine, Agent
import requests
import os
from forge.actions.code_gen.models import Code
from forge.llm import chat_completion_request

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
            "type": "Code object",
            "required": False
        }
    ],
    output_type="str"
)
async def generate_architecture_diagram(agent: Agent, task_id: str, specification: str, code: Code) -> str:
    try:
        prompt_engine = PromptEngine("gpt-3.5-turbo")
        diagram_prompt = prompt_engine.load_prompt(
            "diagram-prompt", specification=specification, code=code
        )

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
        response.raise_for_status()

        result = response.json()
        LOG.info(f"Diagram generated successfully: {result['fileUrl']}")
        return result['fileUrl']
    except requests.RequestException as e:
        LOG.error(f"Error generating diagram: {e}")
        return "Failed to generate diagram."


@action(
    name="gen-use-case-diagram",
    description="Generate a use case diagram from analyzing the code.",
    parameters=[
        {
            "name": "code",
            "description": "A dictionary containing filenames and code for a codebase.",
            "type": "Code object",
            "required": True
        },

        {
            "name": "specification",
            "description": "Specification of the project.",
            "type": "str",
            "required": True
        }
    ],
    output_type="str"
)
async def generate_use_case_diagram(agent: Agent, task_id: str, code: Code, specification: str) -> str:
    try:
        prompt_engine = PromptEngine("gpt-3.5-turbo")
        usecase_diagram_template = prompt_engine.load_prompt(
            "use-case-diagram-gen-return", code=code)

        messages = [
            {"role": "system", "content": "You are a code generation assistant specialized in generating test cases."},
            {"role": "system", "content": usecase_diagram_template}
        ]

        chat_completion_kwargs = {
            "messages": messages,
            "model": "gpt-3.5-turbo",
        }

        chat_response = await chat_completion_request(**chat_completion_kwargs)

        LOG.info(f"Response content: {chat_response}")
        return chat_response['choices'][0]['message']['content']
    except Exception as e:
        LOG.error(f"Error generating use case diagram: {e}")
        return "Failed to generate use case diagram."

