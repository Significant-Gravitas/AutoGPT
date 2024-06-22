
from __future__ import annotations


from ..registry import action
from forge.sdk import ForgeLogger, PromptEngine
from forge.llm import chat_completion_request

LOG = ForgeLogger(__name__)


@action(
    name="generate_solana_code",
    description="Generate Solana on-chain code based on the provided specification",
    parameters=[
        {
            "name": "specification",
            "description": "Code specification",
            "type": "string",
            "required": True,
        }
    ],
    output_type="str",
)
async def generate_solana_code(agent, task_id: str, specification: str) -> str:
    """Generate Solana on-chain code based on the provided specification.

    Args:
        specification (str): The code specification.

    Returns:
        str: The generated Solana code.
    """
    prompt_engine = PromptEngine("gpt-3.5-turbo")

    # Load and populate the Solana code generation prompt
    code_prompt = prompt_engine.load_prompt(
        "solana-code-generation", specification=specification)

    messages = [
        {"role": "system", "content": "You are a code generation assistant specialized in Solana smart contracts."},
        {"role": "user", "content": code_prompt},
    ]

    try:
        chat_completion_kwargs = {
            "messages": messages,
            "model": "gpt-3.5-turbo",
        }
        chat_response = await chat_completion_request(**chat_completion_kwargs)
        generated_code = chat_response["choices"][0]["message"]["content"]

        LOG.info(f"Generated Solana code: {generated_code}")

        return generated_code

    except Exception as e:
        LOG.error(f"Error generating Solana code: {e}")
        return "Error generating Solana code"


@action(
    name="generate_frontend_code",
    description="Generate frontend code based on the provided specification",
    parameters=[
        {
            "name": "specification",
            "description": "Frontend code specification",
            "type": "string",
            "required": True,
        }
    ],
    output_type="str",
)
async def generate_frontend_code(agent, task_id: str, specification: str) -> str:
    """ Generate frontend code based on the provided ui specification or project specification

       Args: specification (str): The UI specification OR project specification


        Returns
    """
    prompt_engine = PromptEngine("gpt-3.5-turbo")
    code_prompt = prompt_engine.load_prompt(
        "frontend-code-generation", specification=specification)
    messages = [
        {"role": "system", "content": "You are a code generation assistant specialized in frontend development."},
        {"role": "user", "content": code_prompt},
    ]

    chat_completion_kwargs = {
        "messages": messages,
        "model": "gpt-3.5-turbo",
    }
    chat_response = await chat_completion_request(**chat_completion_kwargs)
    generated_code = chat_response["choices"][0]["message"]["content"]
    return generated_code

