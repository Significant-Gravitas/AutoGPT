from __future__ import annotations
from typing import Dict, Any
from ..registry import action
from forge.sdk import ForgeLogger, PromptEngine
from forge.llm import chat_completion_request
import os
from forge.sdk import Agent
import subprocess
import json
from ..models import Code, TestCase

LOG = ForgeLogger(__name__)


@action(
    name="test_code",
    description="Test the generated code for errors",
    parameters=[
        {
            "name": "project_path",
            "description": "Path to the project directory",
            "type": "string",
            "required": True,
        }
    ],
    output_type="str",
)
async def test_code(agent: Agent, task_id: str, project_path: str) -> str:
    try:
        result = subprocess.run(
            ['cargo', 'test'], cwd=project_path, capture_output=True, text=True)

        if result.returncode != 0:
            LOG.error(f"Test failed with errors: {result.stderr}")
            return result.stderr  # Return errors
        else:
            LOG.info(f"All tests passed: {result.stdout}")
            return "All tests passed"

    except Exception as e:
        LOG.error(f"Error testing code: {e}")
        return f"Failed to test code: {e}"


@action(
    name="generate_unit_tests",
    description="Generates unit tests for Solana code.",
    parameters=[
        {
            "name": "code_dict",
            "description": "Dictionary containing file names and respective code generated.",
            "type": "dict",
            "required": True
        }
    ],
    output_type="str",
)
async def generate_test_cases(agent: Agent, task_id: str, code_dict: Dict[str, str]) -> str:
    code_type = Code(code_dict)
    messages = [
        {"role": "system", "content": "You are a code generation assistant specialized in generating test cases."}
    ] + [
        {"role": "user", "content": load_test_prompt(file_name, code)}
        for file_name, code in code_type.items()
    ] + [{"role": "user", "content": load_test_struct_prompt()}]

    response_content = await get_chat_response(messages)

    try:
        test_cases = parse_test_cases_response(response_content)
    except Exception as e:
        LOG.error(f"Error parsing test cases response: {e}")
        return "Failed to generate test cases due to response parsing error."

    project_path = os.path.join(agent.workspace.base_path, task_id)
    await write_test_files(agent, task_id, project_path, TestCase(test_cases))

    return "Test cases generated and written to respective files."


async def get_chat_response(messages: list[dict[str, Any]]) -> str:
    chat_completion_kwargs = {
        "messages": messages,
        "model": "gpt-3.5-turbo",
    }
    chat_response = await chat_completion_request(**chat_completion_kwargs)
    return chat_response["choices"][0]["message"]["content"]


def parse_test_cases_response(response_content: str) -> TestCase:
    try:
        response_dict = json.loads(response_content)
        test_cases = TestCase(
            {response_dict["file_name"]: response_dict["test_file"]})
        return test_cases
    except json.JSONDecodeError as e:
        LOG.error(f"Error decoding JSON response: {e}")
        raise


async def write_code_files(agent: Agent, task_id: str, project_path: str, parts: Code) -> None:
    for file_name, content in parts.items():
        await write_file(agent, task_id, os.path.join(project_path, 'src', file_name), content.encode())


async def write_test_files(agent: Agent, task_id: str, project_path: str, test_cases: TestCase) -> None:
    for file_name, test_case in test_cases.items():
        await write_file(agent, task_id, os.path.join(project_path, 'tests', file_name), test_case.encode())


async def write_file(agent: Agent, task_id: str, file_path: str, data: bytes) -> None:
    await agent.abilities.run_action(
        task_id, "write_file", file_path=file_path, data=data
    )


def load_test_prompt(file_name: str, code: str) -> str:
    prompt_engine = PromptEngine("gpt-3.5-turbo")
    return prompt_engine.load_prompt("test-case-generation", file_name=file_name, code=code)


def load_test_struct_prompt() -> str:
    prompt_engine = PromptEngine("gpt-3.5-turbo")
    return prompt_engine.load_prompt("test-case-struct-return")
