from __future__ import annotations
from ..registry import action
from forge.sdk import ForgeLogger, PromptEngine, Agent, LocalWorkspace
from forge.llm import chat_completion_request
import os
import subprocess
import json
from typing import Dict
from .models import Code, TestCase

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
    output_type="TestCase object",
)
async def generate_test_cases(agent: Agent, task_id: str, code_dict: Dict[str, str]) -> TestCase:
    try:
        prompt_engine = PromptEngine("gpt-3.5-turbo")
        messages = [
            {"role": "system", "content": "You are a code generation assistant specialized in generating test cases."}]

        test_prompt_template, test_struct_template, folder_name = determine_templates(
            next(iter(code_dict)))
        if not test_prompt_template:
            return "Unsupported file type."

        code = Code(code_dict)
        for file_name, code_content in code.items():
            LOG.info(f"File Name: {file_name}")
            LOG.info(f"Code: {code_content}")
            test_prompt = prompt_engine.load_prompt(
                test_prompt_template, file_name=file_name, code=code_content)
            messages.append({"role": "user", "content": test_prompt})

        test_struct_prompt = prompt_engine.load_prompt(test_struct_template)
        messages.append({"role": "user", "content": test_struct_prompt})

        response_content = await get_chat_response(messages)
        LOG.info(f"Response content: {response_content}")

        project_path = get_project_path(agent, task_id, folder_name)
        os.makedirs(project_path, exist_ok=True)

        test_cases = parse_test_cases_response(response_content)
        await write_test_cases(agent, task_id, project_path, test_cases)

        return test_cases

    except Exception as e:
        LOG.error(f"Error generating test cases: {e}")
        return "Failed to generate test cases due to an error."


def determine_templates(first_file_name: str):
    if first_file_name.endswith(('.js', '.ts')):
        return "test-case-generation-frontend", "test-case-struct-return-frontend", 'frontend/tests'
    elif first_file_name.endswith('.rs'):
        return "test-case-generation", "test-case-struct-return", 'rust/tests'
    else:
        LOG.error(f"Unsupported file type for: {first_file_name}")
        return None, None, None


async def get_chat_response(messages: list) -> str:
    chat_completion_kwargs = {
        "messages": messages,
        "model": "gpt-3.5-turbo",
    }
    chat_response = await chat_completion_request(**chat_completion_kwargs)
    return chat_response["choices"][0]["message"]["content"]


def get_project_path(agent: Agent, task_id: str, folder_name: str) -> str:
    base_path = agent.workspace.base_path if isinstance(
        agent.workspace, LocalWorkspace) else str(agent.workspace.base_path)
    return os.path.join(base_path, task_id, folder_name)


async def write_test_cases(agent: Agent, task_id: str, project_path: str, test_cases: TestCase):
    for file_name, test_case in test_cases.items():
        test_file_path = os.path.join(project_path, file_name)
        await agent.abilities.run_action(task_id, "write_file", file_path=test_file_path, data=test_case.encode())


def parse_test_cases_response(response_content: str) -> TestCase:
    try:
        json_start = response_content.index('{')
        json_end = response_content.rindex('}') + 1
        json_content = response_content[json_start:json_end]

        LOG.info(f"JSON Content: {json_content}")

        response_dict = json.loads(json_content)
        file_name = response_dict["file_name"]
        test_file = response_dict["test_file"].replace(
            '\\n', '\n').replace('\\t', '\t').strip().strip('"')

        return TestCase({file_name: test_file})
    except (json.JSONDecodeError, ValueError) as e:
        LOG.error(f"Error decoding JSON response: {e}")
        raise

