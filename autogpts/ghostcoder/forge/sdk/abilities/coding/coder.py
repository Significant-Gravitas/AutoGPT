import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Tuple, Optional

from forge.sdk import ForgeLogger
from forge.sdk.abilities.registry import ability
from ghostcoder.codeblocks import create_parser, CodeBlockType
from ghostcoder.filerepository import FileRepository
from ghostcoder.actions import CodeWriter
from ghostcoder.actions.write_code.base import OutputFormat
from ghostcoder.benchmark.utils import create_openai_client
from ghostcoder.schema import Message, TextItem, FileItem, CodeItem
from ghostcoder.test_tools.verify_python_pytest import PythonPytestTestTool

logger = ForgeLogger(__name__)

@ability(
    name="verify_code",
    disabled=True,
    description="Use to verify code by running all unit tests.",
    parameters=[
        {
            "name": "test_file_pattern",
            "description": "File pattern to test files to run, set to \"*.py\" to run all tests.",
            "type": "string",
            "required": False,
        }
    ],
    output_type="string",
)
async def verify_code(
        agent,
        task_id: str,
        step_id: str,
        test_file_pattern: str = "*.py") -> Tuple[bool, str]:
    test_tool = PythonPytestTestTool(current_dir=agent.workspace.base_path / task_id, test_file_pattern=test_file_pattern)
    verification_result = test_tool.run_tests()

    output = ""
    if not verification_result.success:
        output += "\n\n".join([item.to_prompt() for item in verification_result.failures])
        output += f"\n\n{verification_result.verification_count} tests failed!"
    elif verification_result.verification_count > 0:
        output = f"\n\n{verification_result.verification_count} tests passed!"
    else:
        output = f"\n\nNo tests found!"

    return verification_result.success, output


@ability(
    name="write_code",
    disabled=True,
    description="Use this to write code and tests. Provide a file that should be implemented and a test file that should test the implementation.",
    parameters=[
        {
            "name": "file_path",
            "description": "File path to the file that should be updated or created.",
            "type": "string",
            "required": False,
        },
        {
            "name": "code",
            "description": "The code that should be updated or created.",
            "type": "string",
            "required": False,
        },
        {
            "name": "test_file_path",
            "description": "File path to the test file that should be updated or created",
            "type": "string",
            "required": False,
        },
        {
            "name": "test_code",
            "description": "The test code that should be updated or created.",
            "type": "string",
            "required": False,
        },
    ],
    output_type="string",
)
async def write_code(
        agent,
        task_id: str,
        step_id: str,
        file_path: Optional[str] = None,
        test_file_path: Optional[str] = None,
        code: Optional[str] = None,
        test_code: Optional[str] = None,
) -> Tuple[bool, str]:
    task = await agent.db.get_task(task_id)

    retry_inputs = ""
    if file_path:
        existing_code = read_file(agent, task_id, file_path)
        if existing_code:
            code, retry_input = updated_file(code, existing_code)
            if retry_input:
                retry_inputs += retry_input

    if test_file_path:
        existing_test_code = read_file(agent, task_id, test_file_path)
        if existing_test_code:
            test_code, retry_input = updated_file(test_code, existing_test_code)
            if retry_input:
                retry_inputs += retry_input

    if retry_inputs:
        return False, retry_inputs

    if test_code:
        agent.workspace.write(task_id=task_id, path=file_path, data=code)

        # FIXME: Not adding the test file to artifacts to avoid test failures not in the actual benchmark
        # await agent.db.create_artifact(
        #     task_id=task_id,
        #     file_name=test_file,
        #     relative_path="",
        #     agent_created=True,
        # )

    if code:
        agent.workspace.write(task_id=task_id, path=file_path, data=code)

        await agent.db.create_artifact(
            task_id=task_id,
            file_name=file_path,
            relative_path="",
            agent_created=True,
        )

    test_tool = PythonPytestTestTool(current_dir=agent.workspace.base_path / task_id, test_file_pattern=test_file_path)
    verification_result = test_tool.run_tests()

    output = ""
    if not verification_result.success:
        output += "\n\n".join([item.to_prompt() for item in verification_result.failures])
        output += "\n\nThe tests failed!"
    else:
        output = "\n\nThe files was implemented and the tests passed!"

    return verification_result.success, output


def read_file(agent, task_id, file_path: str) -> str:
    data = agent.workspace.read(task_id=task_id, path=file_path)
    if data is None:
        return None

    if isinstance(data, bytes):
        data = data.decode("utf-8")

    return data


def updated_file(updated_content: str, existing_content: str) -> Tuple[Optional[str], Optional[str]]:
    parser = create_parser("python")

    retry_input = ""
    try:
        updated_block = parser.parse(updated_content)
    except Exception as e:
        return None, f"You returned a file with the following error: {e}"

    error_blocks = updated_block.find_errors()
    if error_blocks:
        logger.info("The updated file has errors. ")
        retry_input += "You returned a file with syntax errors. Find them and correct them."
        for error_block in error_blocks:
            retry_input += "```\n" + error_block.to_string() + "\n```"
    else:
        original_block = parser.parse(existing_content)

        updated_content = original_block.to_string()
        merged_block = parser.parse(updated_content)
        error_blocks = merged_block.find_errors()
        if error_blocks:
            logger.info("The merged file has errors..")
            retry_input += "You returned a file with syntax errors. Find them and correct them."
            for error_block in error_blocks:
                retry_input += "```\n" + error_block.to_string() + "\n```"

    return updated_content, retry_input
