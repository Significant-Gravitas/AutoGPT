import logging
import time
from pathlib import Path
from typing import Tuple

from forge.sdk.abilities.registry import ability
from ghostcoder import FileRepository
from ghostcoder.actions import CodeWriter
from ghostcoder.actions.write_code.base import OutputFormat
from ghostcoder.actions.write_code.prompt import FIX_TESTS_PROMPT
from ghostcoder.benchmark.utils import create_openai_client
from ghostcoder.schema import Message, TextItem, FileItem
from ghostcoder.test_tools.verify_python_pytest import PythonPytestTestTool

smart_llm_name = "gpt-4"
basic_llm_name = "gpt-3.5-turbo"

DEFAULT_PROMPT = """You're an AI Developer with superior programming skills.
You can both update the existing files and add new ones if needed. 
Please exclude files that have not been updated in response.
"""


@ability(
    name="write_code",
    description="Use this to write code and tests. Provide a file that should be implemented and a test file that should test the implementation.",
    parameters=[
        {
            "name": "file",
            "description": "File path to the file that should be updated or created.",
            "type": "string",
            "required": True,
        },
        {
            "name": "test_file",
            "description": "File path to the test file that should be updated or created",
            "type": "string",
            "required": True,
        }
    ],
    output_type="None",
)
async def write_code(
    agent,
    task_id: str,
    file: str,
    test_file: str,
) -> str:
    task = await agent.db.get_task(task_id)
    repo_dir = agent.workspace.base_path / task_id
    llm = create_openai_client(log_dir=repo_dir / ".prompt_log", llm_name=basic_llm_name, temperature=0.0, streaming=False)

    repository = FileRepository(repo_path=repo_dir, use_git=False)

    code_writer = CodeWriter(llm=llm,
                             repository=repository,
                             sys_prompt=DEFAULT_PROMPT,
                             output_format=OutputFormat.TEXT,
                             allow_hallucinated_files=True,
                             auto_mode=True)

    test_fix_writer = CodeWriter(llm=llm,
                                 sys_prompt=FIX_TESTS_PROMPT,
                                 output_format=OutputFormat.TEXT,
                                 repository=repository,
                                 auto_mode=True)

    test_tool = PythonPytestTestTool(current_dir=repository.repo_path, test_file_pattern=test_file)
    test_file_item = FileItem(file_path=file, content=None)
    file_item = FileItem(file_path=test_file, content=None)

    message = Message(sender="Human", items=[
        TextItem(text=task.input),
        test_file_item,
        file_item
    ])

    outgoing_messages = code_writer.execute(incoming_messages=[message])

    await agent.db.create_artifact(
        task_id=task_id,
        file_name=file,
        relative_path="",
        agent_created=True,
    )

    await agent.db.create_artifact(
        task_id=task_id,
        file_name=test_file,
        relative_path="",
        agent_created=True,
    )

    verification_result = test_tool.run_tests()

    output = outgoing_messages[-1].to_prompt()

    if not verification_result.success:
        output += "\n".join([item.to_prompt() for item in verification_result.failures])
        output += "\nThe tests failed!"
    else:
        output += "\nThe tests passed!"

    return output
