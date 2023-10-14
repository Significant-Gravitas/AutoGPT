import logging
from typing import Tuple

from forge.sdk.abilities.registry import ability
from ghostcoder.filerepository import FileRepository
from ghostcoder.actions import CodeWriter
from ghostcoder.actions.write_code.base import OutputFormat
from ghostcoder.benchmark.utils import create_openai_client
from ghostcoder.schema import Message, TextItem, FileItem
from ghostcoder.test_tools.verify_python_pytest import PythonPytestTestTool

smart_llm_name = "gpt-4"
basic_llm_name = "gpt-3.5-turbo"

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('openai').setLevel(logging.INFO)
logging.getLogger('urllib3').setLevel(logging.INFO)

DEFAULT_PROMPT = """You're an AI Developer with superior programming skills.
You're tasked to write an implementation based on the provided task. 
You should also write tests for the implementation. Make sure to write tests for all requirements.
"""

FIX_TESTS_PROMPT = """You're a experienced programmer that will review a solution written by an inexperienced programmer. 
The tests failed and you need to help the programmer to fix the code.
"""

FILE_FORMAT = """All files should be presented in the following format:

/file.py
```python
# ... code  
```
"""

@ability(
    name="fix_code",
    description="Use this to fix failing tests. Provide a file that should be implemented and a test file that should test the implementation.",
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
        },
        {
            "name": "instructions",
            "description": "Instruction on what to fix",
            "type": "string",
            "required": True,
        }
    ],
    output_type="string",
)
async def fix_code(
        agent,
        task_id: str,
        instructions: str,
        file: str,
        test_file: str) -> Tuple[bool, str]:
    return await _write_code(agent, task_id, file, test_file, instructions)


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
    output_type="string",
)
async def write_code(
        agent,
        task_id: str,
        file: str,
        test_file: str) -> Tuple[bool, str]:
    return await _write_code(agent, task_id, file, test_file)


async def _write_code(
        agent,
        task_id: str,
        file: str,
        test_file: str,
        fix_code_instructions: str = None,
) -> Tuple[bool, str]:
    task = await agent.db.get_task(task_id)
    repo_dir = agent.workspace.base_path / task_id
    llm = create_openai_client(log_dir=repo_dir / ".prompt_log", llm_name=smart_llm_name, temperature=0.0,
                               streaming=False)

    repository = FileRepository(repo_path=repo_dir, use_git=False)

    if fix_code_instructions:
        system_prompt = FIX_TESTS_PROMPT + FILE_FORMAT
    else:
        system_prompt = DEFAULT_PROMPT + FILE_FORMAT

    code_writer = CodeWriter(llm=llm,
                             repository=repository,
                             sys_prompt=system_prompt,
                             output_format=OutputFormat.TEXT,
                             allow_hallucinated_files=True,
                             auto_mode=True)

    # FIXME: Allow null or retry
    if not test_file:
        test_file = "test_" + file

    test_tool = PythonPytestTestTool(current_dir=repository.repo_path, test_file_pattern=test_file)
    test_file_item = FileItem(file_path=test_file, content=repository.get_file_content(test_file))
    file_item = FileItem(file_path=file, content=repository.get_file_content(file))

    if fix_code_instructions:
        messages = [
            Message(sender="Human", items=[TextItem(text=task.input)]),
            Message(sender="AI", items=[file_item, test_file_item]),
            Message(sender="Human", items=[TextItem(text=fix_code_instructions)])
        ]
    else:
        messages = [Message(sender="Human", items=[
            TextItem(text=task.input),
            test_file_item,
            file_item
        ])]

    outgoing_messages = code_writer.execute(incoming_messages=messages)

    await agent.db.create_artifact(
        task_id=task_id,
        file_name=file,
        relative_path="",
        agent_created=True,
    )

    # FIXME: Not adding the test file to artifacts to avoid test failures not in the actual benchmark
    # await agent.db.create_artifact(
    #     task_id=task_id,
    #     file_name=test_file,
    #     relative_path="",
    #     agent_created=True,
    # )

    verification_result = test_tool.run_tests()

    text_items = outgoing_messages[-1].find_items_by_type("text")

    output = "\n\n".join([item.text for item in text_items])

    if not verification_result.success:
        output += "\n\n".join([item.to_prompt() for item in verification_result.failures])
        output += "\n\nThe tests failed!"
    else:
        output = "\n\nThe files was implemented and the tests passed!"

    return verification_result.success, output
