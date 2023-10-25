from pathlib import Path
from typing import Tuple, Optional, List

import tiktoken

from forge.sdk import ForgeLogger, Artifact
from forge.sdk.abilities.coding.write_code import read_other_files
from forge.sdk.abilities.registry import ability
from ghostcoder.filerepository import FileRepository
from ghostcoder.actions import CodeWriter
from ghostcoder.benchmark.utils import create_openai_client
from ghostcoder.schema import Message, TextItem, FileItem, CodeItem

logger = ForgeLogger(__name__)

use_pytest_parser = True

smart_llm_name = "gpt-4"
basic_llm_name = "gpt-3.5-turbo"

_llm = smart_llm_name
_temperature = 0.0

_only_return_changes = False

ROLE_PROMPT = """You're a Staff Engineer with superior python programming skills."""

DEFAULT_PROMPT = """You're tasked to write an implementation based on the provided task. 
Review the requirements and write out your interpretation of the requirements and then do the full implementation.

You should also write tests for the implementation that will be run with pytest. 

* Make sure to write tests for everything explicitly stated in the requirements.
* The tests will be run with pytest.
* Design your test methods such that each test method verifies only a single test scenario.
"""

FIX_TESTS_PROMPT = """You are reviewing a solution written by an inexperienced programmer based on the provided requirements.
The tests failed because the implementation is incorrect.
Review the requirements and the test output to see if there is anything missing. 
Do only write out the functions you change and comment out the other parts of the code. 
"""

# List the changes that is needed and fix the code to make the tests pass.

FILE_FORMAT = """All files should be presented in the following format:

/file.py
```python
# ... code  
```
"""

@ability(
    name="fix_code",
    disabled=False,
    description="Use this to fix failing tests. Provide a file that failed.",
    parameters=[
        {
            "name": "file",
            "description": "File path to the file that should be updated or created.",
            "type": "string",
            "required": True,
        }
    ],
    output_type="string",
)
async def fix_code(
        agent,
        task_id: str,
        step_id: str,
        file: str) -> str:
    return await _write_code(agent, task_id, step_id, file, fix_code_mode=True)


async def _write_code(
        agent,
        task_id: str,
        step_id: str,
        file: str,
        fix_code_mode: bool = False
) -> str:
    logger.info(f"Writing code in {file} for task {task_id}.")
    task = await agent.db.get_task(task_id)
    step = await agent.db.get_step(task_id, step_id)
    repo_dir = agent.workspace.base_path / task_id
    llm = create_openai_client(log_dir=repo_dir / ".prompt_log",
                               llm_name=_llm,
                               temperature=_temperature,
                               max_tokens=2000,
                               streaming=False)

    repository = FileRepository(repo_path=repo_dir, use_git=False)

    if fix_code_mode:
        system_prompt = FIX_TESTS_PROMPT + FILE_FORMAT
    else:
        system_prompt = DEFAULT_PROMPT + FILE_FORMAT

    code_writer = CodeWriter(llm=llm,
                             role_prompt=ROLE_PROMPT,
                             repository=repository,
                             sys_prompt=FILE_FORMAT,
                             allow_hallucinated_files=True,
                             expect_updated_code=True,
                             max_tokens_in_prompt=6100,
                             auto_mode=True)

    other_files = code_writer.repository.get_source_files(language=None, include_test_files=True)
    has_tests = any("test" in f.file_path for f in other_files)

    file_item = FileItem(file_path=file, content=repository.get_file_content(file))
    file_items = [file_item]

    test_file = "test_" + file
    use_existing_tests = False

    if has_tests:
        if not repository.get_file_content(test_file):
            use_existing_tests = True
    else:
        test_file_item = FileItem(file_path=test_file, content=repository.get_file_content(test_file))
        file_items.append(test_file_item)

    other_file_items = read_other_files(repo_dir, file_item)

    if fix_code_mode:
        fix_code_instructions = step.input

        if use_existing_tests:
            fix_code_instructions += "\n\nThe tests are correct, you must adjust the code to make the tests pass."
        else:
            fix_code_instructions += "\n\nBoth the tests and the implementation might be incorrect."

        messages = [
            #Message(sender="Human", items=[TextItem(text="# Requirements\n\n" + task.input)]),
            Message(sender="Human", items=other_file_items),
            Message(sender="Human", items=[TextItem(text="Here's the implementation done by the inexperienced programmer.")] + file_items),
            Message(sender="Human", items=[TextItem(text=fix_code_instructions)]),
            Message(sender="Human", items=[TextItem(text=FIX_TESTS_PROMPT)])
        ]

        if _only_return_changes:
            start_of_file = file_item.content[:300]
            start_update_file = FileItem(file_path=file, content=start_of_file)
            messages.append(Message(sender="AI", items=[start_update_file]))
            messages.append(Message(sender="Human", items=[TextItem(text="Now you're returning the whole file. You should just return the updated code, remember?")]))
            messages.append(Message(sender="AI", items=[TextItem(text="I apologize for the confusion. Here's the updated code:")]))
    else:
        messages = [Message(sender="Human", items=[TextItem(text="# Requirements\n\n" + task.input)] + file_items)]

    try:
        outgoing_messages = code_writer.execute(incoming_messages=messages)
    except Exception as e:
        logger.warning(f"Failed to run code writer. Error: {e}")
        raise e

    output = ""
    text_items = outgoing_messages[0].find_items_by_type("text")
    if text_items:
        output = "\n".join(item.to_prompt() for item in text_items)

    updated_files = []
    update_items = outgoing_messages[0].find_items_by_type("updated_file")
    for updated_file in update_items:
        if updated_file.invalid:
            logger.warning(f"Skipping invalid file {updated_file.file_path} with reason: {updated_file.invalid}")
            output += f"\n\nI couldn't update {updated_file.file_path} because it is invalid: {updated_file.invalid}."
        else:
            logger.info(f"Updated file {updated_file.file_path}")
            if "test" not in updated_file.file_path:  # FIXME: Just to avoid failures in benchmarks from tests not in the actual benchmark tests
                artifact = await agent.db.create_artifact(
                    task_id=task_id,
                    step_id=step_id,
                    file_name=updated_file.file_path[1:],  # Remove leading slash
                    relative_path="",
                    agent_created=True,
                )
            updated_files.append(updated_file.file_path)

    if not output:
        if fix_code_mode:
            output = f"I fixed the code in {updated_files}."
        else:
            output = f"I implemented {updated_files}"
    elif fix_code and not updated_files:
        output += "\n\nI didn't update any files."

    return output
