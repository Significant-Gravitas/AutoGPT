from typing import Tuple, Optional, List

import tiktoken

from forge.sdk import ForgeLogger, Artifact
from forge.sdk.abilities.registry import ability
from ghostcoder.codeblocks import create_parser, CodeBlockType
from ghostcoder.filerepository import FileRepository
from ghostcoder.actions import CodeWriter
from ghostcoder.benchmark.utils import create_openai_client
from ghostcoder.schema import Message, TextItem, FileItem, CodeItem
from ghostcoder.test_tools.verify_python_pytest import PythonPytestTestTool

logger = ForgeLogger(__name__)

use_pytest_parser = True

smart_llm_name = "gpt-4"
basic_llm_name = "gpt-3.5-turbo"

_llm = smart_llm_name
_temperature = 0.0

_only_return_changes = True

DEFAULT_PROMPT = """You're tasked to write an implementation based on the provided task. 
Review the requirements and write out your interpretation of the requirements and then do the full implementation.

You should also write tests for the implementation that will be run with pytest. 

* Make sure to write tests for everything explicitly stated in the requirements.
* The tests will be run with pytest.
* Design your test methods such that each test method verifies only a single test scenario.
"""

FIX_TESTS_PROMPT = """You are reviewing a solution written by an inexperienced programmer based on the provided requirements.
Fix the code to make the tests pass. 
Do only write out the functions you change.
"""

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


@ability(
    name="write_code",
    disabled=True,
    description="Use this to write code and tests. Provide the name of the file that should be implemented.",
    parameters=[
        {
            "name": "file",
            "description": "Name of the file that should be updated or created.",
            "type": "string",
            "required": True,
        }
    ],
    output_type="string",
)
async def write_code(
        agent,
        task_id: str,
        step_id: str,
        file: str) -> str:
    return await _write_code(agent, task_id, step_id, file)


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
                               streaming=False)

    repository = FileRepository(repo_path=repo_dir, use_git=False)

    if fix_code_mode:
        system_prompt = FIX_TESTS_PROMPT + FILE_FORMAT
    else:
        system_prompt = DEFAULT_PROMPT + FILE_FORMAT

    code_writer = CodeWriter(llm=llm,
                             role_prompt="You're an AI Developer with superior programming skills.",
                             repository=repository,
                             sys_prompt=system_prompt,
                             allow_hallucinated_files=True,
                             expect_updated_code=True,
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

    other_file_items = [FileItem(file_path=other_file.file_path, content=other_file.content, readonly=True)
                        for other_file in other_files
                        if other_file.content and any(file_item.file_path == other_file.file_path for file_item in file_items)]
    other_file_message = Message(sender="Human", items=other_file_items)

    if fix_code_mode:
        fix_code_instructions = step.input

        if use_existing_tests:
            fix_code_instructions += "\n\nThe tests are correct, you must adjust the code to make the tests pass."
        else:
            fix_code_instructions += "\n\nBoth the tests and the implementation might be incorrect."

        messages = [
            #Message(sender="Human", items=[TextItem(text="# Requirements\n\n" + task.input)]),
            other_file_message,
            Message(sender="Human", items=[TextItem(text="Here's the implementation done by the inexperienced programmer.")] + file_items),
            Message(sender="Human", items=[TextItem(text=fix_code_instructions)])
        ]

        if _only_return_changes:
            start_of_file = file_item.content[:300]
            start_update_file = FileItem(file_path=file, content=start_of_file)
            messages.append(Message(sender="AI", items=[start_update_file]))
            messages.append(Message(sender="Human", items=[TextItem(text="Now you're returning the whole file. You should just return the updated code, remember?")]))
            messages.append(Message(sender="AI", items=[TextItem(text="I apologize for the confusion. I'll start by list my interpretations of the requirements and then I do a full implementation.")]))
    else:
        messages = [Message(sender="Human", items=[TextItem(text="# Requirements\n\n" + task.input)] + file_items)]

    exceeding_tokens = calculate_tokens(messages) - 6000
    if exceeding_tokens > 0:
        other_file_message.items = trim_files(exceeding_tokens, fix_code_mode, other_file_items)

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


def trim(content: str):
    parser = create_parser(language="python")
    code_block = parser.parse(content)
    trimmed_block = code_block.trim2(include_types=[CodeBlockType.FUNCTION, CodeBlockType.CLASS])
    return trimmed_block.to_string()

def calculate_tokens(messages: List[Message]):
    msg_str = "\n\n".join([msg.to_prompt() for msg in messages])
    return calculate_tokens(msg_str)

def calculate_tokens(content: str):
    enc = tiktoken.encoding_for_model(smart_llm_name)
    tokens = enc.encode(content)
    return len(tokens)

def trim_files(exceeding_tokens: int, fix_code_mode: bool, other_file_items: List[FileItem], retry = 0) -> List[FileItem]:
    logger.info(f"Exceeding tokens by {exceeding_tokens}, will try to trim request, retry {retry}")
    trimmed_list = []
    for other_file in other_file_items:
        is_test = "test" in other_file.file_path
        low_prio_file = fix_code_mode and not is_test

        trim_file = False
        skip_file = False

        if retry == 1 and low_prio_file:
            trim_file = True
        elif retry == 2 and low_prio_file:
            skip_file = True
        elif retry == 2 and not low_prio_file:
            trim_file = True
        elif retry == 3:
            skip_file = True

        trimmed_content = ""
        before = other_file.content

        if exceeding_tokens <= 0:
            trimmed_list.append(other_file)
        if not skip_file:
            if trim_file:
                logger.info(f"Trimming file {other_file.file_path}")
                trimmed_content = trim(other_file.content)
            else:
                trimmed_content = other_file.content

            other_file.content = trimmed_content

            trimmed_list.append(other_file)
        else:
            logger.info(f"Skipping file {other_file.file_path}")

        exceeding_tokens -= calculate_tokens(before) - calculate_tokens(trimmed_content)

    if exceeding_tokens < 0 and retry < 3:
        return trim_files(exceeding_tokens, fix_code_mode, trimmed_list, retry + 1)
    else:
        return []
