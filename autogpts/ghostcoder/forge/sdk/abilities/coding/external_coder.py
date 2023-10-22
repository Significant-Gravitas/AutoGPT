from typing import Tuple, Optional

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

DEFAULT_PROMPT = """You're tasked to write an implementation based on the provided task. 
You should also write tests for the implementation that will be run with pytest. 
Make sure to write tests for all requirements.
Just verify one test scenario in each test method.
"""

FIX_TESTS_PROMPT = """You are reviewing a solution written by an inexperienced programmer. 
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
    disabled=False,
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
        fix_code_mode: bool = False,
        retry: int = 0
) -> str:
    logger.info(f"Writing code in {file} for task {task_id}, retry {retry}")
    task = await agent.db.get_task(task_id)
    step = await agent.db.get_step(task_id, step_id)
    repo_dir = agent.workspace.base_path / task_id
    llm = create_openai_client(log_dir=repo_dir / ".prompt_log",
                               llm_name=smart_llm_name,
                               temperature=0.0,
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
                             auto_mode=True)

    other_files = code_writer.repository.get_source_files(language="python", include_test_files=True)
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

    for other_file in other_files:
        if not other_file.content:
            logger.info(f"Skipping file {other_file.file_path} because it is empty")
            continue
        if any(file_item.file_path == other_file.file_path for file_item in file_items):
            continue

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

        if not skip_file:
            if trim_file:
                logger.info(f"Trimming file {other_file.file_path}")
                content = trim(other_file.content)
            else:
                content = other_file.content

            other_file.content = content

            if test_file not in other_file.file_path:
                other_file.readonly = True
            file_items.append(other_file)
        else:
            logger.info(f"Skipping file {other_file.file_path}")

    if fix_code_mode:
        fix_code_instructions = step.input

        if use_existing_tests:
            fix_code_instructions += "\n\nThe tests are correct, you must adjust the code to make the tests pass."
        else:
            fix_code_instructions += "\n\nBoth the tests and the implementation might be incorrect."

        messages = [
            Message(sender="Human", items=[TextItem(text=task.input)]),
            Message(sender="AI", items=file_items),
            Message(sender="Human", items=[TextItem(text=fix_code_instructions)])
        ]
    else:
        messages = [Message(sender="Human", items=[TextItem(text=task.input)] + file_items)]
    try:
        outgoing_messages = code_writer.execute(incoming_messages=messages)
    except Exception as e:
        # TODO: Ugly hack that only expects max token errors and will retry with a smaller prompt context... Should be calculated with tiktoken before execution.
        if retry < 3:
            return await _write_code(agent, task_id, file, fix_code_mode=fix_code_mode, retry=retry+1)
        else:
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
            if "test" not in updated_file.file_path:  # FIXME: Just to avoid test failures from tests not in the actual benchmark tests
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

