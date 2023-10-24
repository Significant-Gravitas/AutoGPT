import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Tuple, Optional, List

from forge.sdk import ForgeLogger, Artifact
from forge.sdk.abilities.registry import ability
from ghostcoder.codeblocks import create_parser, CodeBlockType
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

_enable_test = False

_instruct_by_apology = False

SHOW_UNDERSTANDING = "List your interpretation of the requirements in a compact style and do a full implementation of the requirements."

SHOW_UNDERSTANDING_IN_COMMENTS = "Do a full implementation of the requirements. Show that you've understood the requirements by writing comments in the code to show of the code is fulfilling the requirements."

WRITE_CODE_PROMPT = """List your interpretation of the requirements in a compact style and do a full implementation.
The implementation will be verified by a test suite and must therefore be fully functioning.
Do not add placeholders or not fully implemented functions. All requirements must be implemented right away!
Do not provide any more information after you wrote the code. 
"""


THINK_PROMPT = """Start by thinking through how to do the implementation and explain your reasoning. Then do a full implementation."""

WRITE_TEST_PROMPT = """You are tasked with writing tests based on the provided requirements. 
Review the requirements and list your interpretations and the test that is needed. 

* Note that the code implementation will be written by another developer; your sole responsibility is to write the tests.
* The tests will be run with pytest.
* Design your test methods such that each one verifies only a single test scenario.
* Write max three tests. 
* External resources like a database or a file system must be mocked in the test.
* If the requirements is to build a CLI tool, use subprocess to run the CLI tool and verify the output.
* Verify only scenarios specified in the requirements.
* DD NOT verify if the implementation raises exceptions if not explicitly stated in the requirements.
* DO NOT verify invalid input if not explicitly stated in the requirements.
* If there is a test provided in the requirements, you should use it as a starting point for your tests.
"""

CHAIN_OF_THOUGHT = "\n\nLet's work this out in a step by step way to be sure we have the right answer."

FILE_FORMAT = """All files should be presented in the following format:

/file.py
```python
# ... code  
```
"""

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
    task = await agent.db.get_task(task_id)
    repo_dir = agent.workspace.base_path / task_id

    has_tests = False

    if repo_dir.exists():
        for f in repo_dir.iterdir():
            if f.is_file() and "test" in f.name:
                logging.info(f"Found existing test file {f.name}")
                has_tests = True
                break

    if not has_tests and _enable_test:
        logging.info(f"Run parallel coding for test and code files")
        with ProcessPoolExecutor() as executor:
            code_future = executor.submit(_write_code, WRITE_CODE_PROMPT, task.input, file, repo_dir, None)

            test_file = "test_" + file
            test_future = executor.submit(_write_code, WRITE_TEST_PROMPT, task.input, test_file, repo_dir, 900)

            async_code_future = asyncio.wrap_future(code_future)
            async_test_future = asyncio.wrap_future(test_future)

            outgoing_messages = await asyncio.wait_for(async_code_future, timeout=75)
            await asyncio.wait_for(async_test_future, timeout=75)

            logger.info("Both writers are done")
    else:
        outgoing_messages = _write_code(WRITE_CODE_PROMPT, "", file, repo_dir, max_tokens=3000)

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
            artifact = await agent.db.create_artifact(
                task_id=task_id,
                step_id=step_id,
                file_name=updated_file.file_path[1:],  # Remove leading slash
                relative_path="",
                agent_created=True,
            )
            updated_files.append(updated_file.file_path)

    if not output:
        output = f"I implemented {updated_files}"

    return output


def _write_code(
        system_prompt: str,
        instructions: str,
        file: str,
        repo_dir: Path,
        max_tokens: Optional[int] = None) -> List[Message]:

    logging.basicConfig(level=logging.INFO)
    logging.getLogger('openai').setLevel(logging.INFO)
    logging.getLogger('urllib3').setLevel(logging.INFO)
    logging.getLogger('multipart').setLevel(logging.INFO)

    logger.info(f"Writing code in {file}")
    llm = create_openai_client(log_dir=repo_dir / ".prompt_log",
                               llm_name=_llm,
                               max_tokens=max_tokens,
                               temperature=_temperature,
                               streaming=False)

    repository = FileRepository(repo_path=repo_dir, use_git=False)

    code_writer = CodeWriter(llm=llm,
                             role_prompt="You're an AI Developer with superior programming skills.",
                             repository=repository,
                             sys_prompt=system_prompt + FILE_FORMAT,
                             allow_hallucinated_files=False,
                             expect_updated_code=True,
                             auto_mode=True)

    other_files = code_writer.repository.get_source_files(include_test_files=True)

    file_item = FileItem(file_path=file, content=repository.get_file_content(file))
    items = [TextItem(text="# Requirements:\n" + instructions)]

    for other_file in other_files:
        if not other_file.content:
            logger.info(f"Skipping file {other_file.file_path} because it is empty")
            continue
        if other_file.file_path == file_item.file_path:
            continue

        #if "test" in other_file.file_path: # or "txt" in other_file.file_path:
        #    continue

        other_file.readonly = True
        items.append(other_file)

    items.append(file_item)
    messages = [Message(sender="Human", items=items)]

    if _instruct_by_apology:
        messages.append(Message(sender="AI", items=[TextItem(text="Here's a basic implementation: ")]))
        messages.append(Message(sender="Human", items=[TextItem(text="Stop! You should return the full functioning implementation.")]))
        messages.append(Message(sender="AI", items=[TextItem(text="I apologize for the confusion. Here are the full implementation that fulfills the requirements:")]))

    try:
        logger.info(f"Call code writer with {file}.")
        messages = code_writer.execute(incoming_messages=messages)
        logger.info(f"Code writer is finished writing code to {file} and returned {len(messages)} messages.")
        return messages
    except Exception as e:
        raise e


def trim(content: str):
    parser = create_parser(language="python")
    code_block = parser.parse(content)
    trimmed_block = code_block.trim2(include_types=[CodeBlockType.FUNCTION, CodeBlockType.CLASS])
    return trimmed_block.to_string()

