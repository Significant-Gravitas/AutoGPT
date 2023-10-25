import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Tuple, Optional, List

from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI

from forge.sdk import ForgeLogger, Artifact
from forge.sdk.abilities.registry import ability
from ghostcoder.callback import LogCallbackHandler
from ghostcoder.filerepository import FileRepository
from ghostcoder.actions import CodeWriter
from ghostcoder.llm import ChatLLMWrapper
from ghostcoder.schema import Message, TextItem, FileItem, CodeItem

logger = ForgeLogger(__name__)

use_pytest_parser = True

smart_llm_name = "gpt-4"
basic_llm_name = "gpt-3.5-turbo"

_llm = smart_llm_name
_temperature = 0.0

_enable_test = False

_instruct_by_apology = False

ROLE_PROMPT = """You're a Staff Engineer with superior python programming skills that can tackle complex tasks in just one try."""

UNDERSTAND = "List your interpretation of the requirements in a compact style and do a full implementation of the requirements."

ONLY_WRITE_CODE = "Do a full implementation of the requirements."

WRITE_CODE_PROMPT = """The implementation will be verified by a test suite and must therefore be fully functioning.
Do not add placeholders. All requirements must be implemented right away!
Split up complex logic in many functions.
Stop after you wrote the code. 

And most important. You must implement the full solution, not a simplified implementation, even if it might look complex!"""

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

FILE_FORMAT = """All files should be presented in the following format and end with ---

/file.py
```python
# ... code  
```
---
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

    if not has_tests(repo_dir) and _enable_test:
        logging.info(f"Run parallel coding for test and code files")
        with ProcessPoolExecutor() as executor:
            code_future = executor.submit(_write_code, WRITE_CODE_PROMPT, task.input, file, repo_dir, 2000)

            test_file = "test_" + file
            test_future = executor.submit(_write_code, WRITE_TEST_PROMPT, task.input, test_file, repo_dir, 900)

            async_code_future = asyncio.wrap_future(code_future)
            async_test_future = asyncio.wrap_future(test_future)

            outgoing_messages = await asyncio.wait_for(async_code_future, timeout=75)
            await asyncio.wait_for(async_test_future, timeout=75)

            logger.info("Both writers are done")
    else:
        outgoing_messages = _write_code(WRITE_CODE_PROMPT, task.input, file, repo_dir, max_tokens=2000)

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
                               stop_sequence="---",
                               streaming=True)

    repository = FileRepository(repo_path=repo_dir, use_git=False)

    code_writer = CodeWriter(llm=llm,
                             repository=repository,
                             role_prompt=ROLE_PROMPT,
                             sys_prompt=FILE_FORMAT,  # WRITE_CODE_PROMPT is moved to last message
                             allow_hallucinated_files=False,
                             expect_updated_code=True,
                             max_tokens_in_prompt=6100,
                             auto_mode=True)

    file_item = FileItem(file_path=file, content=repository.get_file_content(file), stop_sequence="---")
    items = [TextItem(text="# Requirements:\n" + instructions)]

    other_file_items = read_other_files(repo_dir, file_item)
    items.extend(other_file_items)
    items.append(file_item)

    prompt = UNDERSTAND
    if has_tests(repo_dir):
        prompt = ONLY_WRITE_CODE

    messages = [Message(sender="Human", items=items), Message(sender="Human", items=[TextItem(text=prompt)])]

    try:
        logger.info(f"Call code writer with {file}.")
        messages = code_writer.execute(incoming_messages=messages)
        logger.info(f"Code writer is finished writing code to {file} and returned {len(messages)} messages.")
        return messages
    except Exception as e:
        raise e


def has_tests(dir: Path):
    if dir.exists():
        for f in dir.iterdir():
            if f.is_file() and "test" in f.name:
                logging.info(f"Found existing test file {f.name}")
                return True
    return False


def create_openai_client(log_dir: Path, llm_name: str, temperature: float, streaming: bool = True, max_tokens: Optional[int] = None, stop_sequence: str = None):
    callback = LogCallbackHandler(str(log_dir))
    logger.info(f"create_openai_client(): llm_name={llm_name}, temperature={temperature}, log_dir={log_dir}")

    model_kwargs = {}
    if stop_sequence:
        model_kwargs["stop"] = [stop_sequence]

    return ChatLLMWrapper(ChatOpenAI(
        model=llm_name,
        model_kwargs=model_kwargs,
        max_tokens=max_tokens,
        temperature=temperature,
        streaming=streaming,
        callbacks=[callback, StreamingStdOutCallbackHandler()]
    ))


def read_other_files(repo_dir: Path, file_item: FileItem) -> List[FileItem]:
    other_file_items = []
    for other_file in repo_dir.iterdir():
        if not other_file.is_file():
            continue
        if other_file.name in file_item.file_path:
            continue
        content = other_file.read_text()
        if not content:
            continue

        priority = 0
        if other_file.name.endswith(".py"):
            priority = 1
        if "test" in other_file.name:
            priority = 2

        other_file_items.append(FileItem(file_path=other_file.name, content=content, readonly=True, priority=priority, stop_sequence="---"))

    return other_file_items
