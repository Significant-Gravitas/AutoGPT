import asyncio
import logging
import os
import re
import shutil
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from typing import Tuple, Optional, List

import tiktoken

from forge.sdk import ForgeLogger
from forge.sdk.abilities.registry import ability
from ghostcoder.codeblocks import create_parser, CodeBlockType
from ghostcoder.filerepository import FileRepository
from ghostcoder.actions import CodeWriter
from ghostcoder.actions.write_code.base import OutputFormat
from ghostcoder.benchmark.utils import create_openai_client
from ghostcoder.schema import Message, TextItem, FileItem, CodeItem, VerificationResult
from ghostcoder.test_tools.verify_python_pytest import PythonPytestTestTool

logger = ForgeLogger(__name__)

smart_llm_name = "gpt-4"
basic_llm_name = "gpt-3.5-turbo-16k"

default_llm_name = smart_llm_name


logging.basicConfig(level=logging.INFO)
logging.getLogger('openai').setLevel(logging.INFO)
logging.getLogger('urllib3').setLevel(logging.INFO)
logging.getLogger('multipart').setLevel(logging.INFO)

_basic_specs = [
    {
        "id": "01-" + basic_llm_name + "-0.00",
        "model": basic_llm_name,
        "temperature": 0.00,
    },
    {
        "id": "02-" + basic_llm_name + "-0.01",
        "model": basic_llm_name,
        "temperature": 0.01,
    },
    {
        "id": "03-" + basic_llm_name + "-0.02",
        "model": basic_llm_name,
        "temperature": 0.02,
    },
    {
        "id": "04-" + basic_llm_name + "-0.03",
        "model": basic_llm_name,
        "temperature": 0.03,
    },
    {
        "id": "05-" + basic_llm_name + "-0.04",
        "model": basic_llm_name,
        "temperature": 0.04,
    },
{
        "id": "06-" + basic_llm_name + "-0.05",
        "model": basic_llm_name,
        "temperature": 0.05,
    },
    {
        "id": "07-" + basic_llm_name + "-0.10",
        "model": basic_llm_name,
        "temperature": 0.10,
    },
    {
        "id": "08-" + basic_llm_name + "-0.20",
        "model": basic_llm_name,
        "temperature": 0.20,
    },
    {
        "id": "09-" + basic_llm_name + "-0.30",
        "model": basic_llm_name,
        "temperature": 0.30
    },
    {
        "id": "10-" + basic_llm_name + "-0.40",
        "model": basic_llm_name,
        "temperature": 0.40,
    }
]

_expensive_specs = [
    {
        "id": "01-" + smart_llm_name + "-0.0",
        "model": smart_llm_name,
        "temperature": 0.00,
    },
    {
        "id": "02-" + smart_llm_name + "-0.0",
        "model": smart_llm_name,
        "temperature": 0.01,
    },
    {
        "id": "03-" + smart_llm_name + "-0.1",
        "model": smart_llm_name,
        "temperature": 0.02,
    },
    {
        "id": "04-" + smart_llm_name + "-0.2",
        "model": smart_llm_name,
        "temperature": 0.03,
    },
    {
        "id": "05-" + smart_llm_name + "-0.4",
        "model": smart_llm_name,
        "temperature": 0.04,
    }
]

_job_specs = _basic_specs

timeout = 90

upgrade_to_smart_model = False
use_pytest_parser = True

DEFAULT_PROMPT = """You're tasked to write an implementation based on the provided task. 
You should also write tests for the implementation. Make sure to write tests for all requirements.
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
async def write_code_external(
        agent,
        task_id: str,
        step_id: str,
        file: str) -> Tuple[bool, str]:
    return await _write_code(agent, task_id, file, job_specs=_job_specs)


def process_runner(job_spec, task_input, repo_dir_base, file_name, is_retry) -> Tuple[dict, Optional[VerificationResult]]:
    logger.info(f"Starting process runner for spec {job_spec['id']} in {repo_dir_base}")

    if is_retry and upgrade_to_smart_model:
        job_spec["model"] = smart_llm_name

    try:
        return _write_code_job(input=task_input, repo_dir=repo_dir_base, file=file_name, job_spec=job_spec)
    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(f"Job {job_spec['id']} failed with exception:\n{stack_trace}")
        return job_spec, VerificationResult(success=False, error=True, verification_count=0, failed_tests_count=0, message=str(e))


async def _write_code(agent, task_id, file_name, job_specs: list[dict] = None, retry=0) -> Tuple[bool, str]:
    logger.info(f"Run parallel coder for task {task_id} with {len(job_specs)} jobs.")
    task = await agent.db.get_task(task_id)
    repo_dir: Path = agent.workspace.base_path / task_id
    if not repo_dir.exists():
        logger.debug(f"Creating directory {repo_dir}")
        repo_dir.mkdir()

    common_args = [task.input, agent.workspace.base_path / task_id, file_name, retry > 0]

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_runner, job_spec, *common_args) for job_spec in job_specs]

        results = []
        for i, future in enumerate(futures):
            try:
                async_future = asyncio.wrap_future(future)
                result = await asyncio.wait_for(async_future, timeout)
                if result[1]:
                    results.append(result)
                else:
                    logger.warning(f"Job {job_specs[i]['id']} has no verification results. Skipping this.")
            except TimeoutError:
                logger.warning(f"Job {job_specs[i]['id']} exceeded the timeout limit of {timeout} seconds. Skipping this.")
            except Exception as e:
                logger.error(f"Job {job_specs[i]['id']} failed with an unexpected error: {e}. Skipping this.", exc_info=True)

    successful_results = [result for result in results
                          if result[1] and result[1].success and result[1].verification_count > 0]

    if successful_results:
        sorted_results = sorted(successful_results, key=sort_successful_verification_results)
    else:
        sorted_results = sorted(results, key=sort_failed_verification_results)

    if retry < 6 and not successful_results:
        retry_specs = []
        contents = set()

        for result in sorted_results:
            if len(retry_specs) > 5:
                continue

            job_spec, verification_result = result

            # TODO: Make a better check for which implementations to select.
            updated_file = repo_dir / job_spec["id"] / file_name
            updated_contents = updated_file.read_text().strip()

            if updated_contents in contents:
                logger.info(
                    f"The file {file_name} in {job_spec['id']} was already implemented by another job. Skipping this")
                continue
            else:
                logging.info(f"New contents {updated_contents}")

            if verification_result.verification_count > 1 and "last_result" in job_spec and job_spec["last_result"] <= verification_result.failed_tests_count:
                logger.info(f"The job {job_spec['id']} has more failures ({verification_result.failed_tests_count}) "
                            f"than previous run {job_spec['last_result']}. Skipping this.")
                continue

            job_spec["last_result"] = verification_result.failed_tests_count

            contents.add(updated_contents)

            logger.info(
                f"Retrying with {job_spec['id']} with {verification_result.verification_count} verifications and "
                f"{verification_result.failed_tests_count} failed tests.")

            fix_instructions = ("\n\n".join([item.to_prompt() for item in verification_result.failures])
                                + f"\n\nThe file {file_name} was implemented, but {verification_result.failed_tests_count} "
                                  f"out of {verification_result.verification_count} tests failed!")
            job_spec["fix_instructions"] = fix_instructions
            retry_specs.append(job_spec)

        if retry_specs:
            return await _write_code(agent=agent, task_id=task_id, file_name=file_name, job_specs=retry_specs, retry=retry+1)
        else:
            logger.info(f"No jobs to retry.")

    job_ids = [result[0]["id"] for result in successful_results]
    logger.info(f"Finished all jobs for task {task_id} with {len(successful_results)} jobs ({job_ids}) with successful results. ")

    best_spec, verification_result = sorted_results[0]
    logger.info(
        f"Best result {best_spec['id']} using model {best_spec['model']} with {verification_result.verification_count} "
        f"verifications and {verification_result.failed_tests_count} failed tests.")

    job_dir = repo_dir / best_spec["id"]
    for file in job_dir.iterdir():
        if file.is_file():
            shutil.copy(file, repo_dir)

    await agent.db.create_artifact(
        task_id=task_id,
        file_name=file_name,
        relative_path="",
        agent_created=True,
    )

    output = ""
    if not verification_result.success:
        output += "\n\n".join([item.to_prompt() for item in verification_result.failures])
        output += (f"\n\nThe file {file_name} was implemented, but {verification_result.failed_tests_count} out of"
                   f" {verification_result.verification_count} tests failed!")
    elif verification_result.verification_count > 0:
        output = f"\n\nThe file {file_name} was implemented and {verification_result.verification_count} tests passed!"

    return True, output
    #return verification_result.success, output


def sort_failed_verification_results(job: Tuple[dict, VerificationResult]) -> tuple:
    _, item = job
    method_and_class_set = all([failure.test_method and failure.test_class for failure in item.failures])
    verification_priority = 0 if item.verification_count > 0 else float('inf')
    error_priority = 1 if item.error else 0

    return (
        error_priority,  # Items with error come last
        verification_priority,  # Items with count > 0 come before those with count 0
        -(item.verification_count >= 2),  # Items with count at least 2 come before others
        item.failed_tests_count if item.verification_count >= 2 else float('inf'),  # Fewest failures first, but only for counts >= 2
        -item.verification_count,  # Higher verification counts come before lower counts
        not method_and_class_set  # Failures without method and class come last
    )

def sort_successful_verification_results(job: Tuple[dict, VerificationResult]) -> tuple:
    spec, item = job

    is_smart_model = smart_llm_name in spec.get("model", "")
    gpt_4_priority = 1 if is_smart_model else 0

    return (
        gpt_4_priority,  # Priority level depending on if it's the smart model was used
        -item.verification_count  # Higher verification counts still come before lower counts
    )

def _write_code_job(
        input: str,
        repo_dir: Path,
        file: str,
        job_spec: dict,
        retry: int = 0
) -> Tuple[dict, Optional[VerificationResult]]:
    logger.info(f"Generate code for {file} in {repo_dir}. Using job_spec {job_spec['id']}. Retry {retry}.")
    starttime = time.time()

    job_dir = repo_dir / job_spec["id"]

    if not job_dir.exists():
        logger.debug(f"Creating directory {job_dir}")
        job_dir.mkdir()

    if not any(job_dir.iterdir()):
        for f in repo_dir.iterdir():
            if f.is_file():
                shutil.copy(f, job_dir)

    llm = create_openai_client(log_dir=job_dir / ".prompt_log", llm_name=job_spec["model"], temperature=job_spec["temperature"], streaming=False)

    repository = FileRepository(repo_path=job_dir, use_git=False)

    fix_code_instructions = None
    if "fix_instructions" in job_spec:
        system_prompt = FIX_TESTS_PROMPT + FILE_FORMAT
        fix_code_instructions = job_spec["fix_instructions"]
    else:
        system_prompt = DEFAULT_PROMPT + FILE_FORMAT

    code_writer = CodeWriter(llm=llm,
                             role_prompt="You're an AI Developer with superior programming skills.",
                             repository=repository,
                             sys_prompt=system_prompt,
                             allow_hallucinated_files=True,
                             auto_mode=False)

    other_files = repository.get_source_files(language="python", include_test_files=True)
    has_tests = any("test" in f.file_path for f in other_files)

    file_item = FileItem(file_path=file, content=repository.get_file_content(file))
    file_items = [file_item]

    test_file = "test_" + file

    if has_tests:
        test_tool = PythonPytestTestTool(current_dir=repository.repo_path, test_file_pattern="*.py", parse_test_results=use_pytest_parser)
    else:
        test_file_item = FileItem(file_path=test_file, content=repository.get_file_content(test_file))
        file_items.append(test_file_item)
        test_tool = PythonPytestTestTool(current_dir=repository.repo_path, test_file_pattern=test_file, parse_test_results=use_pytest_parser)

    for other_file in other_files:
        if not other_file.content:
            logger.info(f"Skipping file {other_file.file_path} because it is empty")
            continue
        if any(file_item.file_path == other_file.file_path for file_item in file_items):
            continue

        is_test = "test" in other_file.file_path
        low_prio_file = fix_code_instructions and not is_test

        trim_file = False
        skip_file = False

        # Skip test files pytest parser isn't used as the code of all failing tests will be provided anyway
        if low_prio_file:  # TODO: Check context length first
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

    logger.info(f"Starting job {repo_dir.name} with {len(file_items)} files from {len(other_files)} other files.")

    if fix_code_instructions:
        messages = [
            Message(sender="Human", items=[TextItem(text=input)]),
            Message(sender="AI", items=file_items),
            Message(sender="Human", items=[TextItem(text=fix_code_instructions)])
        ]
    else:
        messages = [Message(sender="Human", items=[TextItem(text=input)] + file_items)]
    try:
        msg_str = "\n\n".join([msg.to_prompt() for msg in messages])
        enc = tiktoken.encoding_for_model(job_spec['model'])
        tokens = enc.encode(msg_str)
        logger.info(f"Prompt length: {len(tokens)}")

        outgoing_messages = code_writer.execute(incoming_messages=messages)
    except Exception as e:
        # TODO: Ugly hack that only expects max token errors...
        if retry < 3:
            logger.info(f"Retry job {job_spec['id']} as it failed with exception: {e}")
            return _write_code_job(input=input, repo_dir=repo_dir, file=file, job_spec=job_spec, retry=retry + 1)
        else:
            logger.info(f"Job {job_spec['id']} failed to finish.")
            return job_spec, None

    if not outgoing_messages[0].find_items_by_type("updated_file"):
        logger.warning(f"Job {job_spec['id']} didn't update any files.")
        return job_spec, None

    logger.info(f"Verifying job {job_spec['id']}...")

    results = test_tool.run_tests()

    if results.success:
        logger.info(f"Job {job_spec['id']} successfully finished in {time.time() - starttime} seconds with {results.verification_count} verifications")
    else:
        logger.info(f"Job {job_spec['id']} finished in {time.time() - starttime} seconds with with {results.failed_tests_count} failed out of {results.verification_count} verifications.")

    return job_spec, results


def trim(content: str):
    parser = create_parser(language="python")
    code_block = parser.parse(content)
    trimmed_block = code_block.trim2(include_types=[CodeBlockType.FUNCTION, CodeBlockType.CLASS])
    return trimmed_block.to_string()
