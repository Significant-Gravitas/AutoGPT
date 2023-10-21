import asyncio
import logging
import os
import re
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from typing import Tuple, Optional, List

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

_job_specs = [
    {
        "id": "1-" + basic_llm_name + "-0.0",
        "model": basic_llm_name,
        "temperature": 0.0,
    },
    {
        "id": "2-" + basic_llm_name + "-0.1",
        "model": basic_llm_name,
        "temperature": 0.1,
    },
    {
        "id": "3-" + basic_llm_name + "-0.4",
        "model": basic_llm_name,
        "temperature": 0.4,
    },
    {
        "id": "4-" + basic_llm_name + "-0.0",
        "model": basic_llm_name,
        "temperature": 0.0,
    },
    {
        "id": "5-" + basic_llm_name + "-0.1",
        "model": default_llm_name,
        "temperature": 0.4,
    },
{
        "id": "6-" + basic_llm_name + "-0.0",
        "model": basic_llm_name,
        "temperature": 0.0,
    },
    {
        "id": "7-" + basic_llm_name + "-0.8",
        "model": basic_llm_name,
        "temperature": 0.1,
    },
    {
        "id": "8-" + basic_llm_name + "-1.0",
        "model": basic_llm_name,
        "temperature": 0.4,
    },
    {
        "id": "9-" + basic_llm_name + "-1.0",
        "model": basic_llm_name,
        "temperature": 0.0,
    }
]

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('openai').setLevel(logging.INFO)
logging.getLogger('urllib3').setLevel(logging.INFO)
logging.getLogger('multipart').setLevel(logging.INFO)

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
async def write_code_external(
        agent,
        task_id: str,
        file: str) -> Tuple[bool, str]:
    return await _write_code(agent, task_id, file, job_specs=_job_specs)


def process_runner(job_spec, task_input, repo_dir_base, file_name):
    logger.info(f"Starting process runner for spec {job_spec} in {repo_dir_base}")

    result = _write_code_job(input=task_input, repo_dir=repo_dir_base, file=file_name, job_spec=job_spec)
    return result


async def _write_code(agent, task_id, file_name, job_specs: list = None, retry=0) -> Tuple[bool, str]:
    logger.info(f"Run parallel coder for task {task_id} with {len(job_specs)} jobs.")
    task = await agent.db.get_task(task_id)
    repo_dir = agent.workspace.base_path / task_id
    if not repo_dir.exists():
        logger.debug(f"Creating directory {repo_dir}")
        repo_dir.mkdir()

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_runner, job_specs, [task.input]*len(job_specs), [agent.workspace.base_path / task_id]*len(job_specs), [file_name]*len(job_specs)))

    successful_results = [result for result in results
                          if result[1] and result[1].success and result[1].verification_count > 0]

    if retry < 3 and not successful_results:
        sorted_results = sorted(results, key=sort_failed_verification_results)

        retry_specs = []
        contents = set()

        for result in sorted_results:
            if len(retry_specs) > 3:
                continue

            job_spec, verification_result = result

            # TODO: Make a better check for which implementations to select. Like:
            updated_file = repo_dir / job_spec["id"] / file_name
            if updated_file in contents:
                logger.info(
                    f"The file {file_name} in {job_spec['id']} was already implemented by another job. Skipping this")
                continue
            contents.add(updated_file)

            top_job, verification_result = result
            logger.info(
                f"Retrying with {job_spec['id']} with {verification_result.verification_count} verifications and "
                f"{verification_result.failed_tests_count} failed tests.")

            fix_instructions = ("\n\n".join([item.to_prompt() for item in verification_result.failures])
                                + f"\n\nThe file {file_name} was implemented, but {verification_result.failed_tests_count} "
                                  f"out of {verification_result.verification_count} tests failed!")
            top_job["fix_instructions"] = fix_instructions
            retry_specs.append(top_job)

        return await _write_code(agent=agent, task_id=task_id, file_name=file_name, job_specs=retry_specs, retry=retry+1)

    logger.info(f"Finished all jobs for task {task_id} with {len(successful_results)} jobs with successful results.")

    sorted_results = sorted(successful_results, key=sort_successful_verification_results)
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
        output += f"\n\nThe file {file_name} was implemented, but {verification_result.failed_tests_count} out of {verification_result.verification_count} tests failed!"
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
        not item.success,  # Successful verifications first
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
    logger.info(f"Generate code for {file} in {repo_dir}. Using job_spec {job_spec}.")
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
                             auto_mode=True)

    other_files = repository.get_source_files(language="python", include_test_files=True)
    has_tests = any("test" in f.file_path for f in other_files)

    file_item = FileItem(file_path=file, content=repository.get_file_content(file))
    file_items = [file_item]

    test_file = "test_" + file

    if has_tests:
        test_tool = PythonPytestTestTool(current_dir=repository.repo_path, test_file_pattern="*.py")
    else:
        test_file_item = FileItem(file_path=test_file, content=repository.get_file_content(test_file))
        file_items.append(test_file_item)
        test_tool = PythonPytestTestTool(current_dir=repository.repo_path, test_file_pattern=test_file)

    for other_file in other_files:
        if not other_file.content:
            logger.info(f"Skipping file {other_file.file_path} because it is empty")
            continue
        if any(file_item.file_path == other_file.file_path for file_item in file_items):
            continue

        is_test = "test" in other_file.file_path
        low_prio_file = (fix_code_instructions and not is_test) or (not fix_code_instructions and is_test)

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
        outgoing_messages = code_writer.execute(incoming_messages=messages)
    except Exception as e:
        # TODO: Ugly hack that only expects max token errors...
        if retry < 3:
            return _write_code_job(input=input, repo_dir=repo_dir, file=file, job_spec=job_spec, retry=retry + 1)
        else:
            logger.info(f"Job {job_spec['id']} failed to finish.")
            return job_spec, None

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
