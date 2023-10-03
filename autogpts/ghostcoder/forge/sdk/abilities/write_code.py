from pathlib import Path
from typing import List

from ghostcoder import FileRepository
from ghostcoder.actions import CodeWriter
from ghostcoder.actions.verify.code_verifier import CodeVerifier
from ghostcoder.actions.write_code.prompt import FIX_TESTS_PROMPT
from ghostcoder.benchmark.utils import create_openai_client
from ghostcoder.codeblocks import create_parser, CodeBlockType
from ghostcoder.schema import Message, TextItem, FileItem, UpdatedFileItem
from ghostcoder.test_tools.verify_python_pytest import PythonPytestTestTool
from ..forge_log import ForgeLogger
from .registry import ability

LOG = ForgeLogger(__name__)

smart_llm_name = "gpt-4"
basic_llm_name = "gpt-3.5-turbo"

@ability(
    name="write_code",
    description="Use this to write code. Provide as detailed instructions as possible in free text and list the files that should be updated or created",
    parameters=[
        {
            "name": "instructions",
            "description": "Detailed instructions for the code to be written.",
            "type": "string",
            "required": True,
        },
        {
            "name": "files",
            "description": "A list of file paths to files that should be updated or created.",
            "type": "list[str]",
            "required": True,
        }
    ],
    output_type="None",
)
async def write_code(
    agent,
    task_id: str,
    instructions: str,
    files: List[str] = [],
) -> str:
    file_items = [FileItem(file_path=file_path) for file_path in files]
    return await run_code_writer(agent, task_id, instructions, file_items)


@ability(
    name="write_tests",
    description="Use this to write and run unit tests to verify the code. Provide as detailed instructions as possible in free text and the file that should be tested and the file where the tests should be written.",
    parameters=[
        {
            "name": "instructions",
            "description": "Detailed instructions for the code to be written.",
            "type": "string",
            "required": True,
        },
        {
            "name": "test_file",
            "description": "File path to the test file.",
            "type": "str",
            "required": True,
        },
        {
            "name": "file_to_test",
            "description": "File path to the file that should be tested.",
            "type": "str",
            "required": True,
        }
    ],
    output_type="None",
)
async def write_tests(
    agent,
    task_id: str,
    instructions: str,
    file_to_test: str,
    test_file: str
) -> str:
    file_items = [FileItem(file_path=file_to_test), FileItem(file_path=test_file)]
    return await run_code_writer(agent, task_id, instructions, file_items, test_file=test_file)


async def run_code_writer(agent, task_id: str, instructions: str, file_items: List[FileItem] = [], test_file: str = None):
    task = await agent.db.get_task(task_id)

    repo_dir = agent.workspace.base_path / task_id

    llm = create_openai_client(log_dir=repo_dir / ".prompt_log", llm_name=smart_llm_name, temperature=0.0, streaming=False)

    repository = FileRepository(repo_path=repo_dir, use_git=False)

    code_writer = CodeWriter(llm=llm,
                             repository=repository,
                             allow_hallucinated_files=True,
                             auto_mode=True)

    for file_item in file_items:
        if not file_item.content and repository:
            LOG.debug(f"Get current file content for {file_item.file_path}")
            content = repository.get_file_content(file_path=file_item.file_path)
            if content:
                file_item.content = content
            elif file_item.new:
                file_item.content = ""

    for file in code_writer.repository.get_source_files(language="python", include_test_files=False):
        parser = create_parser(language="python")
        if not any(file_item.file_path == file.file_path for file_item in file_items):
            code_block = parser.parse(file.content)
            trimmed_block = code_block.trim2(include_types=[CodeBlockType.FUNCTION, CodeBlockType.CLASS])
            file.content = trimmed_block.to_string()
            file.readonly = True
            file_items.append(file)

    items = [TextItem(text=task.input), TextItem(text=instructions)]
    if test_file:
        items.append(TextItem(text=f"Be sure that you write test cases for all requirements and corner cases."))

    message = Message(sender="Human",
                      items=items + file_items)

    outgoing_messages = code_writer.execute(incoming_messages=[message])

    if test_file:
        verification_message = verify(repository, messages=[message] + outgoing_messages, test_file=test_file)
        outgoing_messages.append(verification_message)

    artifacts = []
    for file_item in outgoing_messages[-1].find_items_by_type("file"):
        if file_item.file_path.startswith("/"):
            file_path = file_item.file_path[1:]
        else:
            file_path = file_item.file_path

        artifact = await agent.db.create_artifact(
            task_id=task_id,
            file_name=file_path.split("/")[-1],
            relative_path=file_path,
            agent_created=True,
        )

        LOG.debug(f"Created artifact {artifact.artifact_id} for {file_path}")
        artifacts.append(artifact)

    llm_messages = ""
    for message in outgoing_messages:
        llm_messages += "\n\n" + message.to_prompt()
    return llm_messages


def verify(repository: FileRepository, messages: List[Message], test_file: str, retries: int = 3, last_run: int = 0) -> [Message]:
    test_tool = PythonPytestTestTool(current_dir=repository.repo_path, test_file_pattern=test_file)
    verifier = CodeVerifier(repository=repository, test_tool=test_tool)

    files = dict()

    for message in messages:
        filtered_items = []
        for item in message.items:
            if isinstance(item, UpdatedFileItem) and not item.invalid:
                files[item.file_path] = FileItem(file_path=item.file_path, content=item.content)
            elif isinstance(item, FileItem):
                files[item.file_path] = FileItem(file_path=item.file_path, content=item.content)
            else:
                filtered_items.append(item)
        message.items = filtered_items

    file_items = []
    for file_item in files.values():
        if repository:
            content = repository.get_file_content(file_path=file_item.file_path)
        else:
            content = file_item.content

        file_items.append(FileItem(file_path=file_item.file_path,
                                   content=content))

    LOG.info(f"Run verification  ({retries} tries left)...")
    verification_message = verifier.execute()
    verification_message.items.extend(file_items)

    failures = verification_message.find_items_by_type("verification_failure")
    if failures:
        if retries > 0 or len(failures) < last_run:
            log_dir = repository.repo_path / ".prompt_log"
            retries -= 1

            llm = create_openai_client(log_dir=log_dir, llm_name=smart_llm_name, temperature=0.0, streaming=False)

            test_fix_writer = CodeWriter(llm=llm,
                                         sys_prompt=FIX_TESTS_PROMPT,
                                         repository=repository,
                                         auto_mode=True)
            LOG.info(f"{len(failures)} verifications failed (last run {last_run}, retrying ({retries} left)...")
            response_messages = test_fix_writer.execute(incoming_messages=messages + [verification_message])
            return verify(repository, messages=messages + [verification_message] + response_messages,
                          retries=retries,
                          test_file=test_file,
                          last_run=len(failures))
        else:
            LOG.info(f"Verification failed, giving up...")

    return verification_message
