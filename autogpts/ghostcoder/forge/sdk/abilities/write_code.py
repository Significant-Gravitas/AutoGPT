from pathlib import Path
from typing import List

from ghostcoder import FileRepository
from ghostcoder.actions import CodeWriter
from ghostcoder.actions.verify.code_verifier import CodeVerifier
from ghostcoder.actions.write_code.prompt import FIX_TESTS_PROMPT
from ghostcoder.benchmark.utils import create_openai_client
from ghostcoder.codeblocks import create_parser, CodeBlockType
from ghostcoder.schema import Message, TextItem, FileItem, UpdatedFileItem
from ..forge_log import ForgeLogger
from .registry import ability

LOG = ForgeLogger(__name__)

smart_llm_name = "gpt-4"
basic_llm_name = "gpt-3.5-turbo"

log_dir = Path(".prompt_log")
max_retries = 3

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
    description="Use this to write unit tests. Provide as detailed instructions as possible in free text and the file that should be tested and the file where the tests should be written.",
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
    file_items = [FileItem(file_path=file_to_test, readonly=True), FileItem(file_path=test_file)]
    return await run_code_writer(agent, task_id, instructions, file_items)


async def run_code_writer(agent, task_id: str, instructions: str, file_items: List[FileItem] = []):
    task = await agent.db.get_task(task_id)

    repo_dir = agent.workspace.base_path / task_id

    llm = create_openai_client(log_dir=log_dir, llm_name=smart_llm_name, temperature=0.0, streaming=False)

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

    message = Message(sender="Human",
                      items=[TextItem(text=task.input), TextItem(text=instructions)] + file_items)

    outgoing_messages = code_writer.execute(incoming_messages=[message])
    outgoing_messages.extend(verify(repository, messages=[message] + outgoing_messages))

    artifacts = []
    for file_item in outgoing_messages[0].find_items_by_type("updated_file"):
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

    return outgoing_messages[-1].to_prompt()


def verify(repository: FileRepository, messages: List[Message], retry: int = 0, last_run: int = 0) -> [Message]:
    verifier = CodeVerifier(repository=repository, language="python")

    updated_files = dict()

    for message in messages:
        for item in message.items:
            if isinstance(item, UpdatedFileItem) and not item.invalid:
                updated_files[item.file_path] = item

    if not updated_files:
        # TODO: Handle if no files where updated in last run?
        return []

    file_items = []
    for file_item in updated_files.values():
        if repository:
            content = repository.get_file_content(file_path=file_item.file_path)
        else:
            content = file_item.content

        file_items.append(FileItem(file_path=file_item.file_path,
                                   content=content,
                                   invalid=file_item.invalid))

    outgoing_messages = []

    LOG.info(f"Updated files, verifying...")
    verification_message = verifier.execute()
    outgoing_messages.append(verification_message)

    failures = verification_message.find_items_by_type("verification_failure")
    if failures:
        if retry < max_retries or len(failures) < last_run:
            verification_message.items.extend(file_items)

            retry += 1
            incoming_messages = make_summary(messages)

            llm = create_openai_client(log_dir=log_dir, llm_name=smart_llm_name, temperature=0.0, streaming=False)

            test_fix_writer = CodeWriter(llm=llm,
                                         sys_prompt=FIX_TESTS_PROMPT,
                                         repository=repository,
                                         auto_mode=True)
            LOG.info(f"{len(failures)} verifications failed (last run {last_run}, retrying ({retry}/{max_retries})...")
            incoming_messages.append(verification_message)
            response_messages = test_fix_writer.execute(incoming_messages=incoming_messages)
            return verify(repository, messages=messages + [verification_message] + response_messages,
                          retry=retry,
                          last_run=len(failures))
        else:
            LOG.info(f"Verification failed, giving up...")

    return outgoing_messages


def make_summary(messages: List[Message]) -> List[Message]:
    summarized_messages = []
    sys_prompt = """Make a short summary of the provided message."""

    for message in messages:
        if message.sender == "Human":
            text_items = message.find_items_by_type("text")
            summarized_messages.append(Message(sender=message.sender, items=text_items))
        else:
            if not message.summary:
                llm = create_openai_client(log_dir=log_dir, llm_name=basic_llm_name, temperature=0.0, streaming=False)
                message.summary, stats = llm.generate(sys_prompt, messages=[message])
                LOG.debug(f"Created summary {stats.json}")
            if message.summary:
                summarized_messages.append(Message(sender=message.sender, items=[TextItem(text=message.summary)]))

    return summarized_messages
