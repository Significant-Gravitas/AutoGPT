from typing import List

from ghostcoder import FileRepository
from ghostcoder.actions import CodeWriter
from ghostcoder.benchmark.utils import create_openai_client
from ghostcoder.schema import Message, TextItem, FileItem
from ..forge_log import ForgeLogger
from .registry import ability

logger = ForgeLogger(__name__)


@ability(
    name="write_code",
    description="Use this to write code, provide as detailed instructions as possible in free text and files that should be updated or created",
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
    description="Use this to write or update unit tests. Provide as detailed instructions as possible in free text and the file that should be tested and the file where the tests should be written.",
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

    log_dir = agent.workspace.base_path / ".prompt_log"
    repo_dir = agent.workspace.base_path / task_id

    smart_llm_name = "gpt-4"

    llm = create_openai_client(log_dir=log_dir, llm_name=smart_llm_name, temperature=0.0, streaming=False)

    repository = FileRepository(repo_path=repo_dir, use_git=False)

    code_writer = CodeWriter(llm=llm,
                             repository=repository,
                             allow_hallucinated_files=True,
                             auto_mode=True)

    for file_item in file_items:
        if not file_item.content and repository:
            logger.debug(f"Get current file content for {file_item.file_path}")
            content = repository.get_file_content(file_path=file_item.file_path)
            if content:
                file_item.content = content
            elif file_item.new:
                file_item.content = ""

    message = Message(sender="Human",
                      items=[TextItem(text=task.input), TextItem(text=instructions)] + file_items)

    outgoing_messages = code_writer.execute(incoming_messages=[message])

    text_output = "\n".join([item.text for item in outgoing_messages[0].find_items_by_type("text")])
    return text_output
