import concurrent
import os
import unittest

import vcr
import requests

from autogpt.agent import Agent
from autogpt.commands.command import CommandRegistry
from autogpt.commands.file_operations import delete_file, read_file
from autogpt.commands.web_requests   import scrape_text, scrape_links
from autogpt.config import AIConfig, Config
from autogpt.memory import get_memory
from tests.integration.goal_oriented.vcr_helper import before_record_request
from tests.utils import requires_api_key

current_file_dir = os.path.dirname(os.path.abspath(__file__))

my_vcr = vcr.VCR(
    cassette_library_dir=os.path.join(current_file_dir, "cassettes"),
    record_mode="new_episodes",
    before_record_request=before_record_request,
)

CFG = Config()

@requires_api_key("OPENAI_API_KEY")
def test_browse_website(workspace) -> None:
    CFG.workspace_path = workspace.root
    CFG.file_logger_path = os.path.join(workspace.root, "file_logger.txt")

    file_name = str(workspace.get_path("test_scrape.txt"))
    agent = create_browse_agent(workspace)
    try:
        with my_vcr.use_cassette(
            "scrape_text.vcr.yml",
            filter_headers=[
                "authorization",
                "X-OpenAI-Client-User-Agent",
                "User-Agent",
            ],
        ):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(agent.start_interaction_loop)
                try:
                    result = future.result(timeout=40)
                except concurrent.futures.TimeoutError:
                    assert False, "The process took longer than 40 seconds to complete."
    # catch system exit exceptions
    except SystemExit:  # the agent returns an exception when it shuts down
        content = read_file(file_name)
        assert "£25.89" in content, f"Expected £25.89, got {content}"


def create_browse_agent(workspace):
    command_registry = CommandRegistry()
    command_registry.import_commands("autogpt.commands.file_operations")
    command_registry.import_commands("autogpt.commands.web_requests")
    command_registry.import_commands("autogpt.app")

    ai_config = AIConfig(
        ai_name="scrape_website-GPT",
        ai_role="an AI designed to use the scrape_text command to visit http://books.toscrape.com/catalogue/meditations_33/index.html, scrape the price of the book and write it to a file named \"test_scrape.txt\", and use the task_complete command to complete the task.",
        ai_goals=[
            "Use the scrape_text command to visit http://books.toscrape.com/catalogue/meditations_33/index.html and obtain the price of the book.",
            "Write the price of the book to a file named \"test_scrape.txt\".",
            "Use the task_complete command to complete the task.",
            "Do not use any other commands.",
        ],
    )
    ai_config.command_registry = command_registry
    memory = get_memory(CFG, init=True)
    triggering_prompt = (
        "Determine which next command to use, and respond using the"
        " format specified above:"
    )
    system_prompt = ai_config.construct_full_prompt()

    agent = Agent(
        ai_name="",
        memory=memory,
        full_message_history=[],
        command_registry=command_registry,
        config=ai_config,
        next_action_count=0,
        system_prompt=system_prompt,
        triggering_prompt=triggering_prompt,
        workspace_directory=workspace.root,
    )
    CFG.set_continuous_mode(True)
    CFG.set_memory_backend("no_memory")
    CFG.set_temperature(0)
    os.environ["TIKTOKEN_CACHE_DIR"] = ""

    return agent


if __name__ == "__main__":
    unittest.main()
