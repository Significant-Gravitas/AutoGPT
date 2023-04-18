import concurrent
import os
import unittest

import pytest
import vcr

from autogpt.agent.agent_utils import create_agent
from autogpt.commands.file_operations import LOG_FILE, delete_file, read_file
from autogpt.config import AIConfig, Config
from autogpt.prompt import Prompt
from autogpt.workspace import WORKSPACE_PATH

current_file_dir = os.path.dirname(os.path.abspath(__file__))
# tests_directory = os.path.join(current_file_dir, 'tests')

my_vcr = vcr.VCR(
    cassette_library_dir=os.path.join(current_file_dir, "cassettes"),
    record_mode="once",
)


@pytest.mark.integration_test
def test_write_file() -> None:
    # if file exist
    file_name = "hello_world.txt"

    file_path_to_write_into = f"{WORKSPACE_PATH}/{file_name}"
    if os.path.exists(file_path_to_write_into):
        os.remove(file_path_to_write_into)
    file_logger_path = f"{WORKSPACE_PATH}/{LOG_FILE}"
    if os.path.exists(file_logger_path):
        os.remove(file_logger_path)

    delete_file(file_name)
    ai_config = AIConfig(
        ai_name="write_file-GPT",
        ai_role="an AI designed to use the write_file command to write 'Hello World' into a file named \"hello_world.txt\" and then use the task_complete command to complete the task.",
        ai_goals=[
            "Use the write_file command to write 'Hello World' into a file named \"hello_world.txt\".",
            "Use the task_complete command to complete the task.",
            "Do not use any other commands.",
        ],
    )
    prompt = Prompt(ai_config=ai_config)

    cfg = Config()
    cfg.set_continuous_mode(True)
    cfg.set_memory_backend("no_memory")
    cfg.set_temperature(0)
    agent = create_agent(prompt)
    try:
        with my_vcr.use_cassette("write_file.yaml"):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(agent.start_interaction_loop)
                try:
                    result = future.result(timeout=45)
                except concurrent.futures.TimeoutError:
                    assert False, "The process took longer than 45 seconds to complete."
    # catch system exit exceptions
    except SystemExit:  # the agent returns an exception when it shuts down
        content = ""
        content = read_file(file_name)
        os.remove(file_path_to_write_into)

        assert content == "Hello World", f"Expected 'Hello World', got {content}"


if __name__ == "__main__":
    unittest.main()
