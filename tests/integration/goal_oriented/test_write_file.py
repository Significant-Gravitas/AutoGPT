import concurrent
import json
import os
import re
import unittest

import pytest
import vcr

from autogpt.agent import Agent
from autogpt.commands.file_operations import LOG_FILE, delete_file, read_file
from autogpt.config import AIConfig, Config, check_openai_api_key
from autogpt.memory import get_memory
from autogpt.prompt import Prompt
from autogpt.workspace import WORKSPACE_PATH


def replace_timestamp_in_request(request):
    # Check if the request body contains a JSON object

    try:
        if not request.body:
            return request
        body = json.loads(request.body)
    except ValueError:
        return request

    if "messages" not in body:
        return request

    for message in body["messages"]:
        if "content" in message and "role" in message and message["role"] == "system":
            timestamp_regex = re.compile(r"\w{3} \w{3} \d{2} \d{2}:\d{2}:\d{2} \d{4}")
            message["content"] = timestamp_regex.sub(
                "Tue Jan 01 00:00:00 2000", message["content"]
            )

    request.body = json.dumps(body)
    return request


current_file_dir = os.path.dirname(os.path.abspath(__file__))
# tests_directory = os.path.join(current_file_dir, 'tests')

my_vcr = vcr.VCR(
    cassette_library_dir=os.path.join(current_file_dir, "cassettes"),
    record_mode="once",
    before_record_request=replace_timestamp_in_request,
)


CFG = Config()


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
    agent = create_writer_agent()
    try:
        with my_vcr.use_cassette(
            "write_file.yaml",
            filter_headers=[
                "authorization",
                "X-OpenAI-Client-User-Agent",
                "User-Agent",
            ],
        ):
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


def create_writer_agent():
    ai_config = AIConfig(
        ai_name="write_file-GPT",
        ai_role="an AI designed to use the write_file command to write 'Hello World' into a file named \"hello_world.txt\" and then use the task_complete command to complete the task.",
        ai_goals=[
            "Use the write_file command to write 'Hello World' into a file named \"hello_world.txt\".",
            "Use the task_complete command to complete the task.",
            "Do not use any other commands.",
        ],
    )
    memory = get_memory(CFG, init=True)
    prompt = Prompt(ai_config=ai_config)
    agent = Agent(
        ai_name="",
        memory=memory,
        full_message_history=[],
        next_action_count=0,
        system_prompt=prompt.system_prompt,
        triggering_prompt=prompt.triggering_prompt,
    )
    CFG.set_continuous_mode(True)
    CFG.set_memory_backend("no_memory")
    CFG.set_temperature(0)
    # CFG.use_azure(0)
    return agent


if __name__ == "__main__":
    unittest.main()
