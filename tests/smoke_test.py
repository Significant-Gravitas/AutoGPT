"""Smoke test for the autogpt package."""
import os
import subprocess
import sys

import pytest

from autogpt.commands.file_operations import delete_file, read_file


@pytest.mark.integration_test
def test_write_file() -> None:
    """
    Test case to check if the write_file command can successfully write 'Hello World' to a file
    named 'hello_world.txt'.

    Read the current ai_settings.yaml file and store its content.
    """
    env_vars = {"MEMORY_BACKEND": "no_memory", "TEMPERATURE": "0"}
    ai_settings = None
    if os.path.exists("ai_settings.yaml"):
        with open("ai_settings.yaml", "r") as f:
            ai_settings = f.read()
        os.remove("ai_settings.yaml")

    try:
        if os.path.exists("hello_world.txt"):
            # Clean up any existing 'hello_world.txt' file before testing.
            delete_file("hello_world.txt")
        # Prepare input data for the test.
        input_data = """write_file-GPT
an AI designed to use the write_file command to write 'Hello World' into a file named "hello_world.txt" and then use the task_complete command to complete the task.
Use the write_file command to write 'Hello World' into a file named "hello_world.txt".
Use the task_complete command to complete the task.
Do not use any other commands.

y -5
EOF"""
        command = f"{sys.executable} -m autogpt"

        # Execute the script with the input data.
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            shell=True,
            env={**os.environ, **env_vars},
        )
        process.communicate(input_data.encode())

        # Read the content of the 'hello_world.txt' file created during the test.
        content = read_file("hello_world.txt")
    finally:
        if ai_settings:
            # Restore the original ai_settings.yaml file.
            with open("ai_settings.yaml", "w") as f:
                f.write(ai_settings)

    # Check if the content of the 'hello_world.txt' file is equal to 'Hello World'.
    assert content == "Hello World", f"Expected 'Hello World', got {content}"
