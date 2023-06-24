from agbenchmark.Challenge import Challenge
from ..basic_gpt_agent import basic_gpt_agent


def basic_read_file_mock(task: str, workspace: str):
    """
    This mock reads a file and returns its content.
    """

    file_contents = Challenge.open_file(workspace, "file_to_check.txt")

    Challenge.write_to_file(
        workspace, "file_to_check.txt", f"random string: {file_contents}"
    )


def basic_write_file_mock(task: str, workspace: str):
    """
    This mock writes to a file (creates one if it doesn't exist)
    """

    # Call the basic_gpt_agent to get a response.
    response = basic_gpt_agent(task)

    # Open the file in write mode.
    Challenge.write_to_file(workspace, "file_to_check.txt", response)
