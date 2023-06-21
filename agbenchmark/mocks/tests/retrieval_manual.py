from ..basic_gpt_agent import basic_gpt_agent
from agbenchmark.Challenge import Challenge


def mock_retrieval(task: str, workspace: str):
    # Call the basic_gpt_agent to get a response.
    response = basic_gpt_agent(task)

    # Open the file in write mode.
    Challenge.write_to_file(workspace, "file_to_check.txt", response)
