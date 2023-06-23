from ..basic_gpt_agent import basic_gpt_agent
from agbenchmark.Challenge import Challenge


# TODO: Make it so that you can specify for tests to only run if their prerequisites are met.
# Prerequisites here would be writing to a file (basic_abilities test).
# Should also check if prerequisites exists in regression file
def retrieval_1_mock(task: str, workspace: str):
    # Call the basic_gpt_agent to get a response.
    response = basic_gpt_agent(task)

    # Open the file in write mode.
    Challenge.write_to_file(workspace, "file_to_check.txt", response)
