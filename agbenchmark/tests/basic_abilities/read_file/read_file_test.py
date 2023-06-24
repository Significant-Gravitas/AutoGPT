import pytest
from agbenchmark.challenges.define_task_types import ChallengeData
from agbenchmark.Challenge import Challenge
import os

data = ChallengeData.deserialize(
    os.path.join(os.path.dirname(__file__), "r_file_data.json")
)


@pytest.fixture(scope="module", autouse=True)
def setup_module(workspace):
    if data.ground.should_contain:
        Challenge.write_to_file(
            workspace, data.ground.files[0], "this is how we're doing"
        )


class TestReadFile(Challenge):
    """Testing if LLM can read a file"""

    @pytest.mark.parametrize(
        "server_response",
        [(data.task, data.mock_func)],
        indirect=True,
    )
    @pytest.mark.basic
    def test_retrieval(
        self, workspace
    ):  # create_file simply there for the function to depend on the fixture
        file = self.open_file(workspace, data.ground.files[0])

        score = self.scoring(file, data.ground)

        print("You score is:", score)

        assert score
