import pytest
from agbenchmark.challenges.define_task_types import ChallengeData
from agbenchmark.Challenge import Challenge
from agbenchmark.tests.basic_abilities.BasicChallenge import BasicChallenge
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


class TestReadFile(BasicChallenge):
    """Testing if LLM can read a file"""

    @pytest.mark.parametrize(
        "server_response",
        [(data.task, data.mock_func)],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "regression_data",
        [data],
        indirect=True,
    )
    @pytest.mark.depends(on=data.dependencies)
    def test_read_file(self, workspace):
        files_contents = self.open_files(workspace, data.ground.files)

        scores = []
        for file_content in files_contents:
            score = self.scoring(file_content, data.ground)
            print("Your score is:", score)
            scores.append(score)

        assert 1 in scores
