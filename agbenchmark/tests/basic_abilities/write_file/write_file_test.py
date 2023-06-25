import pytest
from agbenchmark.challenges.define_task_types import ChallengeData
from agbenchmark.tests.basic_abilities.BasicChallenge import BasicChallenge
import os

data = ChallengeData.deserialize(
    os.path.join(os.path.dirname(__file__), "w_file_data.json")
)


class TestWriteFile(BasicChallenge):
    """Testing if LLM can write to a file"""

    @pytest.mark.parametrize(
        "server_response",
        [(data.task, data.mock_func)],
        indirect=True,
    )
    def test_write_file(self, workspace):
        file = self.open_file(workspace, data.ground.files[0])

        score = self.scoring(file, data.ground)

        print("You score is:", score)

        assert score
