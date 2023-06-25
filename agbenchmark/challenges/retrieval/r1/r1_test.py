import pytest
from agbenchmark.challenges.retrieval.Retrieval import RetrievalChallenge
from agbenchmark.challenges.define_task_types import ChallengeData, Ground
import os


data = ChallengeData.deserialize(
    os.path.join(os.path.dirname(__file__), "r1_data.json")
)


class TestRetrieval1(RetrievalChallenge):
    """The first information-retrieval challenge"""

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
    def test_retrieval(self, workspace, current_challenge_data):
        file = self.open_file(workspace, data.ground.files[0])

        score = self.scoring(file, data.ground)

        print("You score is:", score)

        assert score
