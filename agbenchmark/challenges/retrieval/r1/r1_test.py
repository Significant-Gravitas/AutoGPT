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
        files_contents = self.open_files(workspace, data.ground.files)

        scores = []
        for file_content in files_contents:
            score = self.scoring(file_content, data.ground)
            print("Your score is:", score)
            scores.append(score)

        assert 1 in scores
