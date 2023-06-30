from agbenchmark.challenges.retrieval.Retrieval import RetrievalChallenge
import os


class TestRetrieval1(RetrievalChallenge):
    """The first information-retrieval challenge"""

    def get_file_path(self) -> str:  # all tests must implement this method
        return os.path.join(os.path.dirname(__file__), "r1_data.json")

    def test_method(self, config):
        self.setup_challenge(config)
        files_contents = self.open_files(config["workspace"], self.data.ground.files)

        scores = []
        for file_content in files_contents:
            score = self.scoring(file_content, self.data.ground)
            print("Your score is:", score)
            scores.append(score)

        assert 1 in scores
