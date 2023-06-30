import pytest
from agbenchmark.Challenge import Challenge
from agbenchmark.tests.basic_abilities.BasicChallenge import BasicChallenge
import os


class TestReadFile(BasicChallenge):
    """Testing if LLM can read a file"""

    @pytest.fixture(scope="module", autouse=True)
    def setup_module(self, workspace):
        Challenge.write_to_file(
            workspace, self.data.ground.files[0], "this is how we're doing"
        )

    def get_file_path(self) -> str:  # all tests must implement this method
        return os.path.join(os.path.dirname(__file__), "r_file_data.json")

    @pytest.mark.depends(on=["basic_write_file"], name="basic_read_file")
    def test_method(self, config):
        self.setup_challenge(config)
        files_contents = self.open_files(config["workspace"], self.data.ground.files)

        scores = []
        for file_content in files_contents:
            score = self.scoring(file_content, self.data.ground)
            print("Your score is:", score)
            scores.append(score)

        assert 1 in scores
