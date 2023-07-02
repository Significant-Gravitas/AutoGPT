from pathlib import Path

import pytest
from agbenchmark.tests.basic_abilities.BasicChallenge import BasicChallenge
import os


class TestWriteFile(BasicChallenge):
    """Testing if LLM can write to a file"""

    def get_file_path(self) -> str:  # all tests must implement this method
        return os.path.join(os.path.dirname(__file__), "w_file_data.json")

    def test_method(self, config):
        self.setup_challenge(config)

        workspace = Path(os.getcwd()) / config['workspace']
        files_contents = self.open_files(workspace, self.data.ground.files)

        scores = []
        for file_content in files_contents:
            score = self.scoring(file_content, self.data.ground)
            print("Your score is:", score)
            scores.append(score)

        assert 1 in scores
