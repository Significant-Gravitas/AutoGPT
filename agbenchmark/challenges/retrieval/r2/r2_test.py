import os
from pathlib import Path
from typing import Any, Dict

import pytest

from agbenchmark.challenges.retrieval.retrieval import RetrievalChallenge


class TestRetrieval2(RetrievalChallenge):
    """The first information-retrieval challenge"""

    def get_file_path(self) -> str:  # all tests must implement this method
        return os.path.join(os.path.dirname(__file__), "r2_data.json")

    @pytest.mark.depends(on=["basic_write_file"])
    def test_method(self, config: Dict[str, Any]) -> None:
        self.setup_challenge(config)

        workspace = Path(os.getcwd()) / config["workspace"]
        files_contents = self.open_files(workspace, self.data.ground.files)

        scores = []
        for file_content in files_contents:
            score = self.scoring(file_content, self.data.ground)
            print("Your score is:", score)
            scores.append(score)

        assert 1 in scores
