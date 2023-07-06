import os
from typing import Any, Dict

import pytest

from agbenchmark.challenges.memory.memory import MemoryChallenge


class TestRememberMultipleIds(MemoryChallenge):
    """The first memory challenge"""

    def get_file_path(self) -> str:  # all tests must implement this method
        return os.path.join(
            os.path.dirname(__file__), "remember_multiple_ids_data.json"
        )

    @pytest.mark.depends(
        name="test_remember_multiple_ids", depends=["test_basic_memory"]
    )
    def test_method(self, config: Dict[str, Any]) -> None:
        self.setup_challenge(config)

        files_contents = self.open_files(config["workspace"], self.data.ground.files)

        scores = []
        for file_content in files_contents:
            score = self.scoring(file_content, self.data.ground)
            print("Your score is:", score)
            scores.append(score)

        assert 1 in scores
