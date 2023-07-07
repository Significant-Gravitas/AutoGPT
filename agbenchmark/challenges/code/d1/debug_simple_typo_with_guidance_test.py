from typing import Any, Dict

import pytest

from agbenchmark.challenges.code.code import CodeChallenge


class TestDebugSimpleTypoWithGuidance(CodeChallenge):
    """The first memory challenge"""

    @pytest.mark.depends(name="test_debug_simple_typo_with_guidance")
    def test_method(self, config: Dict[str, Any]) -> None:
        self.setup_challenge(config)

        files_contents = self.get_artifacts_out(
            config["workspace"], self.data.ground.files
        )

        scores = []
        for file_content in files_contents:
            score = self.scoring(file_content, self.data.ground)
            print("Your score is:", score)
            scores.append(score)

        assert 1 in scores
