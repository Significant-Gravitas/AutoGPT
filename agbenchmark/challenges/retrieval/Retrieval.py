from agbenchmark.Challenge import Challenge
from agbenchmark.challenges.define_task_types import Ground


class RetrievalChallenge(Challenge):
    """Challenge for information-retrieval"""

    def scoring(self, content: str, ground: Ground):
        if ground.should_contain:
            for should_contain_word in ground.should_contain:
                if should_contain_word not in content:
                    return 0.0
                else:
                    print(
                        f"Word that should exist: {should_contain_word} exists in the content"
                    )

        if ground.should_not_contain:
            for should_not_contain_word in ground.should_not_contain:
                if should_not_contain_word in content:
                    return 0.0
                else:
                    print(
                        f"Word that should not exist: {should_not_contain_word} does not exist in the content"
                    )

        return 1.0
