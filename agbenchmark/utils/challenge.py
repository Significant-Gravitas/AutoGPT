import glob
import os
import subprocess
import sys
from abc import ABC
from typing import Any, Dict, List

import openai

from agbenchmark.agent_interface import MOCK_FLAG
from agbenchmark.utils.data_types import ChallengeData, Ground


class Challenge(ABC):
    """The parent class to all specific challenges classes.
    Defines helper methods for running a challenge"""

    _data_cache: Dict[str, ChallengeData] = {}
    CHALLENGE_LOCATION: str = ""
    ARTIFACTS_LOCATION: str = ""  # this is for suites
    setup_dependencies: List[str] = []  # this is for suites
    scores: dict[str, Any] = {}  # this is for suites

    @property
    def data(self) -> ChallengeData:
        if self.CHALLENGE_LOCATION not in self._data_cache:
            self._data_cache[self.CHALLENGE_LOCATION] = ChallengeData.deserialize(
                self.CHALLENGE_LOCATION
            )
        return self._data_cache[self.CHALLENGE_LOCATION]

    @property
    def task(self) -> str:
        return self.data.task

    @property
    def dependencies(self) -> list:
        return self.data.dependencies

    def setup_challenge(self, config: Dict[str, Any], cutoff: int) -> None:
        from agbenchmark.agent_interface import copy_artifacts_into_workspace, run_agent

        copy_artifacts_into_workspace(
            config["workspace"], "artifacts_in", self.ARTIFACTS_LOCATION
        )

        print(
            f"\033[1;35m============Starting {self.data.name} challenge============\033[0m"
        )

        run_agent(self.task, config, self.ARTIFACTS_LOCATION, cutoff)

        # hidden files are added after the agent runs. Hidden files can be python test files.
        # We copy them in the workspace to make it easy to import the code produced by the agent

        copy_artifacts_into_workspace(
            config["workspace"], "custom_python", self.ARTIFACTS_LOCATION
        )

    def test_method(self, config: Dict[str, Any]) -> None:
        raise NotImplementedError

    @staticmethod
    def open_file(workspace: str, filename: str) -> str:
        script_dir = workspace
        workspace_dir = os.path.join(script_dir, filename)
        with open(workspace_dir, "r") as f:
            return f.read()

    def get_artifacts_out(self, workspace: str, ground: Ground) -> List[str]:
        script_dir = workspace
        files_contents = []

        for file_pattern in ground.files:
            # Check if it is a file extension
            if file_pattern.startswith("."):
                # Find all files with the given extension in the workspace
                matching_files = glob.glob(os.path.join(script_dir, "*" + file_pattern))
            else:
                # Otherwise, it is a specific file
                matching_files = [os.path.join(script_dir, file_pattern)]

            for file_path in matching_files:
                if ground.type == "execute_python_code":
                    result = subprocess.run(
                        [sys.executable, file_path],
                        cwd=os.path.abspath(workspace),
                        capture_output=True,
                        text=True,
                    )
                    files_contents.append(result.stdout)
                else:
                    with open(file_path, "r") as f:
                        files_contents.append(f.read())

        return files_contents

    @staticmethod
    def write_to_file(workspace: str, filename: str, content: str) -> None:
        script_dir = workspace
        print("Writing file at", script_dir)
        workspace_dir = os.path.join(script_dir, filename)

        # Open the file in write mode.
        with open(workspace_dir, "w") as f:
            # Write the content to the file.
            f.write(content)

    def get_filenames_in_workspace(self, workspace: str) -> List[str]:
        return [
            filename
            for filename in os.listdir(workspace)
            if os.path.isfile(os.path.join(workspace, filename))
        ]

    def scoring(self, content: str, ground: Ground) -> float:
        print("\033[1;34mScoring content:\033[0m", content)
        if ground.should_contain:
            for should_contain_word in ground.should_contain:
                print_content = (
                    f"\033[1;34mWord that should exist\033[0m - {should_contain_word}:"
                )
                if self.data.ground.type == "file_llm_evaluation":
                    return self.llm_eval(content, should_contain_word)
                elif should_contain_word not in content:
                    print(print_content, "False")
                    return 0.0
                else:
                    print(print_content, "True")

        if ground.should_not_contain:
            for should_not_contain_word in ground.should_not_contain:
                print_content = f"\033[1;34mWord that should not exist\033[0m - {should_not_contain_word}:"
                if should_not_contain_word in content:
                    print(print_content, "False")
                    return 0.0
                else:
                    print(print_content, "True")

        return 1.0

    def llm_eval(self, content: str, should_contain_word: str) -> float:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if MOCK_FLAG:
            return 1.0
        evaluation_question = f"""
QUESTION:
{should_contain_word} Answer with 0 for no, 1 for yes.
CONTENT:
{content}
ANSWER:

"""
        answer = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": evaluation_question},
            ],
        )
        return float(answer["choices"][0]["message"]["content"])

    def get_scores(self, config: Dict[str, Any]) -> dict[str, Any]:
        scores = []
        scores_dict = {}
        percentage = None

        if isinstance(self.data.ground, Ground):
            files_contents = self.get_artifacts_out(
                config["workspace"], self.data.ground
            )

            for file_content in files_contents:
                score = self.scoring(file_content, self.data.ground)
                print("\033[1;32mYour score is:\033[0m", score)
                scores.append(score)
        elif isinstance(self.data.ground, dict):
            # if it's a dict then we know its a combined suite
            for ground_key in self.data.ground:
                ground = self.data.ground[ground_key]
                files_contents = self.get_artifacts_out(config["workspace"], ground)

                for file_content in files_contents:
                    score = self.scoring(file_content, ground)
                    scores_dict[ground_key] = score
                    print(
                        f"\033[1;35mScore for {ground_key}:\033[0m",
                        scores_dict[ground_key],
                    )

            # Count the number of times the value 1.0 appears in the dictionary
            num_ones = sum(1 for score in scores_dict.values() if score == 1.0)

            # Calculate the percentage
            percentage = round((num_ones / len(scores_dict)) * 100, 2)

            # Print the result in green
            print(f"\033[1;92mPercentage of 1.0 scores:\033[0m {percentage}%")

            # TODO: in an ideal world it only returns 1.0 if all of the tests pass but then the dependencies break.
            # So for now we return 1.0 if there's any that pass
            if percentage > 0:
                scores.append(1.0)
                if percentage != 100:
                    print(
                        "\033[1;93mWARNING:\033[0m Your agent did not pass all the tests in the suite."
                    )

        scores_data = {
            "values": scores,
            "scores_obj": scores_dict,
            "percentage": percentage,
        }

        self.scores[self.__class__.__name__] = scores_data

        return scores_data

    def get_dummy_scores(self, test_name: str, scores: dict[str, Any]) -> int | None:
        if scores["scores_obj"][test_name] == 1:
            return 1

        return None
