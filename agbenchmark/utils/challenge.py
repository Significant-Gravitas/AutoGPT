import glob
import math
import os
import subprocess
import sys
from abc import ABC
from typing import Any, Dict, List

import openai
import pytest

from agbenchmark.agent_api_interface import run_api_agent
from agbenchmark.start_benchmark import OPTIONAL_CATEGORIES
from agbenchmark.utils.data_types import ChallengeData, Ground
from agbenchmark.utils.prompts import (
    END_PROMPT,
    FEW_SHOT_EXAMPLES,
    PROMPT_MAP,
    SCORING_MAP,
)
from agbenchmark.utils.utils import agent_eligibible_for_optional_categories


class Challenge(ABC):
    """The parent class to all specific challenges classes.
    Defines helper methods for running a challenge"""

    _data_cache: Dict[str, ChallengeData] = {}
    CHALLENGE_LOCATION: str = ""
    ARTIFACTS_LOCATION: str = ""  # this is for suites
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

    async def setup_challenge(self, config: Dict[str, Any], cutoff: int) -> None:
        if not self.task:
            return

        from agbenchmark.agent_interface import copy_artifacts_into_workspace, run_agent

        copy_artifacts_into_workspace(
            config["workspace"], "artifacts_in", self.ARTIFACTS_LOCATION
        )

        print(
            f"\033[1;35m============Starting {self.data.name} challenge============\033[0m"
        )
        print(f"\033[1;30mTask: {self.task}\033[0m")

        if "--mock" in sys.argv:
            print("Running mock agent")
            copy_artifacts_into_workspace(
                config["workspace"], "artifacts_out", self.ARTIFACTS_LOCATION
            )
        elif config.get("api_mode"):
            await run_api_agent(self.data, config, self.ARTIFACTS_LOCATION, cutoff)
        else:
            run_agent(self.task, cutoff)

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

    def get_artifacts_out(
        self, workspace: str | dict[str, str], ground: Ground
    ) -> List[str]:
        if isinstance(workspace, dict):
            workspace = workspace["output"]

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
                if ground.eval.type == "python":
                    result = subprocess.run(
                        [sys.executable, file_path],
                        cwd=os.path.abspath(workspace),
                        capture_output=True,
                        text=True,
                    )
                    if "error" in result.stderr or result.returncode != 0:
                        print(result.stderr)
                        assert False, result.stderr
                    files_contents.append(f"Output: {result.stdout}\n")
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

    def scoring(self, config: Dict[str, Any], content: str, ground: Ground) -> float:
        print("\033[1;34mScoring content:\033[0m", content)
        if ground.should_contain:
            for should_contain_word in ground.should_contain:
                print_content = (
                    f"\033[1;34mWord that should exist\033[0m - {should_contain_word}:"
                )
                if should_contain_word not in content:
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

    def llm_eval(self, config: Dict[str, Any], content: str, ground: Ground) -> float:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if "--mock" in sys.argv:
            return 1.0

        # the validation for this is done in the Eval BaseModel
        scoring = SCORING_MAP[ground.eval.scoring]  # type: ignore
        prompt = PROMPT_MAP[ground.eval.template].format(task=self.data.task, scoring=scoring, answer=ground.answer, response=content)  # type: ignore

        if ground.eval.examples:
            prompt += FEW_SHOT_EXAMPLES.format(examples=ground.eval.examples)

        prompt += END_PROMPT

        answer = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
            ],
        )

        return float(answer["choices"][0]["message"]["content"])  # type: ignore

    def get_scores(self, config: Dict[str, Any]) -> dict[str, Any]:
        scores = []
        scores_dict: Any = {}
        percentage = None

        try:
            if self.data.task == "" and "--mock" in sys.argv:
                scores = [1.0]
            elif isinstance(self.data.ground, Ground):
                files_contents = self.get_artifacts_out(
                    config["workspace"], self.data.ground
                )

                for file_content in files_contents:
                    score = self.scoring(config, file_content, self.data.ground)
                    print("\033[1;32mYour score is:\033[0m", score)
                    scores.append(score)

                if self.data.ground.eval.type == "llm":
                    llm_eval = self.llm_eval(
                        config, "\n".join(files_contents), self.data.ground
                    )
                    if self.data.ground.eval.scoring == "percentage":
                        scores.append(math.ceil(llm_eval / 100))
                    elif self.data.ground.eval.scoring == "scale":
                        scores.append(math.ceil(llm_eval / 10))
                    scores.append(llm_eval)
            elif isinstance(self.data.ground, dict):
                # if it's a dict then we know its a combined suite
                for ground_key in self.data.ground:
                    ground = self.data.ground[ground_key]
                    files_contents = self.get_artifacts_out(config["workspace"], ground)

                    for file_content in files_contents:
                        score = self.scoring(config, file_content, ground)
                        scores_dict.setdefault(ground_key, []).append(score)
                        print(
                            f"\033[1;35mScore for {ground_key}:\033[0m",
                            scores_dict[ground_key],
                        )

                    if ground.eval.type == "llm":
                        llm_eval = self.llm_eval(
                            config, "\n".join(files_contents), ground
                        )

                        if ground.eval.scoring == "percentage":
                            scores_dict[ground_key].append(math.ceil(llm_eval / 100))
                        elif ground.eval.scoring == "scale":
                            scores_dict[ground_key].append(math.ceil(llm_eval / 10))
                        scores_dict[ground_key].append(llm_eval)

                # Count the number of times the value 1.0 appears in the dictionary
                num_ones = sum(
                    1
                    for scores in scores_dict.values()
                    for score in scores
                    if score == 1.0
                )

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
        except Exception as e:
            print("Error getting scores", e)

        scores_data = {
            "values": scores,
            "scores_obj": scores_dict,
            "percentage": percentage,
        }

        self.scores[self.__class__.__name__] = scores_data

        return scores_data

    def get_dummy_scores(self, test_name: str, scores: dict[str, Any]) -> int | None:
        return 1  # remove this once this works
        if 1 in scores.get("scores_obj", {}).get(test_name, []):
            return 1

        return None

    def skip_optional_categories(self, config: Dict[str, Any]) -> None:
        challenge_category = self.data.category
        categories = [
            category
            for category in OPTIONAL_CATEGORIES
            if category in challenge_category
        ]
        if not agent_eligibible_for_optional_categories(
            categories, config.get("category", [])
        ):
            pytest.skip("Agent is not eligible for this category")
