import glob
import math
import os
import subprocess
import sys
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List

import openai
import pytest

from agbenchmark.__main__ import OPTIONAL_CATEGORIES, TEMP_FOLDER_ABS_PATH
from agbenchmark.agent_api_interface import run_api_agent
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
        from agbenchmark.agent_interface import copy_artifacts_into_temp_folder

        if not self.task:
            return

        print(
            f"\033[1;35m============Starting {self.data.name} challenge============\033[0m"
        )
        print(f"\033[1;30mTask: {self.task}\033[0m")

        await run_api_agent(self.data, config, self.ARTIFACTS_LOCATION, cutoff)

        # hidden files are added after the agent runs. Hidden files can be python test files.
        # We copy them in the temporary folder to make it easy to import the code produced by the agent
        artifact_paths = [
            self.ARTIFACTS_LOCATION,
            str(Path(self.CHALLENGE_LOCATION).parent),
        ]
        for path in artifact_paths:
            copy_artifacts_into_temp_folder(TEMP_FOLDER_ABS_PATH, "custom_python", path)

    def test_method(self, config: Dict[str, Any]) -> None:
        raise NotImplementedError

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
        else:
            if ground.eval.type == "pytest":
                result = subprocess.run(
                    [sys.executable, "-m", "pytest"],
                    cwd=TEMP_FOLDER_ABS_PATH,
                    capture_output=True,
                    text=True,
                )
                if "error" in result.stderr or result.returncode != 0:
                    print(result.stderr)
                    assert False, result.stderr
                files_contents.append(f"Output: {result.stdout}\n")

        return files_contents

    def scoring(self, config: Dict[str, Any], content: str, ground: Ground) -> float:
        print("\033[1;34mScoring content:\033[0m", content)
        if ground.should_contain:
            for should_contain_word in ground.should_contain:
                if not getattr(ground, "case_sensitive", True):
                    should_contain_word = should_contain_word.lower()
                    content = content.lower()
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
                if not getattr(ground, "case_sensitive", True):
                    should_not_contain_word = should_not_contain_word.lower()
                    content = content.lower()
                print_content = f"\033[1;34mWord that should not exist\033[0m - {should_not_contain_word}:"
                if should_not_contain_word in content:
                    print(print_content, "False")
                    return 0.0
                else:
                    print(print_content, "True")

        return 1.0

    def llm_eval(self, config: Dict[str, Any], content: str, ground: Ground) -> float:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if os.getenv("IS_MOCK"):
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
        answers = {}
        try:
            if self.data.task == "" and os.getenv("IS_MOCK"):
                scores = [1.0]
                answers = {"mock": "This is a mock answer"}
            elif isinstance(self.data.ground, Ground):
                files_contents = self.get_artifacts_out(
                    TEMP_FOLDER_ABS_PATH, self.data.ground
                )
                answers = {"answer": files_contents}
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
                    print("\033[1;32mYour score is:\033[0m", llm_eval)

                    scores.append(llm_eval)
        except Exception as e:
            print("Error getting scores", e)

        scores_data = {
            "values": scores,
            "scores_obj": scores_dict,
            "percentage": percentage,
            "answers": answers,
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
