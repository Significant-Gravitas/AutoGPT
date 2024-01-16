import glob
import json
import logging
import math
import os
import subprocess
import sys
from abc import ABC
from pathlib import Path
from typing import Any, ClassVar, List

import pytest
from colorama import Fore, Style
from openai import OpenAI

from agbenchmark.agent_api_interface import run_api_agent
from agbenchmark.config import AgentBenchmarkConfig
from agbenchmark.utils.data_types import ChallengeData, Ground
from agbenchmark.utils.prompts import (
    END_PROMPT,
    FEW_SHOT_EXAMPLES,
    PROMPT_MAP,
    SCORING_MAP,
)

logger = logging.getLogger(__name__)

with open(
    Path(__file__).parent.parent / "challenges" / "optional_categories.json"
) as f:
    OPTIONAL_CATEGORIES: list[str] = json.load(f)["optional_categories"]


class Challenge(ABC):
    """The parent class to all specific challenges classes.
    Defines helper methods for running a challenge"""

    data: ChallengeData
    CHALLENGE_LOCATION: ClassVar[str]
    ARTIFACTS_LOCATION: ClassVar[str]
    scores: ClassVar[dict[str, Any]] = {}  # this is for suites

    @staticmethod
    def from_challenge_spec(spec_file: Path) -> type["Challenge"]:
        challenge_data = ChallengeData.parse_file(spec_file)

        challenge_class_name = f"Test{challenge_data.name}"
        logger.debug(f"Creating {challenge_class_name} from spec: {spec_file}")
        return type(
            challenge_class_name,
            (Challenge,),
            {
                "data": challenge_data,
                "CHALLENGE_LOCATION": str(spec_file),
                "ARTIFACTS_LOCATION": str(spec_file.resolve().parent),
            },
        )

    # Define test method within the dynamically created class
    @pytest.mark.asyncio
    async def test_method(
        self, config: AgentBenchmarkConfig, request: pytest.FixtureRequest
    ) -> None:
        # skip optional categories
        self.skip_optional_categories(config)

        # if os.environ.get("HELICONE_API_KEY"):
        #     from helicone.lock import HeliconeLockManager

        #     HeliconeLockManager.write_custom_property("challenge", self.data.name)

        timeout = self.data.cutoff or 60

        if request.config.getoption("--nc"):
            timeout = 100000
        elif cutoff := request.config.getoption("--cutoff"):
            timeout = int(cutoff)

        await self.run_challenge(config, timeout)

        scores = self.get_scores(config.temp_folder)
        request.node.answers = (
            scores["answers"] if request.config.getoption("--keep-answers") else None
        )
        del scores["answers"]  # remove answers from scores
        request.node.scores = scores  # store scores in request.node
        is_score_100 = 1 in scores["values"]

        assert is_score_100

    async def run_challenge(self, config: AgentBenchmarkConfig, cutoff: int) -> None:
        from agbenchmark.agent_interface import copy_artifacts_into_temp_folder

        if not self.data.task:
            return

        print(
            f"{Fore.MAGENTA + Style.BRIGHT}{'='*24} "
            f"Starting {self.data.name} challenge"
            f" {'='*24}{Style.RESET_ALL}"
        )
        print(f"{Fore.BLACK}Task: {self.data.task}{Fore.RESET}")

        await run_api_agent(self.data, config, self.ARTIFACTS_LOCATION, cutoff)

        # hidden files are added after the agent runs. Hidden files can be python test files.
        # We copy them in the temporary folder to make it easy to import the code produced by the agent
        artifact_paths = [
            self.ARTIFACTS_LOCATION,
            str(Path(self.CHALLENGE_LOCATION).parent),
        ]
        for path in artifact_paths:
            copy_artifacts_into_temp_folder(config.temp_folder, "custom_python", path)

    @staticmethod
    def get_artifacts_out(
        workspace: str | Path | dict[str, str], ground: Ground
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
                    cwd=os.path.abspath(workspace),
                    capture_output=True,
                    text=True,
                )
                if "error" in result.stderr or result.returncode != 0:
                    print(result.stderr)
                    assert False, result.stderr
                files_contents.append(f"Output: {result.stdout}\n")

        return files_contents

    @staticmethod
    def scoring(content: str, ground: Ground) -> float:
        print(f"{Fore.BLUE}Scoring content:{Style.RESET_ALL}", content)
        if ground.should_contain:
            for should_contain_word in ground.should_contain:
                if not getattr(ground, "case_sensitive", True):
                    should_contain_word = should_contain_word.lower()
                    content = content.lower()
                print_content = (
                    f"{Fore.BLUE}Word that should exist{Style.RESET_ALL}"
                    f" - {should_contain_word}:"
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
                print_content = (
                    f"{Fore.BLUE}Word that should not exist{Style.RESET_ALL}"
                    f" - {should_not_contain_word}:"
                )
                if should_not_contain_word in content:
                    print(print_content, "False")
                    return 0.0
                else:
                    print(print_content, "True")

        return 1.0

    @classmethod
    def llm_eval(cls, content: str, ground: Ground) -> float:
        openai_client = OpenAI()
        if os.getenv("IS_MOCK"):
            return 1.0

        # the validation for this is done in the Eval BaseModel
        scoring = SCORING_MAP[ground.eval.scoring]  # type: ignore
        prompt = PROMPT_MAP[ground.eval.template].format(  # type: ignore
            task=cls.data.task, scoring=scoring, answer=ground.answer, response=content
        )

        if ground.eval.examples:
            prompt += FEW_SHOT_EXAMPLES.format(examples=ground.eval.examples)

        prompt += END_PROMPT

        answer = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
            ],
        )

        return float(answer.choices[0].message.content)  # type: ignore

    @classmethod
    def get_scores(cls, workspace: Path) -> dict[str, Any]:
        scores = []
        scores_dict: Any = {}
        percentage = None
        answers = {}
        try:
            if cls.data.task == "" and os.getenv("IS_MOCK"):
                scores = [1.0]
                answers = {"mock": "This is a mock answer"}
            elif isinstance(cls.data.ground, Ground):
                files_contents = cls.get_artifacts_out(workspace, cls.data.ground)
                answers = {"answer": files_contents}
                for file_content in files_contents:
                    score = cls.scoring(file_content, cls.data.ground)
                    print(f"{Fore.GREEN}Your score is:{Style.RESET_ALL}", score)
                    scores.append(score)

                if cls.data.ground.eval.type == "llm":
                    llm_eval = cls.llm_eval("\n".join(files_contents), cls.data.ground)
                    if cls.data.ground.eval.scoring == "percentage":
                        scores.append(math.ceil(llm_eval / 100))
                    elif cls.data.ground.eval.scoring == "scale":
                        scores.append(math.ceil(llm_eval / 10))
                    print(f"{Fore.GREEN}Your score is:{Style.RESET_ALL}", llm_eval)

                    scores.append(llm_eval)
        except Exception as e:
            print("Error getting scores", e)

        scores_data = {
            "values": scores,
            "scores_obj": scores_dict,
            "percentage": percentage,
            "answers": answers,
        }

        cls.scores[cls.__name__] = scores_data

        return scores_data

    def get_dummy_scores(self, test_name: str, scores: dict[str, Any]) -> int | None:
        return 1  # remove this once this works
        if 1 in scores.get("scores_obj", {}).get(test_name, []):
            return 1

        return None

    @classmethod
    def skip_optional_categories(cls, config: AgentBenchmarkConfig) -> None:
        challenge_categories = set(c.value for c in cls.data.category)
        challenge_optional_categories = challenge_categories & set(OPTIONAL_CATEGORIES)
        if challenge_optional_categories and not (
            config.categories
            and set(challenge_optional_categories).issubset(set(config.categories))
        ):
            pytest.skip(
                f"Category {', '.join(challenge_optional_categories)} is optional, "
                "and not explicitly selected in the benchmark config."
            )
