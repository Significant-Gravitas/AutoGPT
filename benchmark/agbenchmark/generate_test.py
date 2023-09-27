import glob
import importlib
import json
import os
import sys
import types
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pytest

from agbenchmark.__main__ import CHALLENGES_ALREADY_BEATEN
from agbenchmark.agent_api_interface import append_updates_file
from agbenchmark.agent_protocol_client.models.step import Step
from agbenchmark.utils.challenge import Challenge
from agbenchmark.utils.data_types import AgentBenchmarkConfig, ChallengeData

DATA_CATEGORY = {}


def create_single_test(
    data: Dict[str, Any] | ChallengeData,
    challenge_location: str,
    file_datum: Optional[list[dict[str, Any]]] = None,
) -> None:
    challenge_data = None
    artifacts_location = None
    if isinstance(data, ChallengeData):
        challenge_data = data
        data = data.get_data()

    DATA_CATEGORY[data["name"]] = data["category"][0]

    # Define test class dynamically
    challenge_class = types.new_class(f"Test{data['name']}", (Challenge,))
    print(challenge_location)
    # clean_challenge_location = get_test_path(challenge_location)
    setattr(challenge_class, "CHALLENGE_LOCATION", challenge_location)

    setattr(
        challenge_class,
        "ARTIFACTS_LOCATION",
        artifacts_location or str(Path(challenge_location).resolve().parent),
    )

    # Define test method within the dynamically created class
    @pytest.mark.asyncio
    async def test_method(self, config: Dict[str, Any], request) -> None:  # type: ignore
        # create a random number between 0 and 1
        test_name = self.data.name

        try:
            with open(CHALLENGES_ALREADY_BEATEN, "r") as f:
                challenges_beaten_in_the_past = json.load(f)
        except:
            challenges_beaten_in_the_past = {}

        if request.config.getoption("--explore") and challenges_beaten_in_the_past.get(
            test_name, False
        ):
            return None

        # skip optional categories
        self.skip_optional_categories(config)

        from helicone.lock import HeliconeLockManager

        if os.environ.get("HELICONE_API_KEY"):
            HeliconeLockManager.write_custom_property("challenge", self.data.name)

        cutoff = self.data.cutoff or 60

        timeout = cutoff
        if "--nc" in sys.argv:
            timeout = 100000
        if "--cutoff" in sys.argv:
            timeout = int(sys.argv[sys.argv.index("--cutoff") + 1])

        await self.setup_challenge(config, timeout)

        scores = self.get_scores(config)
        request.node.answers = (
            scores["answers"] if "--keep-answers" in sys.argv else None
        )
        del scores["answers"]  # remove answers from scores
        request.node.scores = scores  # store scores in request.node
        is_score_100 = 1 in scores["values"]

        evaluation = "Correct!" if is_score_100 else "Incorrect."
        eval_step = Step(
            input=evaluation,
            additional_input=None,
            task_id="irrelevant, this step is a hack",
            step_id="irrelevant, this step is a hack",
            name="",
            status="created",
            output=None,
            additional_output=None,
            artifacts=[],
            is_last=True,
        )
        await append_updates_file(eval_step)

        assert is_score_100

    # Parametrize the method here
    test_method = pytest.mark.parametrize(
        "challenge_data",
        [data],
        indirect=True,
    )(test_method)

    setattr(challenge_class, "test_method", test_method)

    # Attach the new class to a module so it can be discovered by pytest
    module = importlib.import_module(__name__)
    setattr(module, f"Test{data['name']}", challenge_class)
    return challenge_class


def create_single_suite_challenge(challenge_data: ChallengeData, path: Path) -> None:
    create_single_test(challenge_data, str(path))


def create_challenge(
    data: Dict[str, Any],
    json_file: str,
    json_files: deque,
) -> Union[deque, Any]:
    path = Path(json_file).resolve()
    print("Creating challenge for", path)

    challenge_class = create_single_test(data, str(path))
    print("Creation complete for", path)

    return json_files, challenge_class


def generate_tests() -> None:  # sourcery skip: invert-any-all
    print("Generating tests...")

    challenges_path = os.path.join(os.path.dirname(__file__), "challenges")
    print(f"Looking for challenges in {challenges_path}...")

    json_files = deque(
        glob.glob(
            f"{challenges_path}/**/data.json",
            recursive=True,
        )
    )

    print(f"Found {len(json_files)} challenges.")
    print(f"Sample path: {json_files[0]}")

    agent_benchmark_config_path = str(Path.cwd() / "agbenchmark_config" / "config.json")
    try:
        with open(agent_benchmark_config_path, "r") as f:
            agent_benchmark_config = AgentBenchmarkConfig(**json.load(f))
            agent_benchmark_config.agent_benchmark_config_path = (
                agent_benchmark_config_path
            )
    except json.JSONDecodeError:
        print("Error: benchmark_config.json is not a valid JSON file.")
        raise

    regression_reports_path = agent_benchmark_config.get_regression_reports_path()
    if regression_reports_path and os.path.exists(regression_reports_path):
        with open(regression_reports_path, "r") as f:
            regression_tests = json.load(f)
    else:
        regression_tests = {}

    while json_files:
        json_file = (
            json_files.popleft()
        )  # Take and remove the first element from json_files
        if challenge_should_be_ignored(json_file):
            continue

        data = ChallengeData.get_json_from_path(json_file)

        commands = sys.argv
        # --by flag
        if "--category" in commands:
            categories = data.get("category", [])
            commands_set = set(commands)

            # Convert the combined list to a set
            categories_set = set(categories)

            # If there's no overlap with commands
            if not categories_set.intersection(commands_set):
                continue

        # --test flag, only run the test if it's the exact one specified
        tests = []
        for command in commands:
            if command.startswith("--test="):
                tests.append(command.split("=")[1])

        if tests and data["name"] not in tests:
            continue

        # --maintain and --improve flag
        in_regression = regression_tests.get(data["name"], None)
        improve_flag = in_regression and "--improve" in commands
        maintain_flag = not in_regression and "--maintain" in commands
        if "--maintain" in commands and maintain_flag:
            continue
        elif "--improve" in commands and improve_flag:
            continue
        json_files, challenge_class = create_challenge(data, json_file, json_files)

        print(f"Generated test for {data['name']}.")
    print("Test generation complete.")


def challenge_should_be_ignored(json_file):
    return "challenges/deprecated" in json_file or "challenges/library" in json_file


generate_tests()
