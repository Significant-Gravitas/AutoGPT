import glob
import importlib
import json
import logging
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any

import pytest
from agent_protocol_client.models.step import Step

from agbenchmark.agent_api_interface import append_updates_file
from agbenchmark.config import AgentBenchmarkConfig
from agbenchmark.utils.challenge import Challenge
from agbenchmark.utils.data_types import ChallengeData
from agbenchmark.utils.path_manager import PATH_MANAGER

DATA_CATEGORY = {}

logger = logging.getLogger(__name__)


def create_single_test(
    challenge_data: ChallengeData,
    challenge_location: Path,
) -> type[Challenge]:
    DATA_CATEGORY[challenge_data.name] = challenge_data.category[0].value

    # Define test method within the dynamically created class
    @pytest.mark.asyncio
    async def test_method(
        self: Challenge, config: dict[str, Any], request: pytest.FixtureRequest
    ) -> None:
        # create a random number between 0 and 1
        test_name = self.data.name

        try:
            with open(PATH_MANAGER.challenges_already_beaten, "r") as f:
                challenges_beaten_in_the_past = json.load(f)
        except FileNotFoundError:
            challenges_beaten_in_the_past = {}

        if request.config.getoption("--explore") and challenges_beaten_in_the_past.get(
            test_name, False
        ):
            return None

        # skip optional categories
        self.skip_optional_categories(config)

        if os.environ.get("HELICONE_API_KEY"):
            from helicone.lock import HeliconeLockManager

            HeliconeLockManager.write_custom_property("challenge", self.data.name)

        cutoff = self.data.cutoff or 60

        timeout = cutoff
        if "--nc" in sys.argv:
            timeout = 100000
        if "--cutoff" in sys.argv:
            timeout = int(sys.argv[sys.argv.index("--cutoff") + 1])

        await self.run_challenge(config, timeout)

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

    # Define test class dynamically
    challenge_class: type[Challenge] = type(
        f"Test{challenge_data.name}", (Challenge,), {"test_method": test_method}
    )
    logger.debug(f"Location of challenge spec: {challenge_location}")
    challenge_class.CHALLENGE_LOCATION = str(challenge_location)
    challenge_class.ARTIFACTS_LOCATION = str(challenge_location.resolve().parent)

    return challenge_class


def create_challenge(
    challenge_data: ChallengeData,
    json_file: str,
) -> type[Challenge]:
    path = Path(json_file).resolve()
    logger.debug(f"Creating challenge for {path}")

    challenge_class = create_single_test(challenge_data, path)
    logger.debug(f"Creation complete for {path}")

    return challenge_class


def load_challenges() -> None:
    logger.info("Loading challenges...")

    challenges_path = os.path.join(os.path.dirname(__file__), "challenges")
    logger.debug(f"Looking for challenges in {challenges_path}...")

    json_files = deque(
        glob.glob(
            f"{challenges_path}/**/data.json",
            recursive=True,
        )
    )

    logger.debug(f"Found {len(json_files)} challenges.")
    logger.debug(f"Sample path: {json_files[0]}")

    agent_benchmark_config = AgentBenchmarkConfig.load()

    regression_reports_path = agent_benchmark_config.regression_reports_path
    if regression_reports_path and os.path.exists(regression_reports_path):
        with open(regression_reports_path, "r") as f:
            regression_tests = json.load(f)
    else:
        regression_tests = {}

    while json_files:
        # Take and remove the first element from json_files
        json_file = json_files.popleft()
        if challenge_should_be_ignored(json_file):
            continue

        challenge_info = ChallengeData.parse_file(json_file)

        # TODO: move filtering/selection of challenges out of here
        commands = sys.argv
        # --by flag
        if "--category" in commands:
            categories = [c.value for c in challenge_info.category]
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

        if tests and challenge_info.name not in tests:
            continue

        # --maintain and --improve flag
        in_regression = regression_tests.get(challenge_info.name, None)
        improve_flag = in_regression and "--improve" in commands
        maintain_flag = not in_regression and "--maintain" in commands
        if "--maintain" in commands and maintain_flag:
            continue
        elif "--improve" in commands and improve_flag:
            continue
        challenge_class = create_challenge(challenge_info, json_file)

        logger.debug(f"Generated test for {challenge_info.name}")
        _add_challenge_to_module(challenge_class)

    logger.info("Loading challenges complete.")


def challenge_should_be_ignored(json_file_path: str):
    return (
        "challenges/deprecated" in json_file_path
        or "challenges/library" in json_file_path
    )


def _add_challenge_to_module(challenge: type[Challenge]):
    # Attach the Challenge class to this module so it can be discovered by pytest
    module = importlib.import_module(__name__)
    setattr(module, f"{challenge.__name__}", challenge)


load_challenges()
