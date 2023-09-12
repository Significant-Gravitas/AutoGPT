import glob
import importlib
import json
import os
import sys
import types
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pytest

from benchmark.utils.challenge import Challenge
from benchmark.utils.data_types import AgentBenchmarkConfig, ChallengeData, SuiteConfig
from benchmark.utils.utils import get_test_path

DATA_CATEGORY = {}


def setup_dummy_dependencies(
    file_datum: list[dict[str, Any]],
    challenge_class: Any,
    challenge_data: ChallengeData,
) -> None:
    """Sets up the dependencies if it's a suite. Creates tests that pass
    based on the main test run."""

    def create_test_func(test_name: str) -> Callable[[Any, dict[str, Any]], None]:
        # This function will return another function

        # Define a dummy test function that does nothing
        def setup_dependency_test(self: Any, scores: dict[str, Any]) -> None:
            scores = self.get_dummy_scores(test_name, scores)
            assert scores == 1

        return setup_dependency_test

    for datum in file_datum:
        DATA_CATEGORY[datum["name"]] = challenge_data.category[0]
        test_func = create_test_func(datum["name"])
        # TODO: replace this once I figure out actual dependencies
        test_func = pytest.mark.depends(on=[challenge_data.name], name=datum["name"])(
            test_func
        )
        test_func = pytest.mark.parametrize(
            "challenge_data",
            [None],
            indirect=True,
        )(test_func)

        # Add category markers
        for category in challenge_data.category:
            test_func = getattr(pytest.mark, category)(test_func)

        test_func = pytest.mark.usefixtures("scores")(test_func)
        setattr(challenge_class, f"test_{datum['name']}", test_func)


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
    challenge_class = types.new_class(data["name"], (Challenge,))
    print(challenge_location)
    # clean_challenge_location = get_test_path(challenge_location)
    setattr(challenge_class, "CHALLENGE_LOCATION", challenge_location)

    # in the case of a suite
    if isinstance(challenge_data, ChallengeData):
        if file_datum:  # same task suite
            setup_dummy_dependencies(file_datum, challenge_class, challenge_data)

        artifacts_location = str(Path(challenge_location).resolve())
        if "--test" in sys.argv or "--maintain" in sys.argv or "--improve" in sys.argv:
            artifacts_location = str(Path(challenge_location).resolve().parent.parent)
        setattr(
            challenge_class,
            "_data_cache",
            {challenge_location: challenge_data},
        )

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
            with open("challenges_already_beaten.json", "r") as f:
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
        request.node.answers = scores["answers"]  # store answers in request.node
        del scores["answers"]  # remove answers from scores
        request.node.scores = scores  # store scores in request.node
        assert 1 in scores["values"]

    # Parametrize the method here
    test_method = pytest.mark.parametrize(
        "challenge_data",
        [data],
        indirect=True,
    )(test_method)

    setattr(challenge_class, "test_method", test_method)

    # Attach the new class to a module so it can be discovered by pytest
    module = importlib.import_module(__name__)
    setattr(module, data["name"], challenge_class)


def create_single_suite_challenge(challenge_data: ChallengeData, path: Path) -> None:
    create_single_test(challenge_data, str(path))


def create_challenge(
    data: Dict[str, Any],
    json_file: str,
    suite_config: SuiteConfig | None,
    json_files: deque,
) -> deque:
    path = Path(json_file).resolve()
    print("Creating challenge for", path)
    if suite_config is not None:
        grandparent_dir = path.parent.parent

        # if its a single test running we dont care about the suite
        if "--test" in sys.argv or "--maintain" in sys.argv or "--improve" in sys.argv:
            challenge_data = suite_config.challenge_from_test_data(data)
            create_single_suite_challenge(challenge_data, path)
            return json_files

        # Get all data.json files within the grandparent directory
        suite_files = suite_config.get_data_paths(grandparent_dir)

        # Remove all data.json files from json_files list, except for current_file
        json_files = deque(
            file
            for file in json_files
            if file not in suite_files
            and Path(file).resolve() != Path(json_file).resolve()
        )

        suite_file_datum = [
            ChallengeData.get_json_from_path(suite_file)
            for suite_file in suite_files
            if suite_file != json_file
        ]

        file_datum = [data, *suite_file_datum]

        if suite_config.same_task:
            challenge_data = suite_config.challenge_from_datum(file_datum)

            create_single_test(
                challenge_data, str(grandparent_dir), file_datum=file_datum
            )
        else:
            reverse = suite_config.reverse_order

            # TODO: reversing doesn't work, for the same reason why the ordering of dummy tests doesn't work
            if reverse:
                paired_data = list(reversed(list(zip(file_datum, suite_files))))
            else:
                paired_data = list(zip(file_datum, suite_files))

            for file_data, file_path in paired_data:
                # if we're running in reverse we don't want dependencies to get in the way
                if reverse:
                    file_data["dependencies"] = []
                create_single_test(file_data, file_path)

    else:
        create_single_test(data, str(path))
    print("Creation complete for", path)

    return json_files


# if there's any suite.json files with that prefix


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

    if "--agent-config" in sys.argv:
        agent_benchmark_config_path = sys.argv[sys.argv.index("--agent-config") + 1]
    else:
        print(sys.argv)
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
    # for suites to know if the file has already been used to generate the tests
    # Dynamic class creation

    while json_files:
        json_file = (
            json_files.popleft()
        )  # Take and remove the first element from json_files
        if challenge_should_be_ignored(json_file):
            continue
        data = ChallengeData.get_json_from_path(json_file)
        suite_config = SuiteConfig.suite_data_if_suite(Path(json_file))

        commands = sys.argv
        # --by flag
        if "--category" in commands:
            categories = data.get("category", [])
            commands_set = set(commands)

            # Add the shared category if the conditions are met
            if suite_config and suite_config.same_task:
                # handled by if same_task is false in types
                categories += suite_config.shared_category  # type: ignore

            # Convert the combined list to a set
            categories_set = set(categories)

            # If there's no overlap with commands
            if not categories_set.intersection(commands_set):
                continue

        # --test flag, only run the test if it's the exact one specified
        test_flag = "--test" in commands
        if test_flag and data["name"] not in commands:
            continue

        # --maintain and --improve flag
        in_regression = regression_tests.get(data["name"], None)
        improve_flag = in_regression and "--improve" in commands
        maintain_flag = not in_regression and "--maintain" in commands
        if "--maintain" in commands and maintain_flag:
            continue
        elif "--improve" in commands and improve_flag:
            continue

        # "--suite flag
        if "--suite" in commands:
            if not suite_config:
                # not a test from a suite
                continue
            elif not any(command in data["name"] for command in commands):
                continue

            # elif (
            #     not any(command in data["name"] for command in commands)
            #     and suite_config.prefix not in data["name"]
            # ):
            #     # a part of the suite but not the one specified
            #     continue
        json_files = create_challenge(data, json_file, suite_config, json_files)

        if suite_config and not (test_flag or maintain_flag or improve_flag):
            print(f"Generated suite for {suite_config.prefix}.")
        else:
            print(f"Generated test for {data['name']}.")
    print("Test generation complete.")


def challenge_should_be_ignored(json_file):
    return "challenges/deprecated" in json_file or "challenges/library" in json_file


generate_tests()
