import glob
import importlib
import logging
import os
from collections import deque
from pathlib import Path

from agbenchmark.utils.challenge import Challenge
from agbenchmark.utils.data_types import ChallengeData

DATA_CATEGORY = {}

logger = logging.getLogger(__name__)


def create_challenge_from_spec_file(spec_file: Path) -> type[Challenge]:
    challenge = Challenge.from_challenge_spec(spec_file)
    DATA_CATEGORY[challenge.data.name] = challenge.data.category[0].value
    return challenge


def create_challenge_from_spec_file_path(spec_file_path: str) -> type[Challenge]:
    spec_file = Path(spec_file_path).resolve()
    return create_challenge_from_spec_file(spec_file)


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

    loaded, ignored = 0, 0
    while json_files:
        # Take and remove the first element from json_files
        json_file = json_files.popleft()
        if challenge_should_be_ignored(json_file):
            ignored += 1
            continue

        challenge_info = ChallengeData.parse_file(json_file)

        challenge_class = create_challenge_from_spec_file_path(json_file)

        logger.debug(f"Generated test for {challenge_info.name}")
        _add_challenge_to_module(challenge_class)
        loaded += 1

    logger.info(f"Loading challenges complete: loaded {loaded}, ignored {ignored}.")


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
