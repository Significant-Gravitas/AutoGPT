import json
import os
from typing import Any, Dict

from tests.integration.challenges.challenge_decorator.challenge import Challenge


def update_new_score(
    filename_new_score: str, new_score: Dict[str, Any], challenge: Challenge
) -> None:
    if os.environ.get("CI") == "true":
        write_new_score(new_score, challenge)
        write_new_score_to_file(new_score, filename_new_score)


def write_new_score(new_score: Dict[str, Any], challenge: Challenge) -> Dict[str, Any]:
    # Initialize the folder and method if they are not yet in the results
    current_level_beaten = (
        challenge.current_level_beaten if challenge.succeeded else None
    )
    new_score.setdefault(challenge.category, {})
    new_score[challenge.category][challenge.name] = {
        "current_level_beaten": current_level_beaten,
        "max_level": challenge.max_level,
    }
    return new_score


def write_new_score_to_file(new_score: Dict[str, Any], filename: str) -> None:
    with open(filename, "w") as file:
        json.dump(new_score, file, indent=4)
