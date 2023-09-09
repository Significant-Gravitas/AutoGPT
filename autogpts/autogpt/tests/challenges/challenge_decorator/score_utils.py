import json
import os
from typing import Any, Dict, Optional, Tuple

from tests.challenges.challenge_decorator.challenge import Challenge

CURRENT_SCORE_LOCATION = "../current_score"
NEW_SCORE_LOCATION = "../new_score"


def update_new_score(
    filename_new_score: str,
    new_score: Dict[str, Any],
    challenge: Challenge,
    new_max_level_beaten: Optional[int],
) -> None:
    write_new_score(new_score, challenge, new_max_level_beaten)
    write_new_score_to_file(new_score, filename_new_score)


def write_new_score(
    new_score: Dict[str, Any], challenge: Challenge, new_max_level_beaten: Optional[int]
) -> Dict[str, Any]:
    new_score.setdefault(challenge.category, {})
    new_score[challenge.category][challenge.name] = {
        "max_level_beaten": new_max_level_beaten,
        "max_level": challenge.max_level,
    }
    return new_score


def write_new_score_to_file(new_score: Dict[str, Any], filename: str) -> None:
    with open(filename, "w") as file:
        json.dump(new_score, file, indent=4)


def get_scores() -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    filename_current_score, filename_new_score = get_score_locations()
    current_score = load_json(filename_current_score)
    new_score = load_json(filename_new_score)
    return current_score, new_score, filename_new_score


def load_json(filename: str) -> Dict[str, Any]:
    if os.path.isfile(filename):
        with open(filename, "r") as file:
            return json.load(file)
    else:
        return {}


def get_score_locations() -> Tuple[str, str]:
    pid = os.getpid()
    project_root = os.path.dirname(os.path.abspath(__file__))
    filename_current_score = os.path.join(
        project_root, f"{CURRENT_SCORE_LOCATION}.json"
    )
    filename_new_score = os.path.join(project_root, f"{NEW_SCORE_LOCATION}_{pid}.json")
    return filename_current_score, filename_new_score
