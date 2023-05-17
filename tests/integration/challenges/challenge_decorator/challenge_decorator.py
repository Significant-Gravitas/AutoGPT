import json
import os
from functools import wraps
from typing import Any, Callable, Dict, Tuple

import pytest

from tests.integration.challenges.challenge_decorator.challenge_helpers import (
    create_challenge,
)
from tests.integration.challenges.challenge_decorator.update_new_score import (
    update_new_score,
)

CURRENT_SCORE_LOCATION = "../current_score.json"
NEW_SCORE_LOCATION = "../new_score.json"


def challenge(func: Callable[..., Any]) -> Callable[..., None]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        current_score, new_score, new_score_location = get_scores()

        challenge = create_challenge(func, current_score)

        challenge.run(func, args, kwargs)

        update_new_score(new_score_location, new_score, challenge)
        if challenge.skipped:
            pytest.skip("This test has not been unlocked yet.")

        if not challenge.succeeded:
            raise AssertionError("Challenge failed")

    return wrapper


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
    project_root = os.path.dirname(os.path.abspath(__file__))
    filename_current_score = os.path.join(project_root, CURRENT_SCORE_LOCATION)
    filename_new_score = os.path.join(project_root, NEW_SCORE_LOCATION)
    return filename_current_score, filename_new_score
