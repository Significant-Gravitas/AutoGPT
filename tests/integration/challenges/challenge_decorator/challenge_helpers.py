import os
from typing import Any, Callable, Dict, Optional, Tuple

from tests.integration.challenges.challenge_decorator.challenge import Challenge

CHALLENGE_PREFIX = "test_"


def create_challenge(
    func: Callable[..., Any], current_score: Dict[str, Any]
) -> Challenge:
    challenge_category, challenge_name = get_challenge_identifiers(func)
    max_level = get_max_level(current_score, challenge_category, challenge_name)
    current_level_beaten = get_current_level_beaten(
        current_score, challenge_category, challenge_name
    )
    return Challenge(
        name=challenge_name,
        category=challenge_category,
        max_level=max_level,
        current_level_beaten=int(current_level_beaten)
        if current_level_beaten
        else None,
    )


def get_challenge_identifiers(func: Callable[..., Any]) -> Tuple[str, str]:
    full_path = os.path.dirname(os.path.abspath(func.__code__.co_filename))
    challenge_category = os.path.basename(full_path)
    challenge_name = func.__name__.replace(CHALLENGE_PREFIX, "")
    return challenge_category, challenge_name


def get_max_level(
    current_score: Dict[str, Any],
    challenge_category: str,
    challenge_name: str,
) -> Optional[int]:
    return (
        current_score.get(challenge_category, {})
        .get(challenge_name, {})
        .get("max_level", None)
    )


def get_current_level_beaten(
    current_score: Dict[str, Any],
    challenge_category: str,
    challenge_name: str,
) -> Optional[int]:
    return (
        current_score.get(challenge_category, {})
        .get(challenge_name, {})
        .get("current_level_beaten", None)
    )
