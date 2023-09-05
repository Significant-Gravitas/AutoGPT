import os
from typing import Any, Callable, Dict, Optional, Tuple

from tests.challenges.challenge_decorator.challenge import Challenge

CHALLENGE_PREFIX = "test_"


def create_challenge(
    func: Callable[..., Any],
    current_score: Dict[str, Any],
    is_beat_challenges: bool,
    level_to_run: Optional[int] = None,
) -> Challenge:
    challenge_category, challenge_name = get_challenge_identifiers(func)
    is_new_challenge = challenge_name not in current_score.get(challenge_category, {})
    max_level = get_max_level(current_score, challenge_category, challenge_name)
    max_level_beaten = get_max_level_beaten(
        current_score, challenge_category, challenge_name
    )
    level_to_run = get_level_to_run(
        is_beat_challenges, level_to_run, max_level, max_level_beaten, is_new_challenge
    )

    return Challenge(
        name=challenge_name,
        category=challenge_category,
        max_level=max_level,
        max_level_beaten=max_level_beaten,
        level_to_run=level_to_run,
        is_new_challenge=is_new_challenge,
    )


def get_level_to_run(
    is_beat_challenges: bool,
    level_to_run: Optional[int],
    max_level: int,
    max_level_beaten: Optional[int],
    is_new_challenge: bool,
) -> Optional[int]:
    if is_new_challenge:
        return 1
    if level_to_run is not None:
        if level_to_run > max_level:
            raise ValueError(
                f"Level to run ({level_to_run}) is greater than max level ({max_level})"
            )
        return level_to_run
    if is_beat_challenges:
        if max_level_beaten == max_level:
            return None
        return 1 if max_level_beaten is None else max_level_beaten + 1
    return max_level_beaten


def get_challenge_identifiers(func: Callable[..., Any]) -> Tuple[str, str]:
    full_path = os.path.dirname(os.path.abspath(func.__code__.co_filename))
    challenge_category = os.path.basename(full_path)
    challenge_name = func.__name__.replace(CHALLENGE_PREFIX, "")
    return challenge_category, challenge_name


def get_max_level(
    current_score: Dict[str, Any],
    challenge_category: str,
    challenge_name: str,
) -> int:
    return (
        current_score.get(challenge_category, {})
        .get(challenge_name, {})
        .get("max_level", 1)
    )


def get_max_level_beaten(
    current_score: Dict[str, Any],
    challenge_category: str,
    challenge_name: str,
) -> Optional[int]:
    return (
        current_score.get(challenge_category, {})
        .get(challenge_name, {})
        .get("max_level_beaten", None)
    )
