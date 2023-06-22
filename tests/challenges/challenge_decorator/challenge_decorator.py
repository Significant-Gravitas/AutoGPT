import os
from functools import wraps
from typing import Any, Callable, Optional

import pytest

from tests.challenges.challenge_decorator.challenge import Challenge
from tests.challenges.challenge_decorator.challenge_utils import create_challenge
from tests.challenges.challenge_decorator.score_utils import (
    get_scores,
    update_new_score,
)

MAX_LEVEL_TO_IMPROVE_ON = (
    1  # we will attempt to beat 1 level above the current level for now.
)

CHALLENGE_FAILED_MESSAGE = "Challenges can sometimes fail randomly, please run this test again and if it fails reach out to us on https://discord.gg/autogpt in the 'challenges' channel to let us know the challenge you're struggling with."


def challenge() -> Callable[[Callable[..., Any]], Callable[..., None]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., None]:
        @pytest.mark.requires_openai_api_key
        @pytest.mark.vcr
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> None:
            run_remaining = MAX_LEVEL_TO_IMPROVE_ON if Challenge.BEAT_CHALLENGES else 1
            original_error: Optional[Exception] = None

            while run_remaining > 0:
                current_score, new_score, new_score_location = get_scores()
                level_to_run = (
                    kwargs["level_to_run"] if "level_to_run" in kwargs else None
                )
                challenge = create_challenge(
                    func, current_score, Challenge.BEAT_CHALLENGES, level_to_run
                )
                if challenge.level_to_run is not None:
                    kwargs["level_to_run"] = challenge.level_to_run
                    kwargs["challenge_name"] = challenge.name
                    try:
                        func(*args, **kwargs)
                        challenge.succeeded = True
                    except AssertionError as err:
                        original_error = AssertionError(
                            f"{CHALLENGE_FAILED_MESSAGE}\n{err}"
                        )
                        challenge.succeeded = False
                    except Exception as err:
                        original_error = err
                        challenge.succeeded = False
                else:
                    challenge.skipped = True
                if os.environ.get("CI") == "true":
                    new_max_level_beaten = get_new_max_level_beaten(
                        challenge, Challenge.BEAT_CHALLENGES
                    )
                    update_new_score(
                        new_score_location, new_score, challenge, new_max_level_beaten
                    )
                if challenge.level_to_run is None:
                    pytest.skip("This test has not been unlocked yet.")

                if not challenge.succeeded:
                    if Challenge.BEAT_CHALLENGES or challenge.is_new_challenge:
                        pytest.xfail(str(original_error))
                    if original_error:
                        raise original_error
                run_remaining -= 1

        return wrapper

    return decorator


def get_new_max_level_beaten(
    challenge: Challenge, beat_challenges: bool
) -> Optional[int]:
    if challenge.succeeded:
        return challenge.level_to_run
    if challenge.skipped:
        return challenge.max_level_beaten
    # Challenge failed
    return challenge.max_level_beaten if beat_challenges else None
