from typing import Optional


class Challenge:
    BEAT_CHALLENGES = False
    DEFAULT_CHALLENGE_NAME = "default_challenge_name"

    def __init__(
        self,
        name: str,
        category: str,
        max_level: int,
        is_new_challenge: bool,
        max_level_beaten: Optional[int] = None,
        level_to_run: Optional[int] = None,
    ) -> None:
        self.name = name
        self.category = category
        self.max_level_beaten = max_level_beaten
        self.max_level = max_level
        self.succeeded = False
        self.skipped = False
        self.level_to_run = level_to_run
        self.is_new_challenge = is_new_challenge
