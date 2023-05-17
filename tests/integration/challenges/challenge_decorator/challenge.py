import contextlib
from typing import Any, Callable, Dict, Optional, Tuple


class Challenge:
    def __init__(
        self,
        name: str,
        category: str,
        max_level: Optional[int],
        current_level_beaten: Optional[int],
    ) -> None:
        self.name = name
        self.category = category
        self.current_level_beaten = current_level_beaten
        self.max_level = max_level
        self.succeeded = False
        self.skipped = False

    def run(
        self, func: Callable[..., Any], args: tuple, kwargs: Dict[str, Any]
    ) -> None:
        if self.current_level_beaten is None:
            self.skipped = True
            return

        kwargs = self.add_level_if_not_present(kwargs)
        self.run_challenge_and_update_result(func, *args, **kwargs)

    def add_level_if_not_present(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if "level_to_run" in kwargs:
            kwargs = self.choose_challenge_level(kwargs)
        return kwargs

    def run_challenge_and_update_result(
        self, func: Callable[..., Any], *args: Tuple[Any, ...], **kwargs: Dict[str, Any]
    ) -> None:
        with contextlib.suppress(AssertionError):
            func(*args, **kwargs)
            self.succeeded = True

    def choose_challenge_level(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        kwargs["level_to_run"] = self.current_level_beaten
        return kwargs
