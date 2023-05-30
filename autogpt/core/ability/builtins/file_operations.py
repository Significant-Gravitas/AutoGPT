import logging
from pathlib import Path

from autogpt.core.ability.base import Ability
from autogpt.core.ability.schema import (
    AbilityArguments,
    AbilityRequirements,
    AbilityResult,
)
from autogpt.core.workspace import Workspace

FILE_OPERATIONS_LOG_PATH = Path("logs/file_operations.log")


class ReadFile(Ability):
    def __init__(
        self,
        logger: logging.Logger,
        workspace: Workspace,
    ):
        self._logger = logger
        self._workspace = workspace

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read a file and return its contents."

    @property
    def arguments(self) -> list[str]:
        return [AbilityArguments.FILENAME]

    @property
    def requirements(self) -> AbilityRequirements:
        return AbilityRequirements(
            packages=["charset_normalizer"],
            workspace=True,
        )

    def _check_preconditions(self, filename: str) -> AbilityResult | None:
        message = ""
        try:
            import charset_normalizer
        except ImportError:
            message = "Package charset_normalizer is not installed."

        try:
            file_path = self._workspace.get_path(filename)
            if not file_path.exists():
                message = f"File {filename} does not exist."
            if not file_path.is_file():
                message = f"{filename} is not a file."
        except ValueError as e:
            message = str(e)

        if message:
            return AbilityResult(
                success=False,
                message=message,
                data=None,
            )

    def __call__(self, filename: str) -> AbilityResult:
        if result := self._check_preconditions(filename):
            return result

        from charset_normalizer import from_path

        file_path = self._workspace.get_path(filename)
        try:
            charset_match = from_path(file_path).best()
            encoding = charset_match.encoding
            self._logger.debug(f"Read file '{filename}' with encoding '{encoding}'")
            success = True
            message = str(charset_match)
        except IOError as e:
            success = False
            message = str(e)

        return AbilityResult(
            success=success,
            message=message,
            data=None,
        )
