import logging
import os

from autogpt.core.ability.base import Ability, AbilityConfiguration
from autogpt.core.ability.schema import AbilityResult
from autogpt.core.workspace import Workspace


class ReadFile(Ability):
    default_configuration = AbilityConfiguration(
        packages_required=["unstructured"],
        workspace_required=True,
    )

    def __init__(
        self,
        logger: logging.Logger,
        workspace: Workspace,
    ):
        self._logger = logger
        self._workspace = workspace

    @property
    def description(self) -> str:
        return "Read and parse all text from a file."

    @property
    def arguments(self) -> dict:
        return {
            "filename": {
                "type": "string",
                "description": "The name of the file to read.",
            },
        }

    def _check_preconditions(self, filename: str) -> AbilityResult | None:
        message = ""
        try:
            import unstructured
        except ImportError:
            message = "Package unstructured is not installed."

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

        from unstructured.partition.auto import partition

        file_path = self._workspace.get_path(filename)
        try:
            elements = partition(str(file_path))
            # TODO: Lots of other potentially useful information is available
            #   in the partitioned file. Consider returning more of it.
            data = "\n\n".join([element.text for element in elements])
            success = True
            message = f"File {file_path} read successfully."
        except IOError as e:
            data = None
            success = False
            message = str(e)

        return AbilityResult(
            success=success,
            message=message,
            data=data,
        )


class WriteFile(Ability):
    default_configuration = AbilityConfiguration(
        packages_required=["os"],
        workspace_required=True,
    )

    def __init__(
        self,
        logger: logging.Logger,
        workspace: Workspace,
    ):
        self._logger = logger
        self._workspace = workspace

    @property
    def description(self) -> str:
        return "Write text to a file."

    @property
    def arguments(self) -> dict:
        return {
            "filename": {
                "type": "string",
                "description": "The name of the file to write.",
            },
            "contents": {
                "type": "string",
                "description": "The contents of the file to write.",
            },
        }

    def _check_preconditions(
        self, filename: str, contents: str
    ) -> AbilityResult | None:
        message = ""
        try:
            file_path = self._workspace.get_path(filename)
            if file_path.exists():
                message = f"File {filename} already exists."
            if len(contents):
                message = f"File {filename} was not given any content."
        except ValueError as e:
            message = str(e)

        if message:
            return AbilityResult(
                success=False,
                message=message,
                data=None,
            )

    def __call__(self, filename: str, contents: str) -> AbilityResult:
        if result := self._check_preconditions(filename, contents):
            return result

        file_path = self._workspace.get_path(filename)
        try:
            directory = os.path.dirname(file_path)
            os.makedirs(directory)
            with open(filename, "w", encoding="utf-8") as f:
                f.write(contents)
            success = True
            message = f"File {file_path} written successfully."
        except IOError as e:
            data = None
            success = False
            message = str(e)

        return AbilityResult(
            success=success,
            message=message,
            data=data,
        )


class ListFiles(Ability):
    default_configuration = AbilityConfiguration(
        packages_required=["os"],
        workspace_required=True,
    )

    def __init__(
        self,
        logger: logging.Logger,
        workspace: Workspace,
    ):
        self._logger = logger
        self._workspace = workspace

    @property
    def description(self) -> str:
        return "Lists Files in a Directory."

    @property
    def arguments(self) -> dict:
        return {
            "directory": {
                "type": "string",
                "description": "The directory to list the files",
            }
        }

    def _check_preconditions(self, directory: str) -> AbilityResult | None:
        message = ""
        try:
            directory = self._workspace.get_path(directory)

            if not os.path.exists(directory):
                message = f"Given directory: {directory} does not exist."
            elif not os.path.isdir(directory):
                message = f"Given directory: {directory} is not a directory."
        except ValueError as e:
            message = str(e)

        if message:
            return AbilityResult(
                success=False,
                message=message,
                data=None,
            )

    def __call__(self, directory: str) -> AbilityResult:
        if result := self._check_preconditions(directory):
            return result

        file_path = self._workspace.get_path(directory)
        try:
            found_files = []

            for root, _, files in os.walk(file_path):
                for file in files:
                    if file.startswith("."):
                        continue
                    relative_path = os.path.relpath(
                        os.path.join(root, file), self._workspace.root
                    )
                    found_files.append(relative_path)
            success = True
            message = f"Files listed successfully."
            data = found_files
        except IOError as e:
            data = None
            success = False
            message = str(e)

        return AbilityResult(
            success=success,
            message=message,
            data=data,
        )
