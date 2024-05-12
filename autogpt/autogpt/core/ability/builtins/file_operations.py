import logging
import os
from typing import ClassVar

from forge.json.model import JSONSchema

from autogpt.core.ability.base import Ability, AbilityConfiguration
from autogpt.core.ability.schema import AbilityResult, ContentType, Knowledge
from autogpt.core.plugin.simple import PluginLocation, PluginStorageFormat
from autogpt.core.workspace import Workspace


class ReadFile(Ability):
    default_configuration = AbilityConfiguration(
        location=PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            storage_route="autogpt.core.ability.builtins.ReadFile",
        ),
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

    description: ClassVar[str] = "Read and parse all text from a file."

    parameters: ClassVar[dict[str, JSONSchema]] = {
        "filename": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The name of the file to read.",
        ),
    }

    def _check_preconditions(self, filename: str) -> AbilityResult | None:
        message = ""
        try:
            pass
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
                ability_name=self.name(),
                ability_args={"filename": filename},
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
            new_knowledge = Knowledge(
                content="\n\n".join([element.text for element in elements]),
                content_type=ContentType.TEXT,
                content_metadata={"filename": filename},
            )
            success = True
            message = f"File {file_path} read successfully."
        except IOError as e:
            new_knowledge = None
            success = False
            message = str(e)

        return AbilityResult(
            ability_name=self.name(),
            ability_args={"filename": filename},
            success=success,
            message=message,
            new_knowledge=new_knowledge,
        )


class WriteFile(Ability):
    default_configuration = AbilityConfiguration(
        location=PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            storage_route="autogpt.core.ability.builtins.WriteFile",
        ),
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

    description: ClassVar[str] = "Write text to a file."

    parameters: ClassVar[dict[str, JSONSchema]] = {
        "filename": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The name of the file to write.",
        ),
        "contents": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The contents of the file to write.",
        ),
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
                ability_name=self.name(),
                ability_args={"filename": filename, "contents": contents},
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
            success = False
            message = str(e)

        return AbilityResult(
            ability_name=self.name(),
            ability_args={"filename": filename},
            success=success,
            message=message,
        )
