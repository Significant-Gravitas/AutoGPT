import logging
from typing import ClassVar

from forge.json.model import JSONSchema

from autogpt.core.ability.base import Ability, AbilityConfiguration
from autogpt.core.ability.schema import AbilityResult
from autogpt.core.plugin.simple import PluginLocation, PluginStorageFormat


class CreateNewAbility(Ability):
    default_configuration = AbilityConfiguration(
        location=PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            storage_route="autogpt.core.ability.builtins.CreateNewAbility",
        ),
    )

    def __init__(
        self,
        logger: logging.Logger,
        configuration: AbilityConfiguration,
    ):
        self._logger = logger
        self._configuration = configuration

    description: ClassVar[str] = "Create a new ability by writing python code."

    parameters: ClassVar[dict[str, JSONSchema]] = {
        "ability_name": JSONSchema(
            description="A meaningful and concise name for the new ability.",
            type=JSONSchema.Type.STRING,
            required=True,
        ),
        "description": JSONSchema(
            description=(
                "A detailed description of the ability and its uses, "
                "including any limitations."
            ),
            type=JSONSchema.Type.STRING,
            required=True,
        ),
        "arguments": JSONSchema(
            description="A list of arguments that the ability will accept.",
            type=JSONSchema.Type.ARRAY,
            items=JSONSchema(
                type=JSONSchema.Type.OBJECT,
                properties={
                    "name": JSONSchema(
                        description="The name of the argument.",
                        type=JSONSchema.Type.STRING,
                    ),
                    "type": JSONSchema(
                        description=(
                            "The type of the argument. "
                            "Must be a standard json schema type."
                        ),
                        type=JSONSchema.Type.STRING,
                    ),
                    "description": JSONSchema(
                        description=(
                            "A detailed description of the argument and its uses."
                        ),
                        type=JSONSchema.Type.STRING,
                    ),
                },
            ),
        ),
        "required_arguments": JSONSchema(
            description="A list of the names of the arguments that are required.",
            type=JSONSchema.Type.ARRAY,
            items=JSONSchema(
                description="The names of the arguments that are required.",
                type=JSONSchema.Type.STRING,
            ),
        ),
        "package_requirements": JSONSchema(
            description=(
                "A list of the names of the Python packages that are required to "
                "execute the ability."
            ),
            type=JSONSchema.Type.ARRAY,
            items=JSONSchema(
                description=(
                    "The of the Python package that is required to execute the ability."
                ),
                type=JSONSchema.Type.STRING,
            ),
        ),
        "code": JSONSchema(
            description=(
                "The Python code that will be executed when the ability is called."
            ),
            type=JSONSchema.Type.STRING,
            required=True,
        ),
    }

    async def __call__(
        self,
        ability_name: str,
        description: str,
        arguments: list[dict],
        required_arguments: list[str],
        package_requirements: list[str],
        code: str,
    ) -> AbilityResult:
        raise NotImplementedError
