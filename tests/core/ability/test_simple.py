"""
This file provides all the necessary pytests for autogpt/core/ability/simple.py.
"""
import asyncio
import logging
import typing
from typing import Any, Callable

import pytest

import autogpt.core.plugin.simple
from autogpt.core.ability.schema import AbilityResult
from autogpt.core.ability.simple import (
    AbilityConfiguration,
    AbilityRegistryConfiguration,
    AbilityRegistrySettings,
    SimpleAbilityRegistry,
)
from autogpt.core.memory.base import Memory
from autogpt.core.plugin.base import PluginLocation, PluginStorageFormat
from autogpt.core.resource.model_providers.schema import (
    LanguageModelFunction,
    LanguageModelMessage,
    LanguageModelProvider,
    LanguageModelProviderModelResponse,
    ModelProviderModelInfo,
    ModelProviderName,
    ModelProviderService,
)
from autogpt.workspace.workspace import Workspace

if typing.TYPE_CHECKING:
    from autogpt.core.plugin.base import PluginType


class TestAbilityRegistryConfiguration:
    """
    Provides some tests for AbilityRegistryConfiguration class.
    """

    def setup_method(self) -> None:
        # Define test data for abilities
        self.abilities_data = {
            "ability1": {
                "location": PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="non-existent.route",
                ),
                "packages_required": ["package1", "package2"],
            },
            "ability2": {
                "location": PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="non-existent.route",
                ),
                "packages_required": ["package3"],
            },
        }

        # mapping of ability'name to its configuration (ability_name -> AbilityConfiguration)
        self.ability_configurations = {
            ability_name: AbilityConfiguration(**ability_data)
            for ability_name, ability_data in self.abilities_data.items()
        }

    def test_ability_registry_configuration(self) -> None:
        # Create an instance of AbilityRegistryConfiguration with test data
        registry_config = AbilityRegistryConfiguration(
            abilities=self.ability_configurations
        )

        # Assert that the abilities dictionary is correctly set
        assert registry_config.abilities == self.ability_configurations

        # Test get_user_config method
        user_config = registry_config.get_user_config()

        # Verify that the user_config dictionary contains the expected keys and values
        assert "abilities" in user_config
        assert isinstance(user_config["abilities"], dict)
        assert len(user_config["abilities"]) == len(self.abilities_data)

        for ability_name, ability_data in self.abilities_data.items():
            assert ability_name in user_config["abilities"]
            assert isinstance(user_config["abilities"][ability_name], dict)

            # Validate the presence of specific fields in the user config
            assert "location" in user_config["abilities"][ability_name]
            # TODO: assert "packages_required" in user_config["abilities"][ability_name] FIXME: requires get_user_config() to return it

            # Validate the type of specific fields
            assert isinstance(user_config["abilities"][ability_name]["location"], dict)
            # TODO: assert isinstance(user_config["abilities"][ability_name]["packages_required"], list) FIXME: requires get_user_config() to return it


class TestAbilityRegistrySettings:
    """
    Provides some tests for AbilityRegistrySettings class.
    """

    def setup_method(self) -> None:
        # Define test data for abilities
        self.abilities_data = {
            "ability1": {
                "location": PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="non-existent.route",
                ),
                "packages_required": ["package1", "package2"],
            },
            "ability2": {
                "location": PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="non-existent.route",
                ),
                "packages_required": ["package3"],
            },
        }

        # mapping of ability'name to its configuration (ability_name -> AbilityConfiguration)
        self.ability_configurations = {
            ability_name: AbilityConfiguration(**ability_data)
            for ability_name, ability_data in self.abilities_data.items()
        }

    def test_ability_registry_settings(self) -> None:
        # Create an instance of AbilityRegistryConfiguration using the abilities dictionary
        registry_configuration = AbilityRegistryConfiguration(
            abilities=self.ability_configurations
        )

        # Create an instance of AbilityRegistrySettings with the AbilityRegistryConfiguration
        settings = AbilityRegistrySettings(
            name="ability-reg-settings-test",
            description="test desc.",
            configuration=registry_configuration,
        )

        # Assert that the configuration is correctly set
        assert settings.configuration == registry_configuration
        assert settings.name == "ability-reg-settings-test"
        assert settings.description == "test desc."


class TestSimpleAbilityRegistry:
    """
    Provides some tests for SimpleAbilityRegistry class.
    """

    class MockAbility:
        """ "
        Mock implementation of Ability class to be employed by various tests
        inside TestSimpleAbilityRegistry.
        """

        def __init__(self, **kwargs: Any) -> None:
            pass

        def name(self) -> str:
            return "mock_ability"

        def description(self) -> str:
            return "Mock ability description"

        def dump(self) -> dict:
            return {"name": "mock_ability", "description": "Mock ability description"}

        async def __call__(self, **kwargs: Any) -> AbilityResult:
            return AbilityResult(
                ability_name=self.name(),
                success=True,
                message="Mock ability result",
                ability_args={},
            )

    def setup_method(self) -> None:
        class MockSimpleAbilityPlugin:
            """
            A mock implementation of PluginService to be employed inside
            autogpt.core.ability.simple as SimplePluginService. In other words,
            it mimics the essential functionality of SimplePluginService without
            actually doing the real work that is done by SimplePluginService.
            """

            @staticmethod
            def get_plugin(plugin_location: PluginLocation | dict) -> "PluginType":
                return TestSimpleAbilityRegistry.MockAbility

        # overwrite SimplePluginService in autogpt.core.ability.simple with the
        # mock one
        autogpt.core.ability.simple.SimplePluginService = MockSimpleAbilityPlugin

        class MockLanguageModelProvider(LanguageModelProvider):
            """
            A mock implementation of LanguageModelProvider class to be employed
            when creating a SimpleAbilityRegistry class.
            """

            def get_remaining_budget(self) -> float:
                return 1.1  # any float would work

            def get_token_limit(self, model_name: str) -> int:
                return 1  # any int would work

            async def create_language_completion(
                self,
                model_prompt: list[LanguageModelMessage],
                functions: list[LanguageModelFunction],
                model_name: str,
                completion_parser: Callable[[dict], dict],
                **kwargs: Any,
            ) -> LanguageModelProviderModelResponse:
                mock_response = LanguageModelProviderModelResponse(
                    prompt_tokens_used=1,  # any int would work
                    completion_tokens_used=2,  # any int would work
                    model_info=ModelProviderModelInfo(
                        name="mock-openai-model-info-name",
                        provider_name=ModelProviderName.OPENAI,
                        service=ModelProviderService.LANGUAGE,
                    ),
                    content={"foo": "bar"},
                )
                return mock_response

        settings = AbilityRegistrySettings(
            name="ability-reg-settings-test",
            description="test desc.",
            configuration=AbilityRegistryConfiguration(
                abilities={
                    "ability1": AbilityConfiguration(
                        location=PluginLocation(
                            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                            storage_route="non-existent.route",
                        ),
                        packages_required=["package1"],
                    )
                }
            ),
        )

        logger = logging.getLogger(__name__)
        memory = Memory()
        workspace = Workspace(workspace_root="/tmp", restrict_to_workspace=True)
        model_providers = {ModelProviderName.OPENAI: MockLanguageModelProvider()}

        # Create an instance of SimpleAbilityRegistry with mock instances
        self.registry = SimpleAbilityRegistry(
            settings=settings,
            logger=logger,
            memory=memory,
            workspace=workspace,
            model_providers=model_providers,
        )

    # def test_register_ability(self) -> None:
    # Test the register_ability method
    # NOTE: register_ability method is implicityly called by the ctor of the
    # SimpleAbilityRegistry class

    def test_list_abilities(self) -> None:
        # Test the list_abilities method
        abilities_list = self.registry.list_abilities()
        assert abilities_list == ["mock_ability: Mock ability description"]

    def test_dump_abilities(self) -> None:
        # Test the dump_abilities method
        abilities_dump = self.registry.dump_abilities()
        assert abilities_dump == [
            {"name": "mock_ability", "description": "Mock ability description"}
        ]

    def test_get_ability(self) -> None:
        # Test the get_ability method
        ability = self.registry.get_ability("mock_ability")
        assert isinstance(ability, TestSimpleAbilityRegistry.MockAbility)
        with pytest.raises(ValueError):
            self.registry.get_ability("non-existing_ability")

    def test_perform(self) -> None:
        # Test the perform method
        result = asyncio.run(self.registry.perform("mock_ability"))
        assert isinstance(result, AbilityResult)
        assert result.success
        assert result.message == "Mock ability result"
