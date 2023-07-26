"""
This file provides pytests for the classes and codes defined in
autogpt/core/ability/base.py location.
"""
import abc
from importlib import import_module
from pprint import pformat
from typing import Any, ClassVar, Dict, List

import pydantic
import pytest
from inflection import underscore

from autogpt.core.ability import Ability, AbilityRegistry, AbilityResult
from autogpt.core.ability.base import AbilityConfiguration
from autogpt.core.ability.schema import Knowledge
from autogpt.core.ability.simple import (
    AbilityRegistryConfiguration,
    AbilityRegistrySettings,
)
from autogpt.core.planning.simple import LanguageModelConfiguration
from autogpt.core.plugin.base import (
    PluginLocation,
    PluginStorageFormat,
    SystemConfiguration,
)
from autogpt.core.resource.model_providers.schema import ModelProviderName


class MockDerivedAbility(Ability):
    """
    This is a mock class simulating implementation of Ability class. It is
    used inside the various test methods of TestAbility class in the following.
    """

    default_configuration = AbilityConfiguration(
        location=PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            # storage_route="tests.core.ability.test_base.MockDerivedAbility",
            storage_route="test_base.MockDerivedAbility",
        ),
    )

    @classmethod
    def description(cls) -> str:
        return "Test Ability Description"

    @classmethod
    def arguments(cls) -> dict[str, str]:
        return {"arg1": "int", "arg2": "str"}

    async def __call__(self, *args: Any, **kwargs: Any) -> AbilityResult:
        arg1 = kwargs.get("arg1")
        return AbilityResult(
            success=True,
            ability_name="Test-name",
            ability_args={"arg1": str(arg1)},
            message="Test msg",
        )


class MockDerivedAbilityRegistry(AbilityRegistry):
    """
    This is a mock class simulating implementation of AbilityRegistry class. It
    is used inside the various test methods of TestAbilityRegistry class in the
    following.
    """

    def __init__(
        self,
        settings: AbilityRegistrySettings,
    ):
        self.abilities: List[Ability] = []

        for (
            ability_name,
            ability_configuration,
        ) in settings.configuration.abilities.items():
            self.register_ability(ability_name, ability_configuration)

    def register_ability(
        self, ability_name: str, ability_configuration: AbilityConfiguration
    ) -> None:
        # storage_route = test_base.MockDerivedAbility
        (
            module_path,
            _,
            class_name,
        ) = ability_configuration.location.storage_route.rpartition(".")
        ability_class = getattr(import_module(module_path), class_name)
        ability_args: Dict[str, Any] = {}
        ability = ability_class(**ability_args)
        self.abilities.append(ability)

    def list_abilities(self) -> list[str]:
        return [
            f"{ability.name()}: {ability.description()}" for ability in self.abilities
        ]

    def dump_abilities(self) -> list[dict]:
        return [ability.dump() for ability in self.abilities]

    def get_ability(self, ability_name: str) -> Ability:
        for ability in self.abilities:
            if ability.name() == ability_name:
                return ability
        raise ValueError(f"Ability '{ability_name}' not found.")

    async def perform(self, ability_name: str, **kwargs: Any) -> AbilityResult:
        ability = self.get_ability(ability_name)
        return await ability(**kwargs)


class TestAbilityRegistry:
    """
    Provides various tests for AbilityRegistry class.
    """

    MOCK_ABILITIES = {MockDerivedAbility.name(): MockDerivedAbility}

    @staticmethod
    def test_base_methods_exist() -> None:
        assert issubclass(AbilityRegistry, abc.ABC)
        assert hasattr(AbilityRegistry, "register_ability")
        assert hasattr(AbilityRegistry, "list_abilities")
        assert hasattr(AbilityRegistry, "dump_abilities")
        assert hasattr(AbilityRegistry, "get_ability")
        assert hasattr(AbilityRegistry, "perform")
        assert callable(AbilityRegistry.dump_abilities)
        assert callable(AbilityRegistry.get_ability)
        assert callable(AbilityRegistry.perform)
        assert callable(AbilityRegistry.list_abilities)
        assert callable(AbilityRegistry.register_ability)

    def setup_method(self) -> None:
        ability_registry_settings = AbilityRegistrySettings(
            name="mock_derived_ability_registry",
            description="A mock ability registry for testing.",
            configuration=AbilityRegistryConfiguration(
                abilities={
                    ability_name: ability.default_configuration
                    for ability_name, ability in self.MOCK_ABILITIES.items()
                },
            ),
        )
        self.ability_registry = MockDerivedAbilityRegistry(ability_registry_settings)

    def test_get_ability(self) -> None:
        # Test if the ability can be retrieved from the registry
        ability = self.ability_registry.get_ability("mock_derived_ability")
        assert isinstance(ability, MockDerivedAbility)

        with pytest.raises(ValueError):
            self.ability_registry.get_ability("DOESN't EXIST")

    def test_list_abilities(self) -> None:
        # Test if the list of abilities returns all registered abilities
        abilities = self.ability_registry.list_abilities()
        assert len(abilities) == len(self.MOCK_ABILITIES)
        assert abilities[0] == "mock_derived_ability: Test Ability Description"

    @pytest.mark.asyncio
    async def test_perform_ability(self) -> None:
        # Test if the ability can be performed and returns a valid AbilityResult
        result = await self.ability_registry.perform("mock_derived_ability", arg1=42)
        assert isinstance(result, AbilityResult)
        assert result.success
        assert result.ability_name == "Test-name"
        assert result.ability_args == {"arg1": "42"}
        assert result.message == "Test msg"

    def test_dump_abilities(self) -> None:
        # Test all the abilities are dumped appropirately
        dumped_abilities = self.ability_registry.dump_abilities()
        assert len(dumped_abilities) == len(self.MOCK_ABILITIES)

        # check the returned type
        assert isinstance(dumped_abilities, list)
        for dumped in dumped_abilities:
            assert isinstance(dumped, dict)


class TestAbility:
    """
    Provides various tests for Ability class
    """

    def setup_method(self) -> None:
        self.ability = MockDerivedAbility()

    @staticmethod
    def test_base_variables_exist() -> None:
        # assert 'default_configuration' exists in Ability class
        assert "default_configuration" in Ability.__annotations__

        # assert 'default_configuration' has the correct type
        default_configuration_type = Ability.__annotations__["default_configuration"]
        assert default_configuration_type is ClassVar[AbilityConfiguration]

    @staticmethod
    def test_base_methods_exist() -> None:
        # Check if the base methods are present in the Ability class
        assert issubclass(Ability, abc.ABC)
        assert hasattr(Ability, "name")
        assert callable(Ability.name)
        assert hasattr(Ability, "description")
        assert callable(Ability.description)
        assert hasattr(Ability, "arguments")
        assert callable(Ability.arguments)
        assert hasattr(Ability, "required_arguments")
        assert callable(Ability.required_arguments)
        assert hasattr(Ability, "__call__")
        assert callable(Ability.__call__)
        assert hasattr(Ability, "__str__")
        assert callable(Ability.__str__)
        assert hasattr(Ability, "dump")
        assert callable(Ability.dump)

    def test_ability_name(self) -> None:
        # Test if the name method returns the correct name for the Ability class
        # instance (snake_case)
        assert self.ability.name() == underscore(
            self.ability.__class__.__name__
        )  # "mock_derived_ability"
        assert isinstance(self.ability.name(), str)

    def test_ability_description(self) -> None:
        # Test if the description method returns the correct ability description
        assert self.ability.description() == "Test Ability Description"
        assert isinstance(self.ability.description(), str)

    def test_ability_arguments(self) -> None:
        # Test if the arguments method returns the correct ability arguments
        assert self.ability.arguments() == {"arg1": "int", "arg2": "str"}
        assert isinstance(self.ability.arguments(), dict)
        assert isinstance(self.ability.arguments()["arg1"], str)

    def test_ability_required_arguments(self) -> None:
        # Test if the required_arguments method returns the correct ability arguments
        assert self.ability.required_arguments() == []
        assert isinstance(self.ability.required_arguments(), list)

    @pytest.mark.asyncio
    async def test_ability_call(self) -> None:
        # Test if the __call__ method returns a valid AbilityResult
        result = await self.ability(arg1=42)
        assert isinstance(result, AbilityResult)
        assert result.success
        assert result.ability_name == "Test-name"
        assert result.ability_args == {"arg1": "42"}
        assert result.message == "Test msg"
        # Test if result.knowledge can be either None or an instance of Knowledge
        assert result.new_knowledge is None or isinstance(
            result.new_knowledge, Knowledge
        )

    def test_ability_str(self) -> None:
        # Test if the __str__ method returns the correct string representation
        assert str(self.ability) == pformat(self.ability.dump())

    def test_ability_dump(self) -> None:
        # Test if the dump method returns the correct dictionary representation of the ability
        dumped = self.ability.dump()
        assert dumped["name"] == underscore(
            self.ability.__class__.__name__
        )  # evalued: mock_derived_ability
        assert dumped["description"] == "Test Ability Description"
        assert dumped["parameters"]["type"] == "object"
        assert dumped["parameters"]["properties"] == {"arg1": "int", "arg2": "str"}
        assert dumped["parameters"]["required"] == []


class TestAbilityConfiguration:
    """
    Provides necessary tests for the AbilityConfiguration class/struct.
    """

    @staticmethod
    def test_ability_configuration_is_subclass_of_system_configuration() -> None:
        assert issubclass(AbilityConfiguration, SystemConfiguration)

    @staticmethod
    def test_default_values() -> None:
        ability_config = AbilityConfiguration(
            location=PluginLocation(
                storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                storage_route="autogopt.core.etc",
            )
        )

        assert ability_config.packages_required == []
        assert ability_config.language_model_required is None
        assert ability_config.memory_provider_required is False
        assert ability_config.workspace_required is False

    @staticmethod
    def test_user_config_fields_included() -> None:
        language_model_config = LanguageModelConfiguration(
            model_name="model_name",
            provider_name=ModelProviderName.OPENAI,
            temperature=0.8,
        )
        ability_config = AbilityConfiguration(
            location=PluginLocation(
                storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                storage_route="route",
            ),
            packages_required=["package1", "package2"],
            language_model_required=language_model_config,
            memory_provider_required=True,
            workspace_required=True,
        )

        user_config = ability_config.get_user_config()

        assert user_config == {
            "location": {
                "storage_format": "installed_package",
                "storage_route": "route",
            },
            "packages_required": ["package1", "package2"],
            "language_model_required": {
                "model_name": "model_name",
                "provider_name": "openai",
                "temperature": 0.8,
            },
            "memory_provider_required": True,
            "workspace_required": True,
        }

    @staticmethod
    def test_validations() -> None:
        """
        Tests the validation of each individual field by feeding 'junk_value' as
        the value.
        """
        junk_value = None
        language_model_config = LanguageModelConfiguration(
            model_name="model_name",
            provider_name=ModelProviderName.OPENAI,
            temperature=0.8,
        )
        with pytest.raises(pydantic.error_wrappers.ValidationError):
            AbilityConfiguration(
                location=junk_value,
                packages_required=["package1", "package2"],
                language_model_required=language_model_config,
                memory_provider_required=True,
                workspace_required=True,
            )
        with pytest.raises(pydantic.error_wrappers.ValidationError):
            AbilityConfiguration(
                location=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="route",
                ),
                packages_required=junk_value,
                language_model_required=language_model_config,
                memory_provider_required=True,
                workspace_required=True,
            )
        with pytest.raises(pydantic.error_wrappers.ValidationError):
            AbilityConfiguration(
                location=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="route",
                ),
                packages_required=["package1", "package2"],
                language_model_required=junk_value,
                memory_provider_required=True,
                workspace_required=True,
            )
        with pytest.raises(pydantic.error_wrappers.ValidationError):
            AbilityConfiguration(
                location=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="route",
                ),
                packages_required=["package1", "package2"],
                language_model_required=language_model_config,
                memory_provider_required=junk_value,
                workspace_required=True,
            )
        with pytest.raises(pydantic.error_wrappers.ValidationError):
            AbilityConfiguration(
                location=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="route",
                ),
                packages_required=["package1", "package2"],
                language_model_required=language_model_config,
                memory_provider_required=True,
                workspace_required=junk_value,
            )


if __name__ == "__main__":
    pytest.main()
    # pass
