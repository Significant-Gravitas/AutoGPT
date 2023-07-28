"""
This file provides all the necessary pytests for autogpt/core/ability/simple.py.
"""
from autogpt.core.ability.simple import (
    AbilityConfiguration,
    AbilityRegistryConfiguration,
    AbilityRegistrySettings,
)
from autogpt.core.plugin.base import PluginLocation, PluginStorageFormat


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
