import abc
from pprint import pformat
from typing import Any

import pytest
from inflection import underscore

from autogpt.core.ability import Ability, AbilityResult
from autogpt.core.ability.schema import Knowledge


class MockDerivedAbility(Ability):
    """
    This is a mock class simulating implementation of Ability class. It is
    used inside the various test methods of TestAbility class in the following.
    """

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


class TestAbility:
    """
    Provides various tests for Ability class
    """

    def setup_method(self) -> None:
        self.ability = MockDerivedAbility()

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


if __name__ == "__main__":
    pytest.main()
