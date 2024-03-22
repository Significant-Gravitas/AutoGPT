"""
This file provides necessary pytests for the autogpt.core.agent.base module.
"""
import pytest

from autogpt.core.agent.base import Agent


class TestAgent:
    @staticmethod
    def test_abstract_methods_names() -> None:
        """
        Tests that abstract methods exist.
        """
        methods_names = frozenset(
            {
                "determine_next_ability",
                "from_workspace",
                "__init__",
                "__repr__",
            }
        )
        assert methods_names == Agent.__abstractmethods__

    @staticmethod
    def test_class_abstraction() -> None:
        """
        Tests that the Agent base class is abstract.
        """
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            Agent()
