"""
This file provides necessary tests for the classes and codes defined in
autogpt/core/ability/schema.py
"""
import enum

import pytest

from autogpt.core.ability.schema import AbilityResult, ContentType, Knowledge


class TestContentType:
    """
    Provides necessary tests for the ContentType class.
    """

    @staticmethod
    def test_values() -> None:
        # Test if the ContentType has the correct values
        assert ContentType.TEXT.value == "text"
        assert ContentType.CODE.value == "code"

    @staticmethod
    def test_is_enum_instance() -> None:
        # Test if the ContentType members are instances of the ContentType enum class
        assert isinstance(ContentType.TEXT, ContentType)
        assert isinstance(ContentType.CODE, ContentType)

    @staticmethod
    def test_is_enum() -> None:
        # Test if the ContentType class is indeed an Enum
        assert issubclass(ContentType, enum.Enum)

    @staticmethod
    def test_enum_comparison() -> None:
        # Test if ContentType members can be compared with '==' and 'is'
        assert ContentType.TEXT == ContentType.TEXT
        assert not ContentType.TEXT == ContentType.CODE
        assert ContentType.TEXT is ContentType.TEXT
        assert not ContentType.TEXT is ContentType.CODE

    @staticmethod
    def test_member_counts() -> None:
        # Test there are exactly two TEXT and CODE fields in ContentType
        members = ContentType.__members__.values()
        assert len(members) == 2
        for field_name in members.mapping.keys():
            assert field_name in ("TEXT", "CODE")


class TestKnowledgeModel:
    """
    Provides necessary tests for the Knowledge class.
    """

    @staticmethod
    def test_knowledge_model_attributes() -> None:
        # Test if the Knowledge class has the expected attributes
        knowledge = Knowledge(
            content="Sample content", content_type=ContentType.TEXT, content_metadata={}
        )
        assert knowledge.content == "Sample content"
        assert knowledge.content_type == ContentType.TEXT
        assert knowledge.content_metadata == {}

    @staticmethod
    def test_knowledge_model_types() -> None:
        # Test if the attribute types of Knowledge class match the type hints
        knowledge = Knowledge(
            content="Sample content",
            content_type=ContentType.CODE,
            content_metadata={"key": "value"},
        )

        assert isinstance(knowledge.content, str)
        assert isinstance(knowledge.content_type, ContentType)
        assert isinstance(knowledge.content_metadata, dict)

    @staticmethod
    def test_knowledge_model_invalid_content_type() -> None:
        # Test if the Knowledge model raises a ValueError for an invalid content_type
        with pytest.raises(ValueError):
            Knowledge(
                content="Invalid content",
                content_type="invalid_type",
                content_metadata={},
            )

    @staticmethod
    def test_member_counts() -> None:
        # Test there are exactly three memebers in Knowledge
        members = Knowledge.__fields__
        assert len(members) == 3
        # check they have the right name too
        for field_name in members.keys():
            assert field_name in ("content", "content_type", "content_metadata")


class TestAbilityResultModel:
    """
    Provides necessary tests for the AbilityResult class.
    """

    @staticmethod
    def test_ability_result_attributes() -> None:
        # Test if the AbilityResult class has the expected attributes
        result = AbilityResult(
            ability_name="TestAbility",
            ability_args={"arg1": "value1", "arg2": "value2"},
            success=True,
            message="Ability executed successfully.",
            new_knowledge=Knowledge(
                content="Sample content",
                content_type=ContentType.TEXT,
                content_metadata={},
            ),
        )

        assert result.ability_name == "TestAbility"
        assert result.ability_args == {"arg1": "value1", "arg2": "value2"}
        assert result.success is True
        assert result.message == "Ability executed successfully."
        assert isinstance(result.new_knowledge, (Knowledge, type(None)))

    @staticmethod
    def test_ability_result_summary() -> None:
        # Test the summary method of AbilityResult
        result = AbilityResult(
            ability_name="TestAbility",
            ability_args={"arg1": "value1", "arg2": "value2"},
            success=True,
            message="Ability executed successfully.",
            new_knowledge=Knowledge(
                content="Sample content",
                content_type=ContentType.TEXT,
                content_metadata={},
            ),
        )

        expected_summary = (
            "TestAbility(arg1=value1, arg2=value2): Ability executed successfully."
        )
        assert result.summary() == expected_summary

    @staticmethod
    def test_ability_result_default_new_knowledge() -> None:
        # Test if new_knowledge attribute is None by default
        result = AbilityResult(
            ability_name="TestAbility",
            ability_args={"arg1": "value1", "arg2": "value2"},
            success=True,
            message="Ability executed successfully.",
        )

        assert result.new_knowledge is None
