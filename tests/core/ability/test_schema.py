"""
This file provides necessary tests for the classes and codes defined in
autogpt/core/ability/schema.py
"""
import enum

from autogpt.core.ability.schema import ContentType


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
