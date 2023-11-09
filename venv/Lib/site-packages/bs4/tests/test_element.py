"""Tests of classes in element.py.

The really big classes -- Tag, PageElement, and NavigableString --
are tested in separate files.
"""

from bs4.element import (
    CharsetMetaAttributeValue,
    ContentMetaAttributeValue,
    NamespacedAttribute,
)
from . import SoupTest


class TestNamedspacedAttribute(object):

    def test_name_may_be_none_or_missing(self):
        a = NamespacedAttribute("xmlns", None)
        assert a == "xmlns"

        a = NamespacedAttribute("xmlns", "")
        assert a == "xmlns"

        a = NamespacedAttribute("xmlns")
        assert a == "xmlns"
        
    def test_namespace_may_be_none_or_missing(self):
        a = NamespacedAttribute(None, "tag")
        assert a == "tag"
        
        a = NamespacedAttribute("", "tag")
        assert a == "tag"
        
    def test_attribute_is_equivalent_to_colon_separated_string(self):
        a = NamespacedAttribute("a", "b")
        assert "a:b" == a

    def test_attributes_are_equivalent_if_prefix_and_name_identical(self):
        a = NamespacedAttribute("a", "b", "c")
        b = NamespacedAttribute("a", "b", "c")
        assert a == b

        # The actual namespace is not considered.
        c = NamespacedAttribute("a", "b", None)
        assert a == c

        # But name and prefix are important.
        d = NamespacedAttribute("a", "z", "c")
        assert a != d

        e = NamespacedAttribute("z", "b", "c")
        assert a != e


class TestAttributeValueWithCharsetSubstitution(object):
    """Certain attributes are designed to have the charset of the
    final document substituted into their value.
    """
    
    def test_content_meta_attribute_value(self):
        # The value of a CharsetMetaAttributeValue is whatever
        # encoding the string is in.
        value = CharsetMetaAttributeValue("euc-jp")
        assert "euc-jp" == value
        assert "euc-jp" == value.original_value
        assert "utf8" == value.encode("utf8")
        assert "ascii" == value.encode("ascii")

    def test_content_meta_attribute_value(self):
        value = ContentMetaAttributeValue("text/html; charset=euc-jp")
        assert "text/html; charset=euc-jp" == value
        assert "text/html; charset=euc-jp" == value.original_value
        assert "text/html; charset=utf8" == value.encode("utf8")
        assert "text/html; charset=ascii" == value.encode("ascii")
