"""Tests of the builder registry."""

import pytest
import warnings

from bs4 import BeautifulSoup
from bs4.builder import (
    builder_registry as registry,
    HTMLParserTreeBuilder,
    TreeBuilderRegistry,
)

from . import (
    HTML5LIB_PRESENT,
    LXML_PRESENT,
)

if HTML5LIB_PRESENT:
    from bs4.builder import HTML5TreeBuilder

if LXML_PRESENT:
    from bs4.builder import (
        LXMLTreeBuilderForXML,
        LXMLTreeBuilder,
        )


# TODO: Split out the lxml and html5lib tests into their own classes
# and gate with pytest.mark.skipIf.
class TestBuiltInRegistry(object):
    """Test the built-in registry with the default builders registered."""

    def test_combination(self):
        assert registry.lookup('strict', 'html') == HTMLParserTreeBuilder
        if LXML_PRESENT:
            assert registry.lookup('fast', 'html') == LXMLTreeBuilder
            assert registry.lookup('permissive', 'xml') == LXMLTreeBuilderForXML
        if HTML5LIB_PRESENT:
            assert registry.lookup('html5lib', 'html') == HTML5TreeBuilder

    def test_lookup_by_markup_type(self):
        if LXML_PRESENT:
            assert registry.lookup('html') == LXMLTreeBuilder
            assert registry.lookup('xml') == LXMLTreeBuilderForXML
        else:
            assert registry.lookup('xml') == None
            if HTML5LIB_PRESENT:
                assert registry.lookup('html') == HTML5TreeBuilder
            else:
                assert registry.lookup('html') == HTMLParserTreeBuilder

    def test_named_library(self):
        if LXML_PRESENT:
            assert registry.lookup('lxml', 'xml') == LXMLTreeBuilderForXML
            assert registry.lookup('lxml', 'html') == LXMLTreeBuilder
        if HTML5LIB_PRESENT:
            assert registry.lookup('html5lib') == HTML5TreeBuilder

        assert registry.lookup('html.parser') == HTMLParserTreeBuilder

    def test_beautifulsoup_constructor_does_lookup(self):

        with warnings.catch_warnings(record=True) as w:
            # This will create a warning about not explicitly
            # specifying a parser, but we'll ignore it.

            # You can pass in a string.
            BeautifulSoup("", features="html")
            # Or a list of strings.
            BeautifulSoup("", features=["html", "fast"])
            pass
            
        # You'll get an exception if BS can't find an appropriate
        # builder.
        with pytest.raises(ValueError):
            BeautifulSoup("", features="no-such-feature")

class TestRegistry(object):
    """Test the TreeBuilderRegistry class in general."""

    def setup_method(self):
        self.registry = TreeBuilderRegistry()

    def builder_for_features(self, *feature_list):
        cls = type('Builder_' + '_'.join(feature_list),
                   (object,), {'features' : feature_list})

        self.registry.register(cls)
        return cls

    def test_register_with_no_features(self):
        builder = self.builder_for_features()

        # Since the builder advertises no features, you can't find it
        # by looking up features.
        assert self.registry.lookup('foo') is None

        # But you can find it by doing a lookup with no features, if
        # this happens to be the only registered builder.
        assert self.registry.lookup() == builder

    def test_register_with_features_makes_lookup_succeed(self):
        builder = self.builder_for_features('foo', 'bar')
        assert self.registry.lookup('foo') is builder
        assert self.registry.lookup('bar') is builder

    def test_lookup_fails_when_no_builder_implements_feature(self):
        builder = self.builder_for_features('foo', 'bar')
        assert self.registry.lookup('baz') is None

    def test_lookup_gets_most_recent_registration_when_no_feature_specified(self):
        builder1 = self.builder_for_features('foo')
        builder2 = self.builder_for_features('bar')
        assert self.registry.lookup() == builder2

    def test_lookup_fails_when_no_tree_builders_registered(self):
        assert self.registry.lookup() is None

    def test_lookup_gets_most_recent_builder_supporting_all_features(self):
        has_one = self.builder_for_features('foo')
        has_the_other = self.builder_for_features('bar')
        has_both_early = self.builder_for_features('foo', 'bar', 'baz')
        has_both_late = self.builder_for_features('foo', 'bar', 'quux')
        lacks_one = self.builder_for_features('bar')
        has_the_other = self.builder_for_features('foo')

        # There are two builders featuring 'foo' and 'bar', but
        # the one that also features 'quux' was registered later.
        assert self.registry.lookup('foo', 'bar') == has_both_late

        # There is only one builder featuring 'foo', 'bar', and 'baz'.
        assert self.registry.lookup('foo', 'bar', 'baz') == has_both_early

    def test_lookup_fails_when_cannot_reconcile_requested_features(self):
        builder1 = self.builder_for_features('foo', 'bar')
        builder2 = self.builder_for_features('foo', 'baz')
        assert self.registry.lookup('bar', 'baz') is None
