"""Tests to ensure that the lxml tree builder generates good trees."""

import pickle
import pytest
import re
import warnings
from . import LXML_PRESENT, LXML_VERSION

if LXML_PRESENT:
    from bs4.builder import LXMLTreeBuilder, LXMLTreeBuilderForXML

from bs4 import (
    BeautifulSoup,
    BeautifulStoneSoup,
    )
from bs4.element import Comment, Doctype, SoupStrainer
from . import (
    HTMLTreeBuilderSmokeTest,
    XMLTreeBuilderSmokeTest,
    SOUP_SIEVE_PRESENT,
    SoupTest,
)

@pytest.mark.skipif(
    not LXML_PRESENT,
    reason="lxml seems not to be present, not testing its tree builder."
)
class TestLXMLTreeBuilder(SoupTest, HTMLTreeBuilderSmokeTest):
    """See ``HTMLTreeBuilderSmokeTest``."""

    @property
    def default_builder(self):
        return LXMLTreeBuilder

    def test_out_of_range_entity(self):
        self.assert_soup(
            "<p>foo&#10000000000000;bar</p>", "<p>foobar</p>")
        self.assert_soup(
            "<p>foo&#x10000000000000;bar</p>", "<p>foobar</p>")
        self.assert_soup(
            "<p>foo&#1000000000;bar</p>", "<p>foobar</p>")
        
    def test_entities_in_foreign_document_encoding(self):
        # We can't implement this case correctly because by the time we
        # hear about markup like "&#147;", it's been (incorrectly) converted into
        # a string like u'\x93'
        pass
        
    # In lxml < 2.3.5, an empty doctype causes a segfault. Skip this
    # test if an old version of lxml is installed.

    @pytest.mark.skipif(
        not LXML_PRESENT or LXML_VERSION < (2,3,5,0),
        reason="Skipping doctype test for old version of lxml to avoid segfault."
    )
    def test_empty_doctype(self):
        soup = self.soup("<!DOCTYPE>")
        doctype = soup.contents[0]
        assert "" == doctype.strip()

    def test_beautifulstonesoup_is_xml_parser(self):
        # Make sure that the deprecated BSS class uses an xml builder
        # if one is installed.
        with warnings.catch_warnings(record=True) as w:
            soup = BeautifulStoneSoup("<b />")
        assert "<b/>" == str(soup.b)
        [warning] = w
        assert warning.filename == __file__
        assert "BeautifulStoneSoup class is deprecated" in str(warning.message)

    def test_tracking_line_numbers(self):
        # The lxml TreeBuilder cannot keep track of line numbers from
        # the original markup. Even if you ask for line numbers, we
        # don't have 'em.
        #
        # This means that if you have a tag like <sourceline> or
        # <sourcepos>, attribute access will find it rather than
        # giving you a numeric answer.
        soup = self.soup(
            "\n   <p>\n\n<sourceline>\n<b>text</b></sourceline><sourcepos></p>",
            store_line_numbers=True
        )
        assert "sourceline" == soup.p.sourceline.name
        assert "sourcepos" == soup.p.sourcepos.name
        
@pytest.mark.skipif(
    not LXML_PRESENT,
    reason="lxml seems not to be present, not testing its XML tree builder."
)
class TestLXMLXMLTreeBuilder(SoupTest, XMLTreeBuilderSmokeTest):
    """See ``HTMLTreeBuilderSmokeTest``."""

    @property
    def default_builder(self):
        return LXMLTreeBuilderForXML

    def test_namespace_indexing(self):
        soup = self.soup(
            '<?xml version="1.1"?>\n'
            '<root>'
            '<tag xmlns="http://unprefixed-namespace.com">content</tag>'
            '<prefix:tag2 xmlns:prefix="http://prefixed-namespace.com">content</prefix:tag2>'
            '<prefix2:tag3 xmlns:prefix2="http://another-namespace.com">'
            '<subtag xmlns="http://another-unprefixed-namespace.com">'
            '<subsubtag xmlns="http://yet-another-unprefixed-namespace.com">'
            '</prefix2:tag3>'
            '</root>'
        )

        # The BeautifulSoup object includes every namespace prefix
        # defined in the entire document. This is the default set of
        # namespaces used by soupsieve.
        #
        # Un-prefixed namespaces are not included, and if a given
        # prefix is defined twice, only the first prefix encountered
        # in the document shows up here.
        assert soup._namespaces == {
            'xml': 'http://www.w3.org/XML/1998/namespace',
            'prefix': 'http://prefixed-namespace.com',
            'prefix2': 'http://another-namespace.com'
        }

        # A Tag object includes only the namespace prefixes
        # that were in scope when it was parsed.

        # We do not track un-prefixed namespaces as we can only hold
        # one (the first one), and it will be recognized as the
        # default namespace by soupsieve, even when operating from a
        # tag with a different un-prefixed namespace.
        assert soup.tag._namespaces == {
            'xml': 'http://www.w3.org/XML/1998/namespace',
        }

        assert soup.tag2._namespaces == {
            'prefix': 'http://prefixed-namespace.com',
            'xml': 'http://www.w3.org/XML/1998/namespace',
        }

        assert soup.subtag._namespaces == {
            'prefix2': 'http://another-namespace.com',
            'xml': 'http://www.w3.org/XML/1998/namespace',
        }

        assert soup.subsubtag._namespaces == {
            'prefix2': 'http://another-namespace.com',
            'xml': 'http://www.w3.org/XML/1998/namespace',
        }


    @pytest.mark.skipif(
        not SOUP_SIEVE_PRESENT, reason="Soup Sieve not installed"
    )
    def test_namespace_interaction_with_select_and_find(self):
        # Demonstrate how namespaces interact with select* and
        # find* methods.
        
        soup = self.soup(
            '<?xml version="1.1"?>\n'
            '<root>'
            '<tag xmlns="http://unprefixed-namespace.com">content</tag>'
            '<prefix:tag2 xmlns:prefix="http://prefixed-namespace.com">content</tag>'
            '<subtag xmlns:prefix="http://another-namespace-same-prefix.com">'
             '<prefix:tag3>'
            '</subtag>'
            '</root>'
        )

        # soupselect uses namespace URIs.
        assert soup.select_one('tag').name == 'tag'
        assert soup.select_one('prefix|tag2').name == 'tag2'

        # If a prefix is declared more than once, only the first usage
        # is registered with the BeautifulSoup object.
        assert soup.select_one('prefix|tag3') is None

        # But you can always explicitly specify a namespace dictionary.
        assert soup.select_one(
            'prefix|tag3', namespaces=soup.subtag._namespaces
        ).name == 'tag3'

        # And a Tag (as opposed to the BeautifulSoup object) will
        # have a set of default namespaces scoped to that Tag.
        assert soup.subtag.select_one('prefix|tag3').name=='tag3'

        # the find() methods aren't fully namespace-aware; they just
        # look at prefixes.
        assert soup.find('tag').name == 'tag'
        assert soup.find('prefix:tag2').name == 'tag2'
        assert soup.find('prefix:tag3').name == 'tag3'
        assert soup.subtag.find('prefix:tag3').name == 'tag3'

    def test_pickle_removes_builder(self):
        # The lxml TreeBuilder is not picklable, so it won't be
        # preserved in a pickle/unpickle operation.

        soup = self.soup("<a>some markup</a>")
        assert isinstance(soup.builder, self.default_builder)
        pickled = pickle.dumps(soup)
        unpickled = pickle.loads(pickled)
        assert "some markup" == unpickled.a.string
        assert unpickled.builder is None
