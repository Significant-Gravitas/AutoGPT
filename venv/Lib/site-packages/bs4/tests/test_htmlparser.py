"""Tests to ensure that the html.parser tree builder generates good
trees."""

from pdb import set_trace
import pickle
import pytest
import warnings
from bs4.builder import (
    HTMLParserTreeBuilder,
    ParserRejectedMarkup,
    XMLParsedAsHTMLWarning,
)
from bs4.builder._htmlparser import BeautifulSoupHTMLParser
from . import SoupTest, HTMLTreeBuilderSmokeTest

class TestHTMLParserTreeBuilder(SoupTest, HTMLTreeBuilderSmokeTest):

    default_builder = HTMLParserTreeBuilder

    def test_rejected_input(self):
        # Python's html.parser will occasionally reject markup,
        # especially when there is a problem with the initial DOCTYPE
        # declaration. Different versions of Python sound the alarm in
        # different ways, but Beautiful Soup consistently raises
        # errors as ParserRejectedMarkup exceptions.
        bad_markup = [
            # https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=28873
            # https://github.com/guidovranken/python-library-fuzzers/blob/master/corp-html/519e5b4269a01185a0d5e76295251921da2f0700
            # https://github.com/python/cpython/issues/81928
            b'\n<![\xff\xfe\xfe\xcd\x00',

            #https://github.com/guidovranken/python-library-fuzzers/blob/master/corp-html/de32aa55785be29bbc72a1a8e06b00611fb3d9f8
            # https://github.com/python/cpython/issues/78661
            #
            b'<![n\x00',
            b"<![UNKNOWN[]]>",
        ]
        for markup in bad_markup:
            with pytest.raises(ParserRejectedMarkup):
                soup = self.soup(markup)
    
    def test_namespaced_system_doctype(self):
        # html.parser can't handle namespaced doctypes, so skip this one.
        pass

    def test_namespaced_public_doctype(self):
        # html.parser can't handle namespaced doctypes, so skip this one.
        pass

    def test_builder_is_pickled(self):
        """Unlike most tree builders, HTMLParserTreeBuilder and will
        be restored after pickling.
        """
        tree = self.soup("<a><b>foo</a>")
        dumped = pickle.dumps(tree, 2)
        loaded = pickle.loads(dumped)
        assert isinstance(loaded.builder, type(tree.builder))

    def test_redundant_empty_element_closing_tags(self):
        self.assert_soup('<br></br><br></br><br></br>', "<br/><br/><br/>")
        self.assert_soup('</br></br></br>', "")

    def test_empty_element(self):
        # This verifies that any buffered data present when the parser
        # finishes working is handled.
        self.assert_soup("foo &# bar", "foo &amp;# bar")

    def test_tracking_line_numbers(self):
        # The html.parser TreeBuilder keeps track of line number and
        # position of each element.
        markup = "\n   <p>\n\n<sourceline>\n<b>text</b></sourceline><sourcepos></p>"
        soup = self.soup(markup)
        assert 2 == soup.p.sourceline
        assert 3 == soup.p.sourcepos
        assert "sourceline" == soup.p.find('sourceline').name

        # You can deactivate this behavior.
        soup = self.soup(markup, store_line_numbers=False)
        assert "sourceline" == soup.p.sourceline.name
        assert "sourcepos" == soup.p.sourcepos.name

    def test_on_duplicate_attribute(self):
        # The html.parser tree builder has a variety of ways of
        # handling a tag that contains the same attribute multiple times.

        markup = '<a class="cls" href="url1" href="url2" href="url3" id="id">'

        # If you don't provide any particular value for
        # on_duplicate_attribute, later values replace earlier values.
        soup = self.soup(markup)
        assert "url3" == soup.a['href']
        assert ["cls"] == soup.a['class']
        assert "id" == soup.a['id']
        
        # You can also get this behavior explicitly.
        def assert_attribute(on_duplicate_attribute, expected):
            soup = self.soup(
                markup, on_duplicate_attribute=on_duplicate_attribute
            )
            assert expected == soup.a['href']

            # Verify that non-duplicate attributes are treated normally.
            assert ["cls"] == soup.a['class']
            assert "id" == soup.a['id']
        assert_attribute(None, "url3")
        assert_attribute(BeautifulSoupHTMLParser.REPLACE, "url3")

        # You can ignore subsequent values in favor of the first.
        assert_attribute(BeautifulSoupHTMLParser.IGNORE, "url1")

        # And you can pass in a callable that does whatever you want.
        def accumulate(attrs, key, value):
            if not isinstance(attrs[key], list):
                attrs[key] = [attrs[key]]
            attrs[key].append(value)
        assert_attribute(accumulate, ["url1", "url2", "url3"])            

    def test_html5_attributes(self):
        # The html.parser TreeBuilder can convert any entity named in
        # the HTML5 spec to a sequence of Unicode characters, and
        # convert those Unicode characters to a (potentially
        # different) named entity on the way out.
        for input_element, output_unicode, output_element in (
                ("&RightArrowLeftArrow;", '\u21c4', b'&rlarr;'),
                ('&models;', '\u22a7', b'&models;'),
                ('&Nfr;', '\U0001d511', b'&Nfr;'),
                ('&ngeqq;', '\u2267\u0338', b'&ngeqq;'),
                ('&not;', '\xac', b'&not;'),
                ('&Not;', '\u2aec', b'&Not;'),
                ('&quot;', '"', b'"'),
                ('&there4;', '\u2234', b'&there4;'),
                ('&Therefore;', '\u2234', b'&there4;'),
                ('&therefore;', '\u2234', b'&there4;'),
                ("&fjlig;", 'fj', b'fj'),                
                ("&sqcup;", '\u2294', b'&sqcup;'),
                ("&sqcups;", '\u2294\ufe00', b'&sqcups;'),
                ("&apos;", "'", b"'"),
                ("&verbar;", "|", b"|"),
        ):
            markup = '<div>%s</div>' % input_element
            div = self.soup(markup).div
            without_element = div.encode()
            expect = b"<div>%s</div>" % output_unicode.encode("utf8")
            assert without_element == expect

            with_element = div.encode(formatter="html")
            expect = b"<div>%s</div>" % output_element
            assert with_element == expect
