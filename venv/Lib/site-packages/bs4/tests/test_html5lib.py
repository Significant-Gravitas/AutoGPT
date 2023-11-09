"""Tests to ensure that the html5lib tree builder generates good trees."""

import pytest
import warnings

from bs4 import BeautifulSoup
from bs4.element import SoupStrainer
from . import (
    HTML5LIB_PRESENT,
    HTML5TreeBuilderSmokeTest,
    SoupTest,
)

@pytest.mark.skipif(
    not HTML5LIB_PRESENT,
    reason="html5lib seems not to be present, not testing its tree builder."
)
class TestHTML5LibBuilder(SoupTest, HTML5TreeBuilderSmokeTest):
    """See ``HTML5TreeBuilderSmokeTest``."""

    @property
    def default_builder(self):
        from bs4.builder import HTML5TreeBuilder
        return HTML5TreeBuilder

    def test_soupstrainer(self):
        # The html5lib tree builder does not support SoupStrainers.
        strainer = SoupStrainer("b")
        markup = "<p>A <b>bold</b> statement.</p>"
        with warnings.catch_warnings(record=True) as w:
            soup = BeautifulSoup(markup, "html5lib", parse_only=strainer)
        assert soup.decode() == self.document_for(markup)

        [warning] = w
        assert warning.filename == __file__
        assert "the html5lib tree builder doesn't support parse_only" in str(warning.message)

    def test_correctly_nested_tables(self):
        """html5lib inserts <tbody> tags where other parsers don't."""
        markup = ('<table id="1">'
                  '<tr>'
                  "<td>Here's another table:"
                  '<table id="2">'
                  '<tr><td>foo</td></tr>'
                  '</table></td>')

        self.assert_soup(
            markup,
            '<table id="1"><tbody><tr><td>Here\'s another table:'
            '<table id="2"><tbody><tr><td>foo</td></tr></tbody></table>'
            '</td></tr></tbody></table>')

        self.assert_soup(
            "<table><thead><tr><td>Foo</td></tr></thead>"
            "<tbody><tr><td>Bar</td></tr></tbody>"
            "<tfoot><tr><td>Baz</td></tr></tfoot></table>")

    def test_xml_declaration_followed_by_doctype(self):
        markup = '''<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html>
  <head>
  </head>
  <body>
   <p>foo</p>
  </body>
</html>'''
        soup = self.soup(markup)
        # Verify that we can reach the <p> tag; this means the tree is connected.
        assert b"<p>foo</p>" == soup.p.encode()

    def test_reparented_markup(self):
        markup = '<p><em>foo</p>\n<p>bar<a></a></em></p>'
        soup = self.soup(markup)
        assert "<body><p><em>foo</em></p><em>\n</em><p><em>bar<a></a></em></p></body>" == soup.body.decode()
        assert 2 == len(soup.find_all('p'))


    def test_reparented_markup_ends_with_whitespace(self):
        markup = '<p><em>foo</p>\n<p>bar<a></a></em></p>\n'
        soup = self.soup(markup)
        assert "<body><p><em>foo</em></p><em>\n</em><p><em>bar<a></a></em></p>\n</body>" == soup.body.decode()
        assert 2 == len(soup.find_all('p'))

    def test_reparented_markup_containing_identical_whitespace_nodes(self):
        """Verify that we keep the two whitespace nodes in this
        document distinct when reparenting the adjacent <tbody> tags.
        """
        markup = '<table> <tbody><tbody><ims></tbody> </table>'
        soup = self.soup(markup)
        space1, space2 = soup.find_all(string=' ')
        tbody1, tbody2 = soup.find_all('tbody')
        assert space1.next_element is tbody1
        assert tbody2.next_element is space2

    def test_reparented_markup_containing_children(self):
        markup = '<div><a>aftermath<p><noscript>target</noscript>aftermath</a></p></div>'
        soup = self.soup(markup)
        noscript = soup.noscript
        assert "target" == noscript.next_element
        target = soup.find(string='target')

        # The 'aftermath' string was duplicated; we want the second one.
        final_aftermath = soup.find_all(string='aftermath')[-1]

        # The <noscript> tag was moved beneath a copy of the <a> tag,
        # but the 'target' string within is still connected to the
        # (second) 'aftermath' string.
        assert final_aftermath == target.next_element
        assert target == final_aftermath.previous_element
        
    def test_processing_instruction(self):
        """Processing instructions become comments."""
        markup = b"""<?PITarget PIContent?>"""
        soup = self.soup(markup)
        assert str(soup).startswith("<!--?PITarget PIContent?-->")

    def test_cloned_multivalue_node(self):
        markup = b"""<a class="my_class"><p></a>"""
        soup = self.soup(markup)
        a1, a2 = soup.find_all('a')
        assert a1 == a2
        assert a1 is not a2

    def test_foster_parenting(self):
        markup = b"""<table><td></tbody>A"""
        soup = self.soup(markup)
        assert "<body>A<table><tbody><tr><td></td></tr></tbody></table></body>" == soup.body.decode()

    def test_extraction(self):
        """
        Test that extraction does not destroy the tree.

        https://bugs.launchpad.net/beautifulsoup/+bug/1782928
        """

        markup = """
<html><head></head>
<style>
</style><script></script><body><p>hello</p></body></html>
"""
        soup = self.soup(markup)
        [s.extract() for s in soup('script')]
        [s.extract() for s in soup('style')]

        assert len(soup.find_all("p")) == 1

    def test_empty_comment(self):
        """
        Test that empty comment does not break structure.

        https://bugs.launchpad.net/beautifulsoup/+bug/1806598
        """

        markup = """
<html>
<body>
<form>
<!----><input type="text">
</form>
</body>
</html>
"""
        soup = self.soup(markup)
        inputs = []
        for form in soup.find_all('form'):
            inputs.extend(form.find_all('input'))
        assert len(inputs) == 1

    def test_tracking_line_numbers(self):
        # The html.parser TreeBuilder keeps track of line number and
        # position of each element.
        markup = "\n   <p>\n\n<sourceline>\n<b>text</b></sourceline><sourcepos></p>"
        soup = self.soup(markup)
        assert 2 == soup.p.sourceline
        assert 5 == soup.p.sourcepos
        assert "sourceline" == soup.p.find('sourceline').name

        # You can deactivate this behavior.
        soup = self.soup(markup, store_line_numbers=False)
        assert "sourceline" == soup.p.sourceline.name
        assert "sourcepos" == soup.p.sourcepos.name

    def test_special_string_containers(self):
        # The html5lib tree builder doesn't support this standard feature,
        # because there's no way of knowing, when a string is created,
        # where in the tree it will eventually end up.
        pass

    def test_html5_attributes(self):
        # The html5lib TreeBuilder can convert any entity named in
        # the HTML5 spec to a sequence of Unicode characters, and
        # convert those Unicode characters to a (potentially
        # different) named entity on the way out.
        #
        # This is a copy of the same test from
        # HTMLParserTreeBuilderSmokeTest.  It's not in the superclass
        # because the lxml HTML TreeBuilder _doesn't_ work this way.
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
