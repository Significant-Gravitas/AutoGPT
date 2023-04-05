"""Tests of the bs4.element.PageElement class"""
import copy
import pickle
import pytest
import sys

from bs4 import BeautifulSoup
from bs4.element import (
    Comment,
    ResultSet,
    SoupStrainer,
)
from . import (
    SoupTest,
)

class TestEncoding(SoupTest):
    """Test the ability to encode objects into strings."""

    def test_unicode_string_can_be_encoded(self):
        html = "<b>\N{SNOWMAN}</b>"
        soup = self.soup(html)
        assert soup.b.string.encode("utf-8") == "\N{SNOWMAN}".encode("utf-8")

    def test_tag_containing_unicode_string_can_be_encoded(self):
        html = "<b>\N{SNOWMAN}</b>"
        soup = self.soup(html)
        assert soup.b.encode("utf-8") == html.encode("utf-8")

    def test_encoding_substitutes_unrecognized_characters_by_default(self):
        html = "<b>\N{SNOWMAN}</b>"
        soup = self.soup(html)
        assert soup.b.encode("ascii") == b"<b>&#9731;</b>"

    def test_encoding_can_be_made_strict(self):
        html = "<b>\N{SNOWMAN}</b>"
        soup = self.soup(html)
        with pytest.raises(UnicodeEncodeError):
            soup.encode("ascii", errors="strict")

    def test_decode_contents(self):
        html = "<b>\N{SNOWMAN}</b>"
        soup = self.soup(html)
        assert "\N{SNOWMAN}" == soup.b.decode_contents()

    def test_encode_contents(self):
        html = "<b>\N{SNOWMAN}</b>"
        soup = self.soup(html)
        assert "\N{SNOWMAN}".encode("utf8") == soup.b.encode_contents(
            encoding="utf8"
        )

    def test_encode_deeply_nested_document(self):
        # This test verifies that encoding a string doesn't involve
        # any recursive function calls. If it did, this test would
        # overflow the Python interpreter stack.
        limit = sys.getrecursionlimit() + 1
        markup = "<span>" * limit
        soup = self.soup(markup)
        encoded = soup.encode()
        assert limit == encoded.count(b"<span>")

    def test_deprecated_renderContents(self):
        html = "<b>\N{SNOWMAN}</b>"
        soup = self.soup(html)
        assert "\N{SNOWMAN}".encode("utf8") == soup.b.renderContents()

    def test_repr(self):
        html = "<b>\N{SNOWMAN}</b>"
        soup = self.soup(html)
        assert html == repr(soup)

        
class TestFormatters(SoupTest):
    """Test the formatting feature, used by methods like decode() and
    prettify(), and the formatters themselves.
    """
    
    def test_default_formatter_is_minimal(self):
        markup = "<b>&lt;&lt;Sacr\N{LATIN SMALL LETTER E WITH ACUTE} bleu!&gt;&gt;</b>"
        soup = self.soup(markup)
        decoded = soup.decode(formatter="minimal")
        # The < is converted back into &lt; but the e-with-acute is left alone.
        assert decoded == self.document_for(
                "<b>&lt;&lt;Sacr\N{LATIN SMALL LETTER E WITH ACUTE} bleu!&gt;&gt;</b>"
        )

    def test_formatter_html(self):
        markup = "<br><b>&lt;&lt;Sacr\N{LATIN SMALL LETTER E WITH ACUTE} bleu!&gt;&gt;</b>"
        soup = self.soup(markup)
        decoded = soup.decode(formatter="html")
        assert decoded == self.document_for(
            "<br/><b>&lt;&lt;Sacr&eacute; bleu!&gt;&gt;</b>"
        )

    def test_formatter_html5(self):
        markup = "<br><b>&lt;&lt;Sacr\N{LATIN SMALL LETTER E WITH ACUTE} bleu!&gt;&gt;</b>"
        soup = self.soup(markup)
        decoded = soup.decode(formatter="html5")
        assert decoded == self.document_for(
            "<br><b>&lt;&lt;Sacr&eacute; bleu!&gt;&gt;</b>"
        )
        
    def test_formatter_minimal(self):
        markup = "<b>&lt;&lt;Sacr\N{LATIN SMALL LETTER E WITH ACUTE} bleu!&gt;&gt;</b>"
        soup = self.soup(markup)
        decoded = soup.decode(formatter="minimal")
        # The < is converted back into &lt; but the e-with-acute is left alone.
        assert decoded == self.document_for(
                "<b>&lt;&lt;Sacr\N{LATIN SMALL LETTER E WITH ACUTE} bleu!&gt;&gt;</b>"
        )

    def test_formatter_null(self):
        markup = "<b>&lt;&lt;Sacr\N{LATIN SMALL LETTER E WITH ACUTE} bleu!&gt;&gt;</b>"
        soup = self.soup(markup)
        decoded = soup.decode(formatter=None)
        # Neither the angle brackets nor the e-with-acute are converted.
        # This is not valid HTML, but it's what the user wanted.
        assert decoded == self.document_for(
            "<b><<Sacr\N{LATIN SMALL LETTER E WITH ACUTE} bleu!>></b>"
        )

    def test_formatter_custom(self):
        markup = "<b>&lt;foo&gt;</b><b>bar</b><br/>"
        soup = self.soup(markup)
        decoded = soup.decode(formatter = lambda x: x.upper())
        # Instead of normal entity conversion code, the custom
        # callable is called on every string.
        assert decoded == self.document_for("<b><FOO></b><b>BAR</b><br/>")

    def test_formatter_is_run_on_attribute_values(self):
        markup = '<a href="http://a.com?a=b&c=é">e</a>'
        soup = self.soup(markup)
        a = soup.a

        expect_minimal = '<a href="http://a.com?a=b&amp;c=é">e</a>'

        assert expect_minimal == a.decode()
        assert expect_minimal == a.decode(formatter="minimal")

        expect_html = '<a href="http://a.com?a=b&amp;c=&eacute;">e</a>'
        assert expect_html == a.decode(formatter="html")

        assert markup == a.decode(formatter=None)
        expect_upper = '<a href="HTTP://A.COM?A=B&C=É">E</a>'
        assert expect_upper == a.decode(formatter=lambda x: x.upper())

    def test_formatter_skips_script_tag_for_html_documents(self):
        doc = """
  <script type="text/javascript">
   console.log("< < hey > > ");
  </script>
"""
        encoded = BeautifulSoup(doc, 'html.parser').encode()
        assert b"< < hey > >" in encoded

    def test_formatter_skips_style_tag_for_html_documents(self):
        doc = """
  <style type="text/css">
   console.log("< < hey > > ");
  </style>
"""
        encoded = BeautifulSoup(doc, 'html.parser').encode()
        assert b"< < hey > >" in encoded

    def test_prettify_leaves_preformatted_text_alone(self):
        soup = self.soup("<div>  foo  <pre>  \tbar\n  \n  </pre>  baz  <textarea> eee\nfff\t</textarea></div>")
        # Everything outside the <pre> tag is reformatted, but everything
        # inside is left alone.
        assert '<div>\n foo\n <pre>  \tbar\n  \n  </pre>\n baz\n <textarea> eee\nfff\t</textarea>\n</div>\n' == soup.div.prettify()

    def test_prettify_handles_nested_string_literal_tags(self):
        # Most of this markup is inside a <pre> tag, so prettify()
        # only does three things to it:
        # 1. Add a newline and a space between the <div> and the <pre>
        # 2. Add a newline after the </pre>
        # 3. Add a newline at the end.
        #
        # The contents of the <pre> tag are left completely alone.  In
        # particular, we don't start adding whitespace again once we
        # encounter the first </pre> tag, because we know it's not
        # the one that put us into string literal mode.
        markup = """<div><pre><code>some
<script><pre>code</pre></script> for you 
</code></pre></div>"""

        expect = """<div>
 <pre><code>some
<script><pre>code</pre></script> for you 
</code></pre>
</div>
"""
        soup = self.soup(markup)
        assert expect == soup.div.prettify()

    def test_prettify_accepts_formatter_function(self):
        soup = BeautifulSoup("<html><body>foo</body></html>", 'html.parser')
        pretty = soup.prettify(formatter = lambda x: x.upper())
        assert "FOO" in pretty

    def test_prettify_outputs_unicode_by_default(self):
        soup = self.soup("<a></a>")
        assert str == type(soup.prettify())

    def test_prettify_can_encode_data(self):
        soup = self.soup("<a></a>")
        assert bytes == type(soup.prettify("utf-8"))

    def test_html_entity_substitution_off_by_default(self):
        markup = "<b>Sacr\N{LATIN SMALL LETTER E WITH ACUTE} bleu!</b>"
        soup = self.soup(markup)
        encoded = soup.b.encode("utf-8")
        assert encoded == markup.encode('utf-8')

    def test_encoding_substitution(self):
        # Here's the <meta> tag saying that a document is
        # encoded in Shift-JIS.
        meta_tag = ('<meta content="text/html; charset=x-sjis" '
                    'http-equiv="Content-type"/>')
        soup = self.soup(meta_tag)

        # Parse the document, and the charset apprears unchanged.
        assert soup.meta['content'] == 'text/html; charset=x-sjis'

        # Encode the document into some encoding, and the encoding is
        # substituted into the meta tag.
        utf_8 = soup.encode("utf-8")
        assert b"charset=utf-8" in utf_8

        euc_jp = soup.encode("euc_jp")
        assert b"charset=euc_jp" in euc_jp

        shift_jis = soup.encode("shift-jis")
        assert b"charset=shift-jis" in shift_jis

        utf_16_u = soup.encode("utf-16").decode("utf-16")
        assert "charset=utf-16" in utf_16_u

    def test_encoding_substitution_doesnt_happen_if_tag_is_strained(self):
        markup = ('<head><meta content="text/html; charset=x-sjis" '
                    'http-equiv="Content-type"/></head><pre>foo</pre>')

        # Beautiful Soup used to try to rewrite the meta tag even if the
        # meta tag got filtered out by the strainer. This test makes
        # sure that doesn't happen.
        strainer = SoupStrainer('pre')
        soup = self.soup(markup, parse_only=strainer)
        assert soup.contents[0].name == 'pre'


class TestPersistence(SoupTest):
    "Testing features like pickle and deepcopy."

    def setup_method(self):
        self.page = """<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.0 Transitional//EN"
"http://www.w3.org/TR/REC-html40/transitional.dtd">
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
<title>Beautiful Soup: We called him Tortoise because he taught us.</title>
<link rev="made" href="mailto:leonardr@segfault.org">
<meta name="Description" content="Beautiful Soup: an HTML parser optimized for screen-scraping.">
<meta name="generator" content="Markov Approximation 1.4 (module: leonardr)">
<meta name="author" content="Leonard Richardson">
</head>
<body>
<a href="foo">foo</a>
<a href="foo"><b>bar</b></a>
</body>
</html>"""
        self.tree = self.soup(self.page)

    def test_pickle_and_unpickle_identity(self):
        # Pickling a tree, then unpickling it, yields a tree identical
        # to the original.
        dumped = pickle.dumps(self.tree, 2)
        loaded = pickle.loads(dumped)
        assert loaded.__class__ == BeautifulSoup
        assert loaded.decode() == self.tree.decode()
        
    def test_deepcopy_identity(self):
        # Making a deepcopy of a tree yields an identical tree.
        copied = copy.deepcopy(self.tree)
        assert copied.decode() == self.tree.decode()

    def test_copy_deeply_nested_document(self):
        # This test verifies that copy and deepcopy don't involve any
        # recursive function calls. If they did, this test would
        # overflow the Python interpreter stack.
        limit = sys.getrecursionlimit() + 1
        markup = "<span>" * limit

        soup = self.soup(markup)
        
        copied = copy.copy(soup)
        copied = copy.deepcopy(soup)

    def test_copy_preserves_encoding(self):
        soup = BeautifulSoup(b'<p>&nbsp;</p>', 'html.parser')
        encoding = soup.original_encoding
        copy = soup.__copy__()
        assert "<p> </p>" == str(copy)
        assert encoding == copy.original_encoding

    def test_copy_preserves_builder_information(self):

        tag = self.soup('<p></p>').p

        # Simulate a tag obtained from a source file.
        tag.sourceline = 10
        tag.sourcepos = 33
        
        copied = tag.__copy__()

        # The TreeBuilder object is no longer availble, but information
        # obtained from it gets copied over to the new Tag object.
        assert tag.sourceline == copied.sourceline
        assert tag.sourcepos == copied.sourcepos
        assert tag.can_be_empty_element == copied.can_be_empty_element
        assert tag.cdata_list_attributes == copied.cdata_list_attributes
        assert tag.preserve_whitespace_tags == copied.preserve_whitespace_tags
        assert tag.interesting_string_types == copied.interesting_string_types
        
    def test_unicode_pickle(self):
        # A tree containing Unicode characters can be pickled.
        html = "<b>\N{SNOWMAN}</b>"
        soup = self.soup(html)
        dumped = pickle.dumps(soup, pickle.HIGHEST_PROTOCOL)
        loaded = pickle.loads(dumped)
        assert loaded.decode() == soup.decode()

    def test_copy_navigablestring_is_not_attached_to_tree(self):
        html = "<b>Foo<a></a></b><b>Bar</b>"
        soup = self.soup(html)
        s1 = soup.find(string="Foo")
        s2 = copy.copy(s1)
        assert s1 == s2
        assert None == s2.parent
        assert None == s2.next_element
        assert None != s1.next_sibling
        assert None == s2.next_sibling
        assert None == s2.previous_element

    def test_copy_navigablestring_subclass_has_same_type(self):
        html = "<b><!--Foo--></b>"
        soup = self.soup(html)
        s1 = soup.string
        s2 = copy.copy(s1)
        assert s1 == s2
        assert isinstance(s2, Comment)

    def test_copy_entire_soup(self):
        html = "<div><b>Foo<a></a></b><b>Bar</b></div>end"
        soup = self.soup(html)
        soup_copy = copy.copy(soup)
        assert soup == soup_copy

    def test_copy_tag_copies_contents(self):
        html = "<div><b>Foo<a></a></b><b>Bar</b></div>end"
        soup = self.soup(html)
        div = soup.div
        div_copy = copy.copy(div)

        # The two tags look the same, and evaluate to equal.
        assert str(div) == str(div_copy)
        assert div == div_copy

        # But they're not the same object.
        assert div is not div_copy

        # And they don't have the same relation to the parse tree. The
        # copy is not associated with a parse tree at all.
        assert None == div_copy.parent
        assert None == div_copy.previous_element
        assert None == div_copy.find(string='Bar').next_element
        assert None != div.find(string='Bar').next_element

