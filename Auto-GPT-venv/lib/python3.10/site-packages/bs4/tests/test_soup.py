# -*- coding: utf-8 -*-
"""Tests of Beautiful Soup as a whole."""

from pdb import set_trace
import logging
import os
import pickle
import pytest
import sys
import tempfile

from bs4 import (
    BeautifulSoup,
    BeautifulStoneSoup,
    GuessedAtParserWarning,
    MarkupResemblesLocatorWarning,
    dammit,
)
from bs4.builder import (
    builder_registry,
    TreeBuilder,
    ParserRejectedMarkup,
)
from bs4.element import (
    Comment,
    SoupStrainer,
    PYTHON_SPECIFIC_ENCODINGS,
    Tag,
    NavigableString,
)

from . import (
    default_builder,
    LXML_PRESENT,
    SoupTest,
)
import warnings
    
class TestConstructor(SoupTest):

    def test_short_unicode_input(self):
        data = "<h1>éé</h1>"
        soup = self.soup(data)
        assert "éé" == soup.h1.string

    def test_embedded_null(self):
        data = "<h1>foo\0bar</h1>"
        soup = self.soup(data)
        assert "foo\0bar" == soup.h1.string

    def test_exclude_encodings(self):
        utf8_data = "Räksmörgås".encode("utf-8")
        soup = self.soup(utf8_data, exclude_encodings=["utf-8"])
        assert "windows-1252" == soup.original_encoding

    def test_custom_builder_class(self):
        # Verify that you can pass in a custom Builder class and
        # it'll be instantiated with the appropriate keyword arguments.
        class Mock(object):
            def __init__(self, **kwargs):
                self.called_with = kwargs
                self.is_xml = True
                self.store_line_numbers = False
                self.cdata_list_attributes = []
                self.preserve_whitespace_tags = []
                self.string_containers = {}
            def initialize_soup(self, soup):
                pass
            def feed(self, markup):
                self.fed = markup
            def reset(self):
                pass
            def ignore(self, ignore):
                pass
            set_up_substitutions = can_be_empty_element = ignore
            def prepare_markup(self, *args, **kwargs):
                yield "prepared markup", "original encoding", "declared encoding", "contains replacement characters"
                
        kwargs = dict(
            var="value",
            # This is a deprecated BS3-era keyword argument, which
            # will be stripped out.
            convertEntities=True,
        )
        with warnings.catch_warnings(record=True):
            soup = BeautifulSoup('', builder=Mock, **kwargs)
        assert isinstance(soup.builder, Mock)
        assert dict(var="value") == soup.builder.called_with
        assert "prepared markup" == soup.builder.fed
        
        # You can also instantiate the TreeBuilder yourself. In this
        # case, that specific object is used and any keyword arguments
        # to the BeautifulSoup constructor are ignored.
        builder = Mock(**kwargs)
        with warnings.catch_warnings(record=True) as w:
            soup = BeautifulSoup(
                '', builder=builder, ignored_value=True,
            )
        msg = str(w[0].message)
        assert msg.startswith("Keyword arguments to the BeautifulSoup constructor will be ignored.")
        assert builder == soup.builder
        assert kwargs == builder.called_with

    def test_parser_markup_rejection(self):
        # If markup is completely rejected by the parser, an
        # explanatory ParserRejectedMarkup exception is raised.
        class Mock(TreeBuilder):
            def feed(self, *args, **kwargs):
                raise ParserRejectedMarkup("Nope.")

        def prepare_markup(self, *args, **kwargs):
            # We're going to try two different ways of preparing this markup,
            # but feed() will reject both of them.
            yield markup, None, None, False
            yield markup, None, None, False
            

        import re
        with pytest.raises(ParserRejectedMarkup) as exc_info:
            BeautifulSoup('', builder=Mock)
        assert "The markup you provided was rejected by the parser. Trying a different parser or a different encoding may help." in str(exc_info.value)
        
    def test_cdata_list_attributes(self):
        # Most attribute values are represented as scalars, but the
        # HTML standard says that some attributes, like 'class' have
        # space-separated lists as values.
        markup = '<a id=" an id " class=" a class "></a>'
        soup = self.soup(markup)

        # Note that the spaces are stripped for 'class' but not for 'id'.
        a = soup.a
        assert " an id " == a['id']
        assert ["a", "class"] == a['class']

        # TreeBuilder takes an argument called 'multi_valued_attributes'  which lets
        # you customize or disable this. As always, you can customize the TreeBuilder
        # by passing in a keyword argument to the BeautifulSoup constructor.
        soup = self.soup(markup, builder=default_builder, multi_valued_attributes=None)
        assert " a class " == soup.a['class']

        # Here are two ways of saying that `id` is a multi-valued
        # attribute in this context, but 'class' is not.
        for switcheroo in ({'*': 'id'}, {'a': 'id'}):
            with warnings.catch_warnings(record=True) as w:
                # This will create a warning about not explicitly
                # specifying a parser, but we'll ignore it.
                soup = self.soup(markup, builder=None, multi_valued_attributes=switcheroo)
            a = soup.a
            assert ["an", "id"] == a['id']
            assert " a class " == a['class']

    def test_replacement_classes(self):
        # Test the ability to pass in replacements for element classes
        # which will be used when building the tree.
        class TagPlus(Tag):
            pass

        class StringPlus(NavigableString):
            pass

        class CommentPlus(Comment):
            pass
        
        soup = self.soup(
            "<a><b>foo</b>bar</a><!--whee-->",
            element_classes = {
                Tag: TagPlus,
                NavigableString: StringPlus,
                Comment: CommentPlus,
            }
        )

        # The tree was built with TagPlus, StringPlus, and CommentPlus objects,
        # rather than Tag, String, and Comment objects.
        assert all(
            isinstance(x, (TagPlus, StringPlus, CommentPlus))
            for x in soup.recursiveChildGenerator()
        )

    def test_alternate_string_containers(self):
        # Test the ability to customize the string containers for
        # different types of tags.
        class PString(NavigableString):
            pass

        class BString(NavigableString):
            pass

        soup = self.soup(
            "<div>Hello.<p>Here is <b>some <i>bolded</i></b> text",
            string_containers = {
                'b': BString,
                'p': PString,
            }
        )

        # The string before the <p> tag is a regular NavigableString.
        assert isinstance(soup.div.contents[0], NavigableString)
        
        # The string inside the <p> tag, but not inside the <i> tag,
        # is a PString.
        assert isinstance(soup.p.contents[0], PString)

        # Every string inside the <b> tag is a BString, even the one that
        # was also inside an <i> tag.
        for s in soup.b.strings:
            assert isinstance(s, BString)

        # Now that parsing was complete, the string_container_stack
        # (where this information was kept) has been cleared out.
        assert [] == soup.string_container_stack


class TestOutput(SoupTest):

    @pytest.mark.parametrize(
        "eventual_encoding,actual_encoding", [
            ("utf-8", "utf-8"),
            ("utf-16", "utf-16"),
        ]
    )
    def test_decode_xml_declaration(self, eventual_encoding, actual_encoding):
        # Most of the time, calling decode() on an XML document will
        # give you a document declaration that mentions the encoding
        # you intend to use when encoding the document as a
        # bytestring.
        soup = self.soup("<tag></tag>")
        soup.is_xml = True
        assert (f'<?xml version="1.0" encoding="{actual_encoding}"?>\n<tag></tag>'
                == soup.decode(eventual_encoding=eventual_encoding))

    @pytest.mark.parametrize(
        "eventual_encoding", [x for x in PYTHON_SPECIFIC_ENCODINGS] + [None]
    )
    def test_decode_xml_declaration_with_missing_or_python_internal_eventual_encoding(self, eventual_encoding):
        # But if you pass a Python internal encoding into decode(), or
        # omit the eventual_encoding altogether, the document
        # declaration won't mention any particular encoding.
        soup = BeautifulSoup("<tag></tag>", "html.parser")
        soup.is_xml = True
        assert (f'<?xml version="1.0"?>\n<tag></tag>'
                == soup.decode(eventual_encoding=eventual_encoding))

    def test(self):
        # BeautifulSoup subclasses Tag and extends the decode() method.
        # Make sure the other Tag methods which call decode() call
        # it correctly.
        soup = self.soup("<tag></tag>")
        assert b"<tag></tag>" == soup.encode(encoding="utf-8")
        assert b"<tag></tag>" == soup.encode_contents(encoding="utf-8")
        assert "<tag></tag>" == soup.decode_contents()
        assert "<tag>\n</tag>\n" == soup.prettify()

        
class TestWarnings(SoupTest):
    # Note that some of the tests in this class create BeautifulSoup
    # objects directly rather than using self.soup(). That's
    # because SoupTest.soup is defined in a different file,
    # which will throw off the assertion in _assert_warning
    # that the code that triggered the warning is in the same
    # file as the test.

    def _assert_warning(self, warnings, cls):
        for w in warnings:
            if isinstance(w.message, cls):
                assert w.filename == __file__
                return w
        raise Exception("%s warning not found in %r" % (cls, warnings))
    
    def _assert_no_parser_specified(self, w):
        warning = self._assert_warning(w, GuessedAtParserWarning)
        message = str(warning.message)
        assert message.startswith(BeautifulSoup.NO_PARSER_SPECIFIED_WARNING[:60])

    def test_warning_if_no_parser_specified(self):
        with warnings.catch_warnings(record=True) as w:
            soup = BeautifulSoup("<a><b></b></a>")
        self._assert_no_parser_specified(w)

    def test_warning_if_parser_specified_too_vague(self):
        with warnings.catch_warnings(record=True) as w:
            soup = BeautifulSoup("<a><b></b></a>", "html")
        self._assert_no_parser_specified(w)

    def test_no_warning_if_explicit_parser_specified(self):
        with warnings.catch_warnings(record=True) as w:
            soup = self.soup("<a><b></b></a>")
        assert [] == w

    def test_parseOnlyThese_renamed_to_parse_only(self):
        with warnings.catch_warnings(record=True) as w:
            soup = BeautifulSoup(
                "<a><b></b></a>", "html.parser",
                parseOnlyThese=SoupStrainer("b"),
            )
        warning = self._assert_warning(w, DeprecationWarning)
        msg = str(warning.message)
        assert "parseOnlyThese" in msg
        assert "parse_only" in msg
        assert b"<b></b>" == soup.encode()

    def test_fromEncoding_renamed_to_from_encoding(self):
        with warnings.catch_warnings(record=True) as w:
            utf8 = b"\xc3\xa9"
            soup = BeautifulSoup(
                utf8, "html.parser", fromEncoding="utf8"
            )
        warning = self._assert_warning(w, DeprecationWarning)
        msg = str(warning.message)
        assert "fromEncoding" in msg
        assert "from_encoding" in msg
        assert "utf8" == soup.original_encoding

    def test_unrecognized_keyword_argument(self):
        with pytest.raises(TypeError):
            self.soup("<a>", no_such_argument=True)

    @pytest.mark.parametrize(
        "extension",
        ['markup.html', 'markup.htm', 'markup.HTML', 'markup.txt',
         'markup.xhtml', 'markup.xml', "/home/user/file", "c:\\user\file"]
    )
    def test_resembles_filename_warning(self, extension):
        # A warning is issued if the "markup" looks like the name of
        # an HTML or text file, or a full path to a file on disk.
        with warnings.catch_warnings(record=True) as w:
            soup = BeautifulSoup("markup" + extension, "html.parser")
            warning = self._assert_warning(w, MarkupResemblesLocatorWarning)
            assert "looks more like a filename" in str(warning.message)

    @pytest.mark.parametrize(
        "extension",
        ['markuphtml', 'markup.com', '', 'markup.js']
    )
    def test_resembles_filename_no_warning(self, extension):
        # The 'looks more like a filename' warning is not issued if
        # the markup looks like a bare string, a domain name, or a
        # file that's not an HTML file.
        with warnings.catch_warnings(record=True) as w:
            soup = self.soup("markup" + extension)
        assert [] == w

    def test_url_warning_with_bytes_url(self):
        url = b"http://www.crummybytes.com/"
        with warnings.catch_warnings(record=True) as warning_list:
            soup = BeautifulSoup(url, "html.parser")
        warning = self._assert_warning(
            warning_list, MarkupResemblesLocatorWarning
        )
        assert "looks more like a URL" in str(warning.message)
        assert url not in str(warning.message).encode("utf8")
        
    def test_url_warning_with_unicode_url(self):
        url = "http://www.crummyunicode.com/"
        with warnings.catch_warnings(record=True) as warning_list:
            # note - this url must differ from the bytes one otherwise
            # python's warnings system swallows the second warning
            soup = BeautifulSoup(url, "html.parser")
        warning = self._assert_warning(
            warning_list, MarkupResemblesLocatorWarning
        )
        assert "looks more like a URL" in str(warning.message)
        assert url not in str(warning.message)

    def test_url_warning_with_bytes_and_space(self):
        # Here the markup contains something besides a URL, so no warning
        # is issued.
        with warnings.catch_warnings(record=True) as warning_list:
            soup = self.soup(b"http://www.crummybytes.com/ is great")
        assert not any("looks more like a URL" in str(w.message) 
                       for w in warning_list)

    def test_url_warning_with_unicode_and_space(self):
        with warnings.catch_warnings(record=True) as warning_list:
            soup = self.soup("http://www.crummyunicode.com/ is great")
        assert not any("looks more like a URL" in str(w.message) 
                       for w in warning_list)


class TestSelectiveParsing(SoupTest):

    def test_parse_with_soupstrainer(self):
        markup = "No<b>Yes</b><a>No<b>Yes <c>Yes</c></b>"
        strainer = SoupStrainer("b")
        soup = self.soup(markup, parse_only=strainer)
        assert soup.encode() == b"<b>Yes</b><b>Yes <c>Yes</c></b>"

        
class TestNewTag(SoupTest):
    """Test the BeautifulSoup.new_tag() method."""
    def test_new_tag(self):
        soup = self.soup("")
        new_tag = soup.new_tag("foo", bar="baz", attrs={"name": "a name"})
        assert isinstance(new_tag, Tag)
        assert "foo" == new_tag.name
        assert dict(bar="baz", name="a name") == new_tag.attrs
        assert None == new_tag.parent

    @pytest.mark.skipif(
        not LXML_PRESENT,
        reason="lxml not installed, cannot parse XML document"
    )
    def test_xml_tag_inherits_self_closing_rules_from_builder(self):
        xml_soup = BeautifulSoup("", "xml")
        xml_br = xml_soup.new_tag("br")
        xml_p = xml_soup.new_tag("p")

        # Both the <br> and <p> tag are empty-element, just because
        # they have no contents.
        assert b"<br/>" == xml_br.encode()
        assert b"<p/>" == xml_p.encode()

    def test_tag_inherits_self_closing_rules_from_builder(self):
        html_soup = BeautifulSoup("", "html.parser")
        html_br = html_soup.new_tag("br")
        html_p = html_soup.new_tag("p")

        # The HTML builder users HTML's rules about which tags are
        # empty-element tags, and the new tags reflect these rules.
        assert b"<br/>" == html_br.encode()
        assert b"<p></p>" == html_p.encode()

class TestNewString(SoupTest):
    """Test the BeautifulSoup.new_string() method."""
    def test_new_string_creates_navigablestring(self):
        soup = self.soup("")
        s = soup.new_string("foo")
        assert "foo" == s
        assert isinstance(s, NavigableString)

    def test_new_string_can_create_navigablestring_subclass(self):
        soup = self.soup("")
        s = soup.new_string("foo", Comment)
        assert "foo" == s
        assert isinstance(s, Comment)


class TestPickle(SoupTest):
   # Test our ability to pickle the BeautifulSoup object itself.

    def test_normal_pickle(self):
        soup = self.soup("<a>some markup</a>")
        pickled = pickle.dumps(soup)
        unpickled = pickle.loads(pickled)
        assert "some markup" == unpickled.a.string
        
    def test_pickle_with_no_builder(self):
        # We had a bug that prevented pickling from working if
        # the builder wasn't set.
        soup = self.soup("some markup")
        soup.builder = None
        pickled = pickle.dumps(soup)
        unpickled = pickle.loads(pickled)
        assert "some markup" == unpickled.string

class TestEncodingConversion(SoupTest):
    # Test Beautiful Soup's ability to decode and encode from various
    # encodings.

    def setup_method(self):
        self.unicode_data = '<html><head><meta charset="utf-8"/></head><body><foo>Sacr\N{LATIN SMALL LETTER E WITH ACUTE} bleu!</foo></body></html>'
        self.utf8_data = self.unicode_data.encode("utf-8")
        # Just so you know what it looks like.
        assert self.utf8_data == b'<html><head><meta charset="utf-8"/></head><body><foo>Sacr\xc3\xa9 bleu!</foo></body></html>'

    def test_ascii_in_unicode_out(self):
        # ASCII input is converted to Unicode. The original_encoding
        # attribute is set to 'utf-8', a superset of ASCII.
        chardet = dammit.chardet_dammit
        logging.disable(logging.WARNING)
        try:
            def noop(str):
                return None
            # Disable chardet, which will realize that the ASCII is ASCII.
            dammit.chardet_dammit = noop
            ascii = b"<foo>a</foo>"
            soup_from_ascii = self.soup(ascii)
            unicode_output = soup_from_ascii.decode()
            assert isinstance(unicode_output, str)
            assert unicode_output == self.document_for(ascii.decode())
            assert soup_from_ascii.original_encoding.lower() == "utf-8"
        finally:
            logging.disable(logging.NOTSET)
            dammit.chardet_dammit = chardet

    def test_unicode_in_unicode_out(self):
        # Unicode input is left alone. The original_encoding attribute
        # is not set.
        soup_from_unicode = self.soup(self.unicode_data)
        assert soup_from_unicode.decode() == self.unicode_data
        assert soup_from_unicode.foo.string == 'Sacr\xe9 bleu!'
        assert soup_from_unicode.original_encoding == None

    def test_utf8_in_unicode_out(self):
        # UTF-8 input is converted to Unicode. The original_encoding
        # attribute is set.
        soup_from_utf8 = self.soup(self.utf8_data)
        assert soup_from_utf8.decode() == self.unicode_data
        assert soup_from_utf8.foo.string == 'Sacr\xe9 bleu!'

    def test_utf8_out(self):
        # The internal data structures can be encoded as UTF-8.
        soup_from_unicode = self.soup(self.unicode_data)
        assert soup_from_unicode.encode('utf-8') == self.utf8_data
