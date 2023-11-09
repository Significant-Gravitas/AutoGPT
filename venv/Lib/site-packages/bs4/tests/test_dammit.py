# encoding: utf-8
import pytest
import logging
import bs4
from bs4 import BeautifulSoup
from bs4.dammit import (
    EntitySubstitution,
    EncodingDetector,
    UnicodeDammit,
)

class TestUnicodeDammit(object):
    """Standalone tests of UnicodeDammit."""

    def test_unicode_input(self):
        markup = "I'm already Unicode! \N{SNOWMAN}"
        dammit = UnicodeDammit(markup)
        assert dammit.unicode_markup == markup

    @pytest.mark.parametrize(
        "smart_quotes_to,expect_converted",
        [(None, "\u2018\u2019\u201c\u201d"),
         ("xml", "&#x2018;&#x2019;&#x201C;&#x201D;"),
         ("html", "&lsquo;&rsquo;&ldquo;&rdquo;"),
         ("ascii", "''" + '""'),
        ]
    )
    def test_smart_quotes_to(self, smart_quotes_to, expect_converted):
        """Verify the functionality of the smart_quotes_to argument
        to the UnicodeDammit constructor."""
        markup = b"<foo>\x91\x92\x93\x94</foo>"
        converted = UnicodeDammit(
            markup, known_definite_encodings=["windows-1252"],
            smart_quotes_to=smart_quotes_to
        ).unicode_markup
        assert converted == "<foo>{}</foo>".format(expect_converted)
        
    def test_detect_utf8(self):
        utf8 = b"Sacr\xc3\xa9 bleu! \xe2\x98\x83"
        dammit = UnicodeDammit(utf8)
        assert dammit.original_encoding.lower() == 'utf-8'
        assert dammit.unicode_markup == 'Sacr\xe9 bleu! \N{SNOWMAN}'

    def test_convert_hebrew(self):
        hebrew = b"\xed\xe5\xec\xf9"
        dammit = UnicodeDammit(hebrew, ["iso-8859-8"])
        assert dammit.original_encoding.lower() == 'iso-8859-8'
        assert dammit.unicode_markup == '\u05dd\u05d5\u05dc\u05e9'

    def test_dont_see_smart_quotes_where_there_are_none(self):
        utf_8 = b"\343\202\261\343\203\274\343\202\277\343\202\244 Watch"
        dammit = UnicodeDammit(utf_8)
        assert dammit.original_encoding.lower() == 'utf-8'
        assert dammit.unicode_markup.encode("utf-8") == utf_8

    def test_ignore_inappropriate_codecs(self):
        utf8_data = "Räksmörgås".encode("utf-8")
        dammit = UnicodeDammit(utf8_data, ["iso-8859-8"])
        assert dammit.original_encoding.lower() == 'utf-8'

    def test_ignore_invalid_codecs(self):
        utf8_data = "Räksmörgås".encode("utf-8")
        for bad_encoding in ['.utf8', '...', 'utF---16.!']:
            dammit = UnicodeDammit(utf8_data, [bad_encoding])
            assert dammit.original_encoding.lower() == 'utf-8'

    def test_exclude_encodings(self):
        # This is UTF-8.
        utf8_data = "Räksmörgås".encode("utf-8")

        # But if we exclude UTF-8 from consideration, the guess is
        # Windows-1252.
        dammit = UnicodeDammit(utf8_data, exclude_encodings=["utf-8"])
        assert dammit.original_encoding.lower() == 'windows-1252'

        # And if we exclude that, there is no valid guess at all.
        dammit = UnicodeDammit(
            utf8_data, exclude_encodings=["utf-8", "windows-1252"])
        assert dammit.original_encoding == None

class TestEncodingDetector(object):
        
    def test_encoding_detector_replaces_junk_in_encoding_name_with_replacement_character(self):
        detected = EncodingDetector(
            b'<?xml version="1.0" encoding="UTF-\xdb" ?>')
        encodings = list(detected.encodings)
        assert 'utf-\N{REPLACEMENT CHARACTER}' in encodings

    def test_detect_html5_style_meta_tag(self):

        for data in (
            b'<html><meta charset="euc-jp" /></html>',
            b"<html><meta charset='euc-jp' /></html>",
            b"<html><meta charset=euc-jp /></html>",
            b"<html><meta charset=euc-jp/></html>"):
            dammit = UnicodeDammit(data, is_html=True)
            assert "euc-jp" == dammit.original_encoding

    def test_last_ditch_entity_replacement(self):
        # This is a UTF-8 document that contains bytestrings
        # completely incompatible with UTF-8 (ie. encoded with some other
        # encoding).
        #
        # Since there is no consistent encoding for the document,
        # Unicode, Dammit will eventually encode the document as UTF-8
        # and encode the incompatible characters as REPLACEMENT
        # CHARACTER.
        #
        # If chardet is installed, it will detect that the document
        # can be converted into ISO-8859-1 without errors. This happens
        # to be the wrong encoding, but it is a consistent encoding, so the
        # code we're testing here won't run.
        #
        # So we temporarily disable chardet if it's present.
        doc = b"""\357\273\277<?xml version="1.0" encoding="UTF-8"?>
<html><b>\330\250\330\252\330\261</b>
<i>\310\322\321\220\312\321\355\344</i></html>"""
        chardet = bs4.dammit.chardet_dammit
        logging.disable(logging.WARNING)
        try:
            def noop(str):
                return None
            bs4.dammit.chardet_dammit = noop
            dammit = UnicodeDammit(doc)
            assert True == dammit.contains_replacement_characters
            assert "\ufffd" in dammit.unicode_markup

            soup = BeautifulSoup(doc, "html.parser")
            assert soup.contains_replacement_characters
        finally:
            logging.disable(logging.NOTSET)
            bs4.dammit.chardet_dammit = chardet

    def test_byte_order_mark_removed(self):
        # A document written in UTF-16LE will have its byte order marker stripped.
        data = b'\xff\xfe<\x00a\x00>\x00\xe1\x00\xe9\x00<\x00/\x00a\x00>\x00'
        dammit = UnicodeDammit(data)
        assert "<a>áé</a>" == dammit.unicode_markup
        assert "utf-16le" == dammit.original_encoding
       
    def test_known_definite_versus_user_encodings(self):
        # The known_definite_encodings are used before sniffing the
        # byte-order mark; the user_encodings are used afterwards.

        # Here's a document in UTF-16LE.
        data = b'\xff\xfe<\x00a\x00>\x00\xe1\x00\xe9\x00<\x00/\x00a\x00>\x00'
        dammit = UnicodeDammit(data)

        # We can process it as UTF-16 by passing it in as a known
        # definite encoding.
        before = UnicodeDammit(data, known_definite_encodings=["utf-16"])
        assert "utf-16" == before.original_encoding
        
        # If we pass UTF-18 as a user encoding, it's not even
        # tried--the encoding sniffed from the byte-order mark takes
        # precedence.
        after = UnicodeDammit(data, user_encodings=["utf-8"])
        assert "utf-16le" == after.original_encoding
        assert ["utf-16le"] == [x[0] for x in dammit.tried_encodings]
        
        # Here's a document in ISO-8859-8.
        hebrew = b"\xed\xe5\xec\xf9"
        dammit = UnicodeDammit(hebrew, known_definite_encodings=["utf-8"],
                               user_encodings=["iso-8859-8"])
        
        # The known_definite_encodings don't work, BOM sniffing does
        # nothing (it only works for a few UTF encodings), but one of
        # the user_encodings does work.
        assert "iso-8859-8" == dammit.original_encoding
        assert ["utf-8", "iso-8859-8"] == [x[0] for x in dammit.tried_encodings]
        
    def test_deprecated_override_encodings(self):
        # override_encodings is a deprecated alias for
        # known_definite_encodings.
        hebrew = b"\xed\xe5\xec\xf9"
        dammit = UnicodeDammit(
            hebrew,
            known_definite_encodings=["shift-jis"],
            override_encodings=["utf-8"],
            user_encodings=["iso-8859-8"],
        )
        assert "iso-8859-8" == dammit.original_encoding

        # known_definite_encodings and override_encodings were tried
        # before user_encodings.
        assert ["shift-jis", "utf-8", "iso-8859-8"] == (
            [x[0] for x in dammit.tried_encodings]
        )

    def test_detwingle(self):
        # Here's a UTF8 document.
        utf8 = ("\N{SNOWMAN}" * 3).encode("utf8")

        # Here's a Windows-1252 document.
        windows_1252 = (
            "\N{LEFT DOUBLE QUOTATION MARK}Hi, I like Windows!"
            "\N{RIGHT DOUBLE QUOTATION MARK}").encode("windows_1252")

        # Through some unholy alchemy, they've been stuck together.
        doc = utf8 + windows_1252 + utf8

        # The document can't be turned into UTF-8:
        with pytest.raises(UnicodeDecodeError):
            doc.decode("utf8")

        # Unicode, Dammit thinks the whole document is Windows-1252,
        # and decodes it into "â˜ƒâ˜ƒâ˜ƒ“Hi, I like Windows!”â˜ƒâ˜ƒâ˜ƒ"

        # But if we run it through fix_embedded_windows_1252, it's fixed:
        fixed = UnicodeDammit.detwingle(doc)
        assert "☃☃☃“Hi, I like Windows!”☃☃☃" == fixed.decode("utf8")

    def test_detwingle_ignores_multibyte_characters(self):
        # Each of these characters has a UTF-8 representation ending
        # in \x93. \x93 is a smart quote if interpreted as
        # Windows-1252. But our code knows to skip over multibyte
        # UTF-8 characters, so they'll survive the process unscathed.
        for tricky_unicode_char in (
            "\N{LATIN SMALL LIGATURE OE}", # 2-byte char '\xc5\x93'
            "\N{LATIN SUBSCRIPT SMALL LETTER X}", # 3-byte char '\xe2\x82\x93'
            "\xf0\x90\x90\x93", # This is a CJK character, not sure which one.
            ):
            input = tricky_unicode_char.encode("utf8")
            assert input.endswith(b'\x93')
            output = UnicodeDammit.detwingle(input)
            assert output == input

    def test_find_declared_encoding(self):
        # Test our ability to find a declared encoding inside an
        # XML or HTML document.
        #
        # Even if the document comes in as Unicode, it may be
        # interesting to know what encoding was claimed
        # originally.

        html_unicode = '<html><head><meta charset="utf-8"></head></html>'
        html_bytes = html_unicode.encode("ascii")

        xml_unicode= '<?xml version="1.0" encoding="ISO-8859-1" ?>'
        xml_bytes = xml_unicode.encode("ascii")

        m = EncodingDetector.find_declared_encoding
        assert m(html_unicode, is_html=False) is None
        assert "utf-8" == m(html_unicode, is_html=True)
        assert "utf-8" == m(html_bytes, is_html=True)

        assert "iso-8859-1" == m(xml_unicode)
        assert "iso-8859-1" == m(xml_bytes)

        # Normally, only the first few kilobytes of a document are checked for
        # an encoding.
        spacer = b' ' * 5000
        assert m(spacer + html_bytes) is None
        assert m(spacer + xml_bytes) is None

        # But you can tell find_declared_encoding to search an entire
        # HTML document.
        assert (
            m(spacer + html_bytes, is_html=True, search_entire_document=True)
            == "utf-8"
        )

        # The XML encoding declaration has to be the very first thing
        # in the document. We'll allow whitespace before the document
        # starts, but nothing else.
        assert m(xml_bytes, search_entire_document=True) == "iso-8859-1"
        assert m(b' ' + xml_bytes, search_entire_document=True) == "iso-8859-1"
        assert m(b'a' + xml_bytes, search_entire_document=True) is None


class TestEntitySubstitution(object):
    """Standalone tests of the EntitySubstitution class."""
    def setup_method(self):
        self.sub = EntitySubstitution


    @pytest.mark.parametrize(
        "original,substituted",
        [
            # Basic case. Unicode characters corresponding to named
            # HTML entites are substituted; others are not.
            ("foo\u2200\N{SNOWMAN}\u00f5bar",
             "foo&forall;\N{SNOWMAN}&otilde;bar"),

            # MS smart quotes are a common source of frustration, so we
            # give them a special test.
            ('‘’foo“”', "&lsquo;&rsquo;foo&ldquo;&rdquo;"),           
        ]
    )
    def test_substitute_html(self, original, substituted):
        assert self.sub.substitute_html(original) == substituted
        
    def test_html5_entity(self):
        for entity, u in (
            # A few spot checks of our ability to recognize
            # special character sequences and convert them
            # to named entities.
            ('&models;', '\u22a7'),
            ('&Nfr;', '\U0001d511'),
            ('&ngeqq;', '\u2267\u0338'),
            ('&not;', '\xac'),
            ('&Not;', '\u2aec'),
                
            # We _could_ convert | to &verbarr;, but we don't, because
            # | is an ASCII character.
            ('|' '|'),

            # Similarly for the fj ligature, which we could convert to
            # &fjlig;, but we don't.
            ("fj", "fj"),

            # We do convert _these_ ASCII characters to HTML entities,
            # because that's required to generate valid HTML.
            ('&gt;', '>'),
            ('&lt;', '<'),
            ('&amp;', '&'),
        ):
            template = '3 %s 4'
            raw = template % u
            with_entities = template % entity
            assert self.sub.substitute_html(raw) == with_entities
            
    def test_html5_entity_with_variation_selector(self):
        # Some HTML5 entities correspond either to a single-character
        # Unicode sequence _or_ to the same character plus U+FE00,
        # VARIATION SELECTOR 1. We can handle this.
        data = "fjords \u2294 penguins"
        markup = "fjords &sqcup; penguins"
        assert self.sub.substitute_html(data) == markup

        data = "fjords \u2294\ufe00 penguins"
        markup = "fjords &sqcups; penguins"
        assert self.sub.substitute_html(data) == markup
        
    def test_xml_converstion_includes_no_quotes_if_make_quoted_attribute_is_false(self):
        s = 'Welcome to "my bar"'
        assert self.sub.substitute_xml(s, False) == s

    def test_xml_attribute_quoting_normally_uses_double_quotes(self):
        assert self.sub.substitute_xml("Welcome", True) == '"Welcome"'
        assert self.sub.substitute_xml("Bob's Bar", True) == '"Bob\'s Bar"'

    def test_xml_attribute_quoting_uses_single_quotes_when_value_contains_double_quotes(self):
        s = 'Welcome to "my bar"'
        assert self.sub.substitute_xml(s, True) == "'Welcome to \"my bar\"'"

    def test_xml_attribute_quoting_escapes_single_quotes_when_value_contains_both_single_and_double_quotes(self):
        s = 'Welcome to "Bob\'s Bar"'
        assert self.sub.substitute_xml(s, True) == '"Welcome to &quot;Bob\'s Bar&quot;"'

    def test_xml_quotes_arent_escaped_when_value_is_not_being_quoted(self):
        quoted = 'Welcome to "Bob\'s Bar"'
        assert self.sub.substitute_xml(quoted) == quoted

    def test_xml_quoting_handles_angle_brackets(self):
        assert self.sub.substitute_xml("foo<bar>") == "foo&lt;bar&gt;"

    def test_xml_quoting_handles_ampersands(self):
        assert self.sub.substitute_xml("AT&T") == "AT&amp;T"

    def test_xml_quoting_including_ampersands_when_they_are_part_of_an_entity(self):
        assert self.sub.substitute_xml("&Aacute;T&T") == "&amp;Aacute;T&amp;T"

    def test_xml_quoting_ignoring_ampersands_when_they_are_part_of_an_entity(self):
        assert self.sub.substitute_xml_containing_entities("&Aacute;T&T") == "&Aacute;T&amp;T"
       
    def test_quotes_not_html_substituted(self):
        """There's no need to do this except inside attribute values."""
        text = 'Bob\'s "bar"'
        assert self.sub.substitute_html(text) == text
