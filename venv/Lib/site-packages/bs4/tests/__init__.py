# encoding: utf-8
"""Helper classes for tests."""

# Use of this source code is governed by the MIT license.
__license__ = "MIT"

import pickle
import copy
import functools
import warnings
import pytest
from bs4 import BeautifulSoup
from bs4.element import (
    CharsetMetaAttributeValue,
    Comment,
    ContentMetaAttributeValue,
    Doctype,
    PYTHON_SPECIFIC_ENCODINGS,
    SoupStrainer,
    Script,
    Stylesheet,
    Tag
)

from bs4.builder import (
    DetectsXMLParsedAsHTML,
    HTMLParserTreeBuilder,
    XMLParsedAsHTMLWarning,
)
default_builder = HTMLParserTreeBuilder

# Some tests depend on specific third-party libraries. We use
# @pytest.mark.skipIf on the following conditionals to skip them
# if the libraries are not installed.
try:
    from soupsieve import SelectorSyntaxError
    SOUP_SIEVE_PRESENT = True
except ImportError:
    SOUP_SIEVE_PRESENT = False

try:
    import html5lib
    HTML5LIB_PRESENT = True
except ImportError:
    HTML5LIB_PRESENT = False

try:
    import lxml.etree
    LXML_PRESENT = True
    LXML_VERSION = lxml.etree.LXML_VERSION
except ImportError:
    LXML_PRESENT = False
    LXML_VERSION = (0,)

BAD_DOCUMENT = """A bare string
<!DOCTYPE xsl:stylesheet SYSTEM "htmlent.dtd">
<!DOCTYPE xsl:stylesheet PUBLIC "htmlent.dtd">
<div><![CDATA[A CDATA section where it doesn't belong]]></div>
<div><svg><![CDATA[HTML5 does allow CDATA sections in SVG]]></svg></div>
<div>A <meta> tag</div>
<div>A <br> tag that supposedly has contents.</br></div>
<div>AT&T</div>
<div><textarea>Within a textarea, markup like <b> tags and <&<&amp; should be treated as literal</textarea></div>
<div><script>if (i < 2) { alert("<b>Markup within script tags should be treated as literal.</b>"); }</script></div>
<div>This numeric entity is missing the final semicolon: <x t="pi&#241ata"></div>
<div><a href="http://example.com/</a> that attribute value never got closed</div>
<div><a href="foo</a>, </a><a href="bar">that attribute value was closed by the subsequent tag</a></div>
<! This document starts with a bogus declaration ><div>a</div>
<div>This document contains <!an incomplete declaration <div>(do you see it?)</div>
<div>This document ends with <!an incomplete declaration
<div><a style={height:21px;}>That attribute value was bogus</a></div>
<! DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN">The doctype is invalid because it contains extra whitespace
<div><table><td nowrap>That boolean attribute had no value</td></table></div>
<div>Here's a nonexistent entity: &#foo; (do you see it?)</div>
<div>This document ends before the entity finishes: &gt
<div><p>Paragraphs shouldn't contain block display elements, but this one does: <dl><dt>you see?</dt></p>
<b b="20" a="1" b="10" a="2" a="3" a="4">Multiple values for the same attribute.</b>
<div><table><tr><td>Here's a table</td></tr></table></div>
<div><table id="1"><tr><td>Here's a nested table:<table id="2"><tr><td>foo</td></tr></table></td></div>
<div>This tag contains nothing but whitespace: <b>    </b></div>
<div><blockquote><p><b>This p tag is cut off by</blockquote></p>the end of the blockquote tag</div>
<div><table><div>This table contains bare markup</div></table></div>
<div><div id="1">\n <a href="link1">This link is never closed.\n</div>\n<div id="2">\n <div id="3">\n   <a href="link2">This link is closed.</a>\n  </div>\n</div></div>
<div>This document contains a <!DOCTYPE surprise>surprise doctype</div>
<div><a><B><Cd><EFG>Mixed case tags are folded to lowercase</efg></CD></b></A></div>
<div><our\u2603>Tag name contains Unicode characters</our\u2603></div>
<div><a \u2603="snowman">Attribute name contains Unicode characters</a></div>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
"""


class SoupTest(object):

    @property
    def default_builder(self):
        return default_builder

    def soup(self, markup, **kwargs):
        """Build a Beautiful Soup object from markup."""
        builder = kwargs.pop('builder', self.default_builder)
        return BeautifulSoup(markup, builder=builder, **kwargs)

    def document_for(self, markup, **kwargs):
        """Turn an HTML fragment into a document.

        The details depend on the builder.
        """
        return self.default_builder(**kwargs).test_fragment_to_document(markup)
   
    def assert_soup(self, to_parse, compare_parsed_to=None):
        """Parse some markup using Beautiful Soup and verify that
        the output markup is as expected.
        """
        builder = self.default_builder
        obj = BeautifulSoup(to_parse, builder=builder)
        if compare_parsed_to is None:
            compare_parsed_to = to_parse

        # Verify that the documents come out the same.
        assert obj.decode() == self.document_for(compare_parsed_to)

        # Also run some checks on the BeautifulSoup object itself:

        # Verify that every tag that was opened was eventually closed.

        # There are no tags in the open tag counter.
        assert all(v==0 for v in list(obj.open_tag_counter.values()))

        # The only tag in the tag stack is the one for the root
        # document.
        assert [obj.ROOT_TAG_NAME] == [x.name for x in obj.tagStack]

    assertSoupEquals = assert_soup
        
    def assertConnectedness(self, element):
        """Ensure that next_element and previous_element are properly
        set for all descendants of the given element.
        """
        earlier = None
        for e in element.descendants:
            if earlier:
                assert e == earlier.next_element
                assert earlier == e.previous_element
            earlier = e

    def linkage_validator(self, el, _recursive_call=False):
        """Ensure proper linkage throughout the document."""
        descendant = None
        # Document element should have no previous element or previous sibling.
        # It also shouldn't have a next sibling.
        if el.parent is None:
            assert el.previous_element is None,\
                "Bad previous_element\nNODE: {}\nPREV: {}\nEXPECTED: {}".format(
                    el, el.previous_element, None
                )
            assert el.previous_sibling is None,\
                "Bad previous_sibling\nNODE: {}\nPREV: {}\nEXPECTED: {}".format(
                    el, el.previous_sibling, None
                )
            assert el.next_sibling is None,\
                "Bad next_sibling\nNODE: {}\nNEXT: {}\nEXPECTED: {}".format(
                    el, el.next_sibling, None
                )

        idx = 0
        child = None
        last_child = None
        last_idx = len(el.contents) - 1
        for child in el.contents:
            descendant = None

            # Parent should link next element to their first child
            # That child should have no previous sibling
            if idx == 0:
                if el.parent is not None:
                    assert el.next_element is child,\
                       "Bad next_element\nNODE: {}\nNEXT: {}\nEXPECTED: {}".format(
                            el, el.next_element, child
                        )
                    assert child.previous_element is el,\
                       "Bad previous_element\nNODE: {}\nPREV: {}\nEXPECTED: {}".format(
                            child, child.previous_element, el
                        )
                    assert child.previous_sibling is None,\
                       "Bad previous_sibling\nNODE: {}\nPREV {}\nEXPECTED: {}".format(
                            child, child.previous_sibling, None
                        )

            # If not the first child, previous index should link as sibling to this index
            # Previous element should match the last index or the last bubbled up descendant
            else:
                assert child.previous_sibling is el.contents[idx - 1],\
                    "Bad previous_sibling\nNODE: {}\nPREV {}\nEXPECTED {}".format(
                        child, child.previous_sibling, el.contents[idx - 1]
                    )
                assert el.contents[idx - 1].next_sibling is child,\
                    "Bad next_sibling\nNODE: {}\nNEXT {}\nEXPECTED {}".format(
                        el.contents[idx - 1], el.contents[idx - 1].next_sibling, child
                    )

                if last_child is not None:
                    assert child.previous_element is last_child,\
                        "Bad previous_element\nNODE: {}\nPREV {}\nEXPECTED {}\nCONTENTS {}".format(
                            child, child.previous_element, last_child, child.parent.contents
                        )
                    assert last_child.next_element is child,\
                        "Bad next_element\nNODE: {}\nNEXT {}\nEXPECTED {}".format(
                            last_child, last_child.next_element, child
                        )

            if isinstance(child, Tag) and child.contents:
                descendant = self.linkage_validator(child, True)
                # A bubbled up descendant should have no next siblings
                assert descendant.next_sibling is None,\
                    "Bad next_sibling\nNODE: {}\nNEXT {}\nEXPECTED {}".format(
                        descendant, descendant.next_sibling, None
                    )

            # Mark last child as either the bubbled up descendant or the current child
            if descendant is not None:
                last_child = descendant
            else:
                last_child = child

            # If last child, there are non next siblings
            if idx == last_idx:
                assert child.next_sibling is None,\
                    "Bad next_sibling\nNODE: {}\nNEXT {}\nEXPECTED {}".format(
                        child, child.next_sibling, None
                    )
            idx += 1

        child = descendant if descendant is not None else child
        if child is None:
            child = el

        if not _recursive_call and child is not None:
            target = el
            while True:
                if target is None:
                    assert child.next_element is None, \
                        "Bad next_element\nNODE: {}\nNEXT {}\nEXPECTED {}".format(
                            child, child.next_element, None
                        )
                    break
                elif target.next_sibling is not None:
                    assert child.next_element is target.next_sibling, \
                        "Bad next_element\nNODE: {}\nNEXT {}\nEXPECTED {}".format(
                            child, child.next_element, target.next_sibling
                        )
                    break
                target = target.parent

            # We are done, so nothing to return
            return None
        else:
            # Return the child to the recursive caller
            return child

    def assert_selects(self, tags, should_match):
        """Make sure that the given tags have the correct text.

        This is used in tests that define a bunch of tags, each
        containing a single string, and then select certain strings by
        some mechanism.
        """
        assert [tag.string for tag in tags] == should_match

    def assert_selects_ids(self, tags, should_match):
        """Make sure that the given tags have the correct IDs.

        This is used in tests that define a bunch of tags, each
        containing a single string, and then select certain strings by
        some mechanism.
        """
        assert [tag['id'] for tag in tags] == should_match


class TreeBuilderSmokeTest(object):
    # Tests that are common to HTML and XML tree builders.

    @pytest.mark.parametrize(
        "multi_valued_attributes",
        [None, {}, dict(b=['class']), {'*': ['notclass']}]
    )
    def test_attribute_not_multi_valued(self, multi_valued_attributes):
        markup = '<html xmlns="http://www.w3.org/1999/xhtml"><a class="a b c"></html>'
        soup = self.soup(markup, multi_valued_attributes=multi_valued_attributes)
        assert soup.a['class'] == 'a b c'

    @pytest.mark.parametrize(
        "multi_valued_attributes", [dict(a=['class']), {'*': ['class']}]
    )
    def test_attribute_multi_valued(self, multi_valued_attributes):
        markup = '<a class="a b c">'
        soup = self.soup(
            markup, multi_valued_attributes=multi_valued_attributes
        )
        assert soup.a['class'] == ['a', 'b', 'c']

    def test_invalid_doctype(self):
        markup = '<![if word]>content<![endif]>'
        markup = '<!DOCTYPE html]ff>'
        soup = self.soup(markup)

class HTMLTreeBuilderSmokeTest(TreeBuilderSmokeTest):

    """A basic test of a treebuilder's competence.

    Any HTML treebuilder, present or future, should be able to pass
    these tests. With invalid markup, there's room for interpretation,
    and different parsers can handle it differently. But with the
    markup in these tests, there's not much room for interpretation.
    """

    def test_empty_element_tags(self):
        """Verify that all HTML4 and HTML5 empty element (aka void element) tags
        are handled correctly.
        """
        for name in [
                'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'keygen', 'link', 'menuitem', 'meta', 'param', 'source', 'track', 'wbr',
                'spacer', 'frame'
        ]:
            soup = self.soup("")
            new_tag = soup.new_tag(name)
            assert new_tag.is_empty_element == True

    def test_special_string_containers(self):
        soup = self.soup(
            "<style>Some CSS</style><script>Some Javascript</script>"
        )
        assert isinstance(soup.style.string, Stylesheet)
        assert isinstance(soup.script.string, Script)

        soup = self.soup(
            "<style><!--Some CSS--></style>"
        )
        assert isinstance(soup.style.string, Stylesheet)
        # The contents of the style tag resemble an HTML comment, but
        # it's not treated as a comment.
        assert soup.style.string == "<!--Some CSS-->"
        assert isinstance(soup.style.string, Stylesheet)
        
    def test_pickle_and_unpickle_identity(self):
        # Pickling a tree, then unpickling it, yields a tree identical
        # to the original.
        tree = self.soup("<a><b>foo</a>")
        dumped = pickle.dumps(tree, 2)
        loaded = pickle.loads(dumped)
        assert loaded.__class__ == BeautifulSoup
        assert loaded.decode() == tree.decode()

    def assertDoctypeHandled(self, doctype_fragment):
        """Assert that a given doctype string is handled correctly."""
        doctype_str, soup = self._document_with_doctype(doctype_fragment)

        # Make sure a Doctype object was created.
        doctype = soup.contents[0]
        assert doctype.__class__ == Doctype
        assert doctype == doctype_fragment
        assert soup.encode("utf8")[:len(doctype_str)] == doctype_str

        # Make sure that the doctype was correctly associated with the
        # parse tree and that the rest of the document parsed.
        assert soup.p.contents[0] == 'foo'

    def _document_with_doctype(self, doctype_fragment, doctype_string="DOCTYPE"):
        """Generate and parse a document with the given doctype."""
        doctype = '<!%s %s>' % (doctype_string, doctype_fragment)
        markup = doctype + '\n<p>foo</p>'
        soup = self.soup(markup)
        return doctype.encode("utf8"), soup

    def test_normal_doctypes(self):
        """Make sure normal, everyday HTML doctypes are handled correctly."""
        self.assertDoctypeHandled("html")
        self.assertDoctypeHandled(
            'html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"')

    def test_empty_doctype(self):
        soup = self.soup("<!DOCTYPE>")
        doctype = soup.contents[0]
        assert "" == doctype.strip()

    def test_mixed_case_doctype(self):
        # A lowercase or mixed-case doctype becomes a Doctype.
        for doctype_fragment in ("doctype", "DocType"):
            doctype_str, soup = self._document_with_doctype(
                "html", doctype_fragment
            )

            # Make sure a Doctype object was created and that the DOCTYPE
            # is uppercase.
            doctype = soup.contents[0]
            assert doctype.__class__ == Doctype
            assert doctype == "html"
            assert soup.encode("utf8")[:len(doctype_str)] == b"<!DOCTYPE html>"

            # Make sure that the doctype was correctly associated with the
            # parse tree and that the rest of the document parsed.
            assert soup.p.contents[0] == 'foo'
        
    def test_public_doctype_with_url(self):
        doctype = 'html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd"'
        self.assertDoctypeHandled(doctype)

    def test_system_doctype(self):
        self.assertDoctypeHandled('foo SYSTEM "http://www.example.com/"')

    def test_namespaced_system_doctype(self):
        # We can handle a namespaced doctype with a system ID.
        self.assertDoctypeHandled('xsl:stylesheet SYSTEM "htmlent.dtd"')

    def test_namespaced_public_doctype(self):
        # Test a namespaced doctype with a public id.
        self.assertDoctypeHandled('xsl:stylesheet PUBLIC "htmlent.dtd"')

    def test_real_xhtml_document(self):
        """A real XHTML document should come out more or less the same as it went in."""
        markup = b"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN">
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Hello.</title></head>
<body>Goodbye.</body>
</html>"""
        with warnings.catch_warnings(record=True) as w:
            soup = self.soup(markup)
        assert soup.encode("utf-8").replace(b"\n", b"") == markup.replace(b"\n", b"")

        # No warning was issued about parsing an XML document as HTML,
        # because XHTML is both.
        assert w == []


    def test_namespaced_html(self):
        # When a namespaced XML document is parsed as HTML it should
        # be treated as HTML with weird tag names.
        markup = b"""<ns1:foo>content</ns1:foo><ns1:foo/><ns2:foo/>"""
        with warnings.catch_warnings(record=True) as w:
            soup = self.soup(markup)

        assert 2 == len(soup.find_all("ns1:foo"))
            
        # n.b. no "you're parsing XML as HTML" warning was given
        # because there was no XML declaration.
        assert [] == w

    def test_detect_xml_parsed_as_html(self):
        # A warning is issued when parsing an XML document as HTML,
        # but basic stuff should still work.
        markup = b"""<?xml version="1.0" encoding="utf-8"?><tag>string</tag>"""
        with warnings.catch_warnings(record=True) as w:
            soup = self.soup(markup)
            assert soup.tag.string == 'string'
        [warning] = w
        assert isinstance(warning.message, XMLParsedAsHTMLWarning)
        assert str(warning.message) == XMLParsedAsHTMLWarning.MESSAGE

        # NOTE: the warning is not issued if the document appears to
        # be XHTML (tested with test_real_xhtml_document in the
        # superclass) or if there is no XML declaration (tested with
        # test_namespaced_html in the superclass).
        
    def test_processing_instruction(self):
        # We test both Unicode and bytestring to verify that
        # process_markup correctly sets processing_instruction_class
        # even when the markup is already Unicode and there is no
        # need to process anything.
        markup = """<?PITarget PIContent?>"""
        soup = self.soup(markup)
        assert markup == soup.decode()

        markup = b"""<?PITarget PIContent?>"""
        soup = self.soup(markup)
        assert markup == soup.encode("utf8")

    def test_deepcopy(self):
        """Make sure you can copy the tree builder.

        This is important because the builder is part of a
        BeautifulSoup object, and we want to be able to copy that.
        """
        copy.deepcopy(self.default_builder)

    def test_p_tag_is_never_empty_element(self):
        """A <p> tag is never designated as an empty-element tag.

        Even if the markup shows it as an empty-element tag, it
        shouldn't be presented that way.
        """
        soup = self.soup("<p/>")
        assert not soup.p.is_empty_element
        assert str(soup.p) == "<p></p>"

    def test_unclosed_tags_get_closed(self):
        """A tag that's not closed by the end of the document should be closed.

        This applies to all tags except empty-element tags.
        """
        self.assert_soup("<p>", "<p></p>")
        self.assert_soup("<b>", "<b></b>")

        self.assert_soup("<br>", "<br/>")

    def test_br_is_always_empty_element_tag(self):
        """A <br> tag is designated as an empty-element tag.

        Some parsers treat <br></br> as one <br/> tag, some parsers as
        two tags, but it should always be an empty-element tag.
        """
        soup = self.soup("<br></br>")
        assert soup.br.is_empty_element
        assert str(soup.br) == "<br/>"

    def test_nested_formatting_elements(self):
        self.assert_soup("<em><em></em></em>")

    def test_double_head(self):
        html = '''<!DOCTYPE html>
<html>
<head>
<title>Ordinary HEAD element test</title>
</head>
<script type="text/javascript">
alert("Help!");
</script>
<body>
Hello, world!
</body>
</html>
'''
        soup = self.soup(html)
        assert "text/javascript" == soup.find('script')['type']

    def test_comment(self):
        # Comments are represented as Comment objects.
        markup = "<p>foo<!--foobar-->baz</p>"
        self.assert_soup(markup)

        soup = self.soup(markup)
        comment = soup.find(string="foobar")
        assert comment.__class__ == Comment

        # The comment is properly integrated into the tree.
        foo = soup.find(string="foo")
        assert comment == foo.next_element
        baz = soup.find(string="baz")
        assert comment == baz.previous_element

    def test_preserved_whitespace_in_pre_and_textarea(self):
        """Whitespace must be preserved in <pre> and <textarea> tags,
        even if that would mean not prettifying the markup.
        """
        pre_markup = "<pre>a   z</pre>\n"
        textarea_markup = "<textarea> woo\nwoo  </textarea>\n"
        self.assert_soup(pre_markup)
        self.assert_soup(textarea_markup)

        soup = self.soup(pre_markup)
        assert soup.pre.prettify() == pre_markup

        soup = self.soup(textarea_markup)
        assert soup.textarea.prettify() == textarea_markup

        soup = self.soup("<textarea></textarea>")
        assert soup.textarea.prettify() == "<textarea></textarea>\n"

    def test_nested_inline_elements(self):
        """Inline elements can be nested indefinitely."""
        b_tag = "<b>Inside a B tag</b>"
        self.assert_soup(b_tag)

        nested_b_tag = "<p>A <i>nested <b>tag</b></i></p>"
        self.assert_soup(nested_b_tag)

        double_nested_b_tag = "<p>A <a>doubly <i>nested <b>tag</b></i></a></p>"
        self.assert_soup(nested_b_tag)

    def test_nested_block_level_elements(self):
        """Block elements can be nested."""
        soup = self.soup('<blockquote><p><b>Foo</b></p></blockquote>')
        blockquote = soup.blockquote
        assert blockquote.p.b.string == 'Foo'
        assert blockquote.b.string == 'Foo'

    def test_correctly_nested_tables(self):
        """One table can go inside another one."""
        markup = ('<table id="1">'
                  '<tr>'
                  "<td>Here's another table:"
                  '<table id="2">'
                  '<tr><td>foo</td></tr>'
                  '</table></td>')

        self.assert_soup(
            markup,
            '<table id="1"><tr><td>Here\'s another table:'
            '<table id="2"><tr><td>foo</td></tr></table>'
            '</td></tr></table>')

        self.assert_soup(
            "<table><thead><tr><td>Foo</td></tr></thead>"
            "<tbody><tr><td>Bar</td></tr></tbody>"
            "<tfoot><tr><td>Baz</td></tr></tfoot></table>")

    def test_multivalued_attribute_with_whitespace(self):
        # Whitespace separating the values of a multi-valued attribute
        # should be ignored.

        markup = '<div class=" foo bar	 "></a>'
        soup = self.soup(markup)
        assert ['foo', 'bar'] == soup.div['class']

        # If you search by the literal name of the class it's like the whitespace
        # wasn't there.
        assert soup.div == soup.find('div', class_="foo bar")
        
    def test_deeply_nested_multivalued_attribute(self):
        # html5lib can set the attributes of the same tag many times
        # as it rearranges the tree. This has caused problems with
        # multivalued attributes.
        markup = '<table><div><div class="css"></div></div></table>'
        soup = self.soup(markup)
        assert ["css"] == soup.div.div['class']

    def test_multivalued_attribute_on_html(self):
        # html5lib uses a different API to set the attributes ot the
        # <html> tag. This has caused problems with multivalued
        # attributes.
        markup = '<html class="a b"></html>'
        soup = self.soup(markup)
        assert ["a", "b"] == soup.html['class']

    def test_angle_brackets_in_attribute_values_are_escaped(self):
        self.assert_soup('<a b="<a>"></a>', '<a b="&lt;a&gt;"></a>')

    def test_strings_resembling_character_entity_references(self):
        # "&T" and "&p" look like incomplete character entities, but they are
        # not.
        self.assert_soup(
            "<p>&bull; AT&T is in the s&p 500</p>",
            "<p>\u2022 AT&amp;T is in the s&amp;p 500</p>"
        )

    def test_apos_entity(self):
        self.assert_soup(
            "<p>Bob&apos;s Bar</p>",
            "<p>Bob's Bar</p>",
        )
        
    def test_entities_in_foreign_document_encoding(self):
        # &#147; and &#148; are invalid numeric entities referencing
        # Windows-1252 characters. &#45; references a character common
        # to Windows-1252 and Unicode, and &#9731; references a
        # character only found in Unicode.
        #
        # All of these entities should be converted to Unicode
        # characters.
        markup = "<p>&#147;Hello&#148; &#45;&#9731;</p>"
        soup = self.soup(markup)
        assert "“Hello” -☃" == soup.p.string
        
    def test_entities_in_attributes_converted_to_unicode(self):
        expect = '<p id="pi\N{LATIN SMALL LETTER N WITH TILDE}ata"></p>'
        self.assert_soup('<p id="pi&#241;ata"></p>', expect)
        self.assert_soup('<p id="pi&#xf1;ata"></p>', expect)
        self.assert_soup('<p id="pi&#Xf1;ata"></p>', expect)
        self.assert_soup('<p id="pi&ntilde;ata"></p>', expect)

    def test_entities_in_text_converted_to_unicode(self):
        expect = '<p>pi\N{LATIN SMALL LETTER N WITH TILDE}ata</p>'
        self.assert_soup("<p>pi&#241;ata</p>", expect)
        self.assert_soup("<p>pi&#xf1;ata</p>", expect)
        self.assert_soup("<p>pi&#Xf1;ata</p>", expect)
        self.assert_soup("<p>pi&ntilde;ata</p>", expect)

    def test_quot_entity_converted_to_quotation_mark(self):
        self.assert_soup("<p>I said &quot;good day!&quot;</p>",
                              '<p>I said "good day!"</p>')

    def test_out_of_range_entity(self):
        expect = "\N{REPLACEMENT CHARACTER}"
        self.assert_soup("&#10000000000000;", expect)
        self.assert_soup("&#x10000000000000;", expect)
        self.assert_soup("&#1000000000;", expect)
       
    def test_multipart_strings(self):
        "Mostly to prevent a recurrence of a bug in the html5lib treebuilder."
        soup = self.soup("<html><h2>\nfoo</h2><p></p></html>")
        assert "p" == soup.h2.string.next_element.name
        assert "p" == soup.p.name
        self.assertConnectedness(soup)

    def test_empty_element_tags(self):
        """Verify consistent handling of empty-element tags,
        no matter how they come in through the markup.
        """
        self.assert_soup('<br/><br/><br/>', "<br/><br/><br/>")
        self.assert_soup('<br /><br /><br />', "<br/><br/><br/>")
        
    def test_head_tag_between_head_and_body(self):
        "Prevent recurrence of a bug in the html5lib treebuilder."
        content = """<html><head></head>
  <link></link>
  <body>foo</body>
</html>
"""
        soup = self.soup(content)
        assert soup.html.body is not None
        self.assertConnectedness(soup)

    def test_multiple_copies_of_a_tag(self):
        "Prevent recurrence of a bug in the html5lib treebuilder."
        content = """<!DOCTYPE html>
<html>
 <body>
   <article id="a" >
   <div><a href="1"></div>
   <footer>
     <a href="2"></a>
   </footer>
  </article>
  </body>
</html>
"""
        soup = self.soup(content)
        self.assertConnectedness(soup.article)

    def test_basic_namespaces(self):
        """Parsers don't need to *understand* namespaces, but at the
        very least they should not choke on namespaces or lose
        data."""

        markup = b'<html xmlns="http://www.w3.org/1999/xhtml" xmlns:mathml="http://www.w3.org/1998/Math/MathML" xmlns:svg="http://www.w3.org/2000/svg"><head></head><body><mathml:msqrt>4</mathml:msqrt><b svg:fill="red"></b></body></html>'
        soup = self.soup(markup)
        assert markup == soup.encode()
        html = soup.html
        assert 'http://www.w3.org/1999/xhtml' == soup.html['xmlns']
        assert 'http://www.w3.org/1998/Math/MathML' == soup.html['xmlns:mathml']
        assert 'http://www.w3.org/2000/svg' == soup.html['xmlns:svg']

    def test_multivalued_attribute_value_becomes_list(self):
        markup = b'<a class="foo bar">'
        soup = self.soup(markup)
        assert ['foo', 'bar'] == soup.a['class']
        
    #
    # Generally speaking, tests below this point are more tests of
    # Beautiful Soup than tests of the tree builders. But parsers are
    # weird, so we run these tests separately for every tree builder
    # to detect any differences between them.
    #

    def test_can_parse_unicode_document(self):
        # A seemingly innocuous document... but it's in Unicode! And
        # it contains characters that can't be represented in the
        # encoding found in the  declaration! The horror!
        markup = '<html><head><meta encoding="euc-jp"></head><body>Sacr\N{LATIN SMALL LETTER E WITH ACUTE} bleu!</body>'
        soup = self.soup(markup)
        assert 'Sacr\xe9 bleu!' == soup.body.string

    def test_soupstrainer(self):
        """Parsers should be able to work with SoupStrainers."""
        strainer = SoupStrainer("b")
        soup = self.soup("A <b>bold</b> <meta/> <i>statement</i>",
                         parse_only=strainer)
        assert soup.decode() == "<b>bold</b>"

    def test_single_quote_attribute_values_become_double_quotes(self):
        self.assert_soup("<foo attr='bar'></foo>",
                              '<foo attr="bar"></foo>')

    def test_attribute_values_with_nested_quotes_are_left_alone(self):
        text = """<foo attr='bar "brawls" happen'>a</foo>"""
        self.assert_soup(text)

    def test_attribute_values_with_double_nested_quotes_get_quoted(self):
        text = """<foo attr='bar "brawls" happen'>a</foo>"""
        soup = self.soup(text)
        soup.foo['attr'] = 'Brawls happen at "Bob\'s Bar"'
        self.assert_soup(
            soup.foo.decode(),
            """<foo attr="Brawls happen at &quot;Bob\'s Bar&quot;">a</foo>""")

    def test_ampersand_in_attribute_value_gets_escaped(self):
        self.assert_soup('<this is="really messed up & stuff"></this>',
                              '<this is="really messed up &amp; stuff"></this>')

        self.assert_soup(
            '<a href="http://example.org?a=1&b=2;3">foo</a>',
            '<a href="http://example.org?a=1&amp;b=2;3">foo</a>')

    def test_escaped_ampersand_in_attribute_value_is_left_alone(self):
        self.assert_soup('<a href="http://example.org?a=1&amp;b=2;3"></a>')

    def test_entities_in_strings_converted_during_parsing(self):
        # Both XML and HTML entities are converted to Unicode characters
        # during parsing.
        text = "<p>&lt;&lt;sacr&eacute;&#32;bleu!&gt;&gt;</p>"
        expected = "<p>&lt;&lt;sacr\N{LATIN SMALL LETTER E WITH ACUTE} bleu!&gt;&gt;</p>"
        self.assert_soup(text, expected)

    def test_smart_quotes_converted_on_the_way_in(self):
        # Microsoft smart quotes are converted to Unicode characters during
        # parsing.
        quote = b"<p>\x91Foo\x92</p>"
        soup = self.soup(quote)
        assert soup.p.string == "\N{LEFT SINGLE QUOTATION MARK}Foo\N{RIGHT SINGLE QUOTATION MARK}"

    def test_non_breaking_spaces_converted_on_the_way_in(self):
        soup = self.soup("<a>&nbsp;&nbsp;</a>")
        assert soup.a.string == "\N{NO-BREAK SPACE}" * 2

    def test_entities_converted_on_the_way_out(self):
        text = "<p>&lt;&lt;sacr&eacute;&#32;bleu!&gt;&gt;</p>"
        expected = "<p>&lt;&lt;sacr\N{LATIN SMALL LETTER E WITH ACUTE} bleu!&gt;&gt;</p>".encode("utf-8")
        soup = self.soup(text)
        assert soup.p.encode("utf-8") == expected

    def test_real_iso_8859_document(self):
        # Smoke test of interrelated functionality, using an
        # easy-to-understand document.

        # Here it is in Unicode. Note that it claims to be in ISO-8859-1.
        unicode_html = '<html><head><meta content="text/html; charset=ISO-8859-1" http-equiv="Content-type"/></head><body><p>Sacr\N{LATIN SMALL LETTER E WITH ACUTE} bleu!</p></body></html>'

        # That's because we're going to encode it into ISO-8859-1,
        # and use that to test.
        iso_latin_html = unicode_html.encode("iso-8859-1")

        # Parse the ISO-8859-1 HTML.
        soup = self.soup(iso_latin_html)

        # Encode it to UTF-8.
        result = soup.encode("utf-8")

        # What do we expect the result to look like? Well, it would
        # look like unicode_html, except that the META tag would say
        # UTF-8 instead of ISO-8859-1.
        expected = unicode_html.replace("ISO-8859-1", "utf-8")

        # And, of course, it would be in UTF-8, not Unicode.
        expected = expected.encode("utf-8")

        # Ta-da!
        assert result == expected

    def test_real_shift_jis_document(self):
        # Smoke test to make sure the parser can handle a document in
        # Shift-JIS encoding, without choking.
        shift_jis_html = (
            b'<html><head></head><body><pre>'
            b'\x82\xb1\x82\xea\x82\xcdShift-JIS\x82\xc5\x83R\x81[\x83f'
            b'\x83B\x83\x93\x83O\x82\xb3\x82\xea\x82\xbd\x93\xfa\x96{\x8c'
            b'\xea\x82\xcc\x83t\x83@\x83C\x83\x8b\x82\xc5\x82\xb7\x81B'
            b'</pre></body></html>')
        unicode_html = shift_jis_html.decode("shift-jis")
        soup = self.soup(unicode_html)

        # Make sure the parse tree is correctly encoded to various
        # encodings.
        assert soup.encode("utf-8") == unicode_html.encode("utf-8")
        assert soup.encode("euc_jp") == unicode_html.encode("euc_jp")

    def test_real_hebrew_document(self):
        # A real-world test to make sure we can convert ISO-8859-9 (a
        # Hebrew encoding) to UTF-8.
        hebrew_document = b'<html><head><title>Hebrew (ISO 8859-8) in Visual Directionality</title></head><body><h1>Hebrew (ISO 8859-8) in Visual Directionality</h1>\xed\xe5\xec\xf9</body></html>'
        soup = self.soup(
            hebrew_document, from_encoding="iso8859-8")
        # Some tree builders call it iso8859-8, others call it iso-8859-9.
        # That's not a difference we really care about.
        assert soup.original_encoding in ('iso8859-8', 'iso-8859-8')
        assert soup.encode('utf-8') == (
            hebrew_document.decode("iso8859-8").encode("utf-8")
        )

    def test_meta_tag_reflects_current_encoding(self):
        # Here's the <meta> tag saying that a document is
        # encoded in Shift-JIS.
        meta_tag = ('<meta content="text/html; charset=x-sjis" '
                    'http-equiv="Content-type"/>')

        # Here's a document incorporating that meta tag.
        shift_jis_html = (
            '<html><head>\n%s\n'
            '<meta http-equiv="Content-language" content="ja"/>'
            '</head><body>Shift-JIS markup goes here.') % meta_tag
        soup = self.soup(shift_jis_html)

        # Parse the document, and the charset is seemingly unaffected.
        parsed_meta = soup.find('meta', {'http-equiv': 'Content-type'})
        content = parsed_meta['content']
        assert 'text/html; charset=x-sjis' == content

        # But that value is actually a ContentMetaAttributeValue object.
        assert isinstance(content, ContentMetaAttributeValue)

        # And it will take on a value that reflects its current
        # encoding.
        assert 'text/html; charset=utf8' == content.encode("utf8")

        # For the rest of the story, see TestSubstitutions in
        # test_tree.py.

    def test_html5_style_meta_tag_reflects_current_encoding(self):
        # Here's the <meta> tag saying that a document is
        # encoded in Shift-JIS.
        meta_tag = ('<meta id="encoding" charset="x-sjis" />')

        # Here's a document incorporating that meta tag.
        shift_jis_html = (
            '<html><head>\n%s\n'
            '<meta http-equiv="Content-language" content="ja"/>'
            '</head><body>Shift-JIS markup goes here.') % meta_tag
        soup = self.soup(shift_jis_html)

        # Parse the document, and the charset is seemingly unaffected.
        parsed_meta = soup.find('meta', id="encoding")
        charset = parsed_meta['charset']
        assert 'x-sjis' == charset

        # But that value is actually a CharsetMetaAttributeValue object.
        assert isinstance(charset, CharsetMetaAttributeValue)

        # And it will take on a value that reflects its current
        # encoding.
        assert 'utf8' == charset.encode("utf8")

    def test_python_specific_encodings_not_used_in_charset(self):
        # You can encode an HTML document using a Python-specific
        # encoding, but that encoding won't be mentioned _inside_ the
        # resulting document. Instead, the document will appear to
        # have no encoding.
        for markup in [
            b'<meta charset="utf8"></head>'
            b'<meta id="encoding" charset="utf-8" />'
        ]:
            soup = self.soup(markup)
            for encoding in PYTHON_SPECIFIC_ENCODINGS:
                if encoding in (
                    'idna', 'mbcs', 'oem', 'undefined',
                    'string_escape', 'string-escape'
                ):
                    # For one reason or another, these will raise an
                    # exception if we actually try to use them, so don't
                    # bother.
                    continue
                encoded = soup.encode(encoding)
                assert b'meta charset=""' in encoded
                assert encoding.encode("ascii") not in encoded
        
    def test_tag_with_no_attributes_can_have_attributes_added(self):
        data = self.soup("<a>text</a>")
        data.a['foo'] = 'bar'
        assert '<a foo="bar">text</a>' == data.a.decode()

    def test_closing_tag_with_no_opening_tag(self):
        # Without BeautifulSoup.open_tag_counter, the </span> tag will
        # cause _popToTag to be called over and over again as we look
        # for a <span> tag that wasn't there. The result is that 'text2'
        # will show up outside the body of the document.
        soup = self.soup("<body><div><p>text1</p></span>text2</div></body>")
        assert "<body><div><p>text1</p>text2</div></body>" == soup.body.decode()
        
    def test_worst_case(self):
        """Test the worst case (currently) for linking issues."""

        soup = self.soup(BAD_DOCUMENT)
        self.linkage_validator(soup)


class XMLTreeBuilderSmokeTest(TreeBuilderSmokeTest):

    def test_pickle_and_unpickle_identity(self):
        # Pickling a tree, then unpickling it, yields a tree identical
        # to the original.
        tree = self.soup("<a><b>foo</a>")
        dumped = pickle.dumps(tree, 2)
        loaded = pickle.loads(dumped)
        assert loaded.__class__ == BeautifulSoup
        assert loaded.decode() == tree.decode()

    def test_docstring_generated(self):
        soup = self.soup("<root/>")
        assert soup.encode() == b'<?xml version="1.0" encoding="utf-8"?>\n<root/>'

    def test_xml_declaration(self):
        markup = b"""<?xml version="1.0" encoding="utf8"?>\n<foo/>"""
        soup = self.soup(markup)
        assert markup == soup.encode("utf8")

    def test_python_specific_encodings_not_used_in_xml_declaration(self):
        # You can encode an XML document using a Python-specific
        # encoding, but that encoding won't be mentioned _inside_ the
        # resulting document.
        markup = b"""<?xml version="1.0"?>\n<foo/>"""
        soup = self.soup(markup)
        for encoding in PYTHON_SPECIFIC_ENCODINGS:
            if encoding in (
                'idna', 'mbcs', 'oem', 'undefined',
                'string_escape', 'string-escape'
            ):
                # For one reason or another, these will raise an
                # exception if we actually try to use them, so don't
                # bother.
                continue
            encoded = soup.encode(encoding)
            assert b'<?xml version="1.0"?>' in encoded
            assert encoding.encode("ascii") not in encoded

    def test_processing_instruction(self):
        markup = b"""<?xml version="1.0" encoding="utf8"?>\n<?PITarget PIContent?>"""
        soup = self.soup(markup)
        assert markup == soup.encode("utf8")

    def test_real_xhtml_document(self):
        """A real XHTML document should come out *exactly* the same as it went in."""
        markup = b"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN">
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Hello.</title></head>
<body>Goodbye.</body>
</html>"""
        soup = self.soup(markup)
        assert soup.encode("utf-8") == markup
       
    def test_nested_namespaces(self):
        doc = b"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">
<parent xmlns="http://ns1/">
<child xmlns="http://ns2/" xmlns:ns3="http://ns3/">
<grandchild ns3:attr="value" xmlns="http://ns4/"/>
</child>
</parent>"""
        soup = self.soup(doc)
        assert doc == soup.encode()
        
    def test_formatter_processes_script_tag_for_xml_documents(self):
        doc = """
  <script type="text/javascript">
  </script>
"""
        soup = BeautifulSoup(doc, "lxml-xml")
        # lxml would have stripped this while parsing, but we can add
        # it later.
        soup.script.string = 'console.log("< < hey > > ");'
        encoded = soup.encode()
        assert b"&lt; &lt; hey &gt; &gt;" in encoded

    def test_can_parse_unicode_document(self):
        markup = '<?xml version="1.0" encoding="euc-jp"><root>Sacr\N{LATIN SMALL LETTER E WITH ACUTE} bleu!</root>'
        soup = self.soup(markup)
        assert 'Sacr\xe9 bleu!' == soup.root.string

    def test_can_parse_unicode_document_begining_with_bom(self):
        markup = '\N{BYTE ORDER MARK}<?xml version="1.0" encoding="euc-jp"><root>Sacr\N{LATIN SMALL LETTER E WITH ACUTE} bleu!</root>'
        soup = self.soup(markup)
        assert 'Sacr\xe9 bleu!' == soup.root.string
        
    def test_popping_namespaced_tag(self):
        markup = '<rss xmlns:dc="foo"><dc:creator>b</dc:creator><dc:date>2012-07-02T20:33:42Z</dc:date><dc:rights>c</dc:rights><image>d</image></rss>'
        soup = self.soup(markup)
        assert str(soup.rss) == markup

    def test_docstring_includes_correct_encoding(self):
        soup = self.soup("<root/>")
        assert soup.encode("latin1") == b'<?xml version="1.0" encoding="latin1"?>\n<root/>'

    def test_large_xml_document(self):
        """A large XML document should come out the same as it went in."""
        markup = (b'<?xml version="1.0" encoding="utf-8"?>\n<root>'
                  + b'0' * (2**12)
                  + b'</root>')
        soup = self.soup(markup)
        assert soup.encode("utf-8") == markup

    def test_tags_are_empty_element_if_and_only_if_they_are_empty(self):
        self.assert_soup("<p>", "<p/>")
        self.assert_soup("<p>foo</p>")

    def test_namespaces_are_preserved(self):
        markup = '<root xmlns:a="http://example.com/" xmlns:b="http://example.net/"><a:foo>This tag is in the a namespace</a:foo><b:foo>This tag is in the b namespace</b:foo></root>'
        soup = self.soup(markup)
        root = soup.root
        assert "http://example.com/" == root['xmlns:a']
        assert "http://example.net/" == root['xmlns:b']

    def test_closing_namespaced_tag(self):
        markup = '<p xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:date>20010504</dc:date></p>'
        soup = self.soup(markup)
        assert str(soup.p) == markup

    def test_namespaced_attributes(self):
        markup = '<foo xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><bar xsi:schemaLocation="http://www.example.com"/></foo>'
        soup = self.soup(markup)
        assert str(soup.foo) == markup

    def test_namespaced_attributes_xml_namespace(self):
        markup = '<foo xml:lang="fr">bar</foo>'
        soup = self.soup(markup)
        assert str(soup.foo) == markup

    def test_find_by_prefixed_name(self):
        doc = """<?xml version="1.0" encoding="utf-8"?>
<Document xmlns="http://example.com/ns0"
    xmlns:ns1="http://example.com/ns1"
    xmlns:ns2="http://example.com/ns2"
    <ns1:tag>foo</ns1:tag>
    <ns1:tag>bar</ns1:tag>
    <ns2:tag key="value">baz</ns2:tag>
</Document>
"""
        soup = self.soup(doc)

        # There are three <tag> tags.
        assert 3 == len(soup.find_all('tag'))

        # But two of them are ns1:tag and one of them is ns2:tag.
        assert 2 == len(soup.find_all('ns1:tag'))
        assert 1 == len(soup.find_all('ns2:tag'))
        
        assert 1, len(soup.find_all('ns2:tag', key='value'))
        assert 3, len(soup.find_all(['ns1:tag', 'ns2:tag']))
        
    def test_copy_tag_preserves_namespace(self):
        xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://example.com/ns0"/>"""
    
        soup = self.soup(xml)
        tag = soup.document
        duplicate = copy.copy(tag)

        # The two tags have the same namespace prefix.
        assert tag.prefix == duplicate.prefix

    def test_worst_case(self):
        """Test the worst case (currently) for linking issues."""

        soup = self.soup(BAD_DOCUMENT)
        self.linkage_validator(soup)


class HTML5TreeBuilderSmokeTest(HTMLTreeBuilderSmokeTest):
    """Smoke test for a tree builder that supports HTML5."""

    def test_real_xhtml_document(self):
        # Since XHTML is not HTML5, HTML5 parsers are not tested to handle
        # XHTML documents in any particular way.
        pass

    def test_html_tags_have_namespace(self):
        markup = "<a>"
        soup = self.soup(markup)
        assert "http://www.w3.org/1999/xhtml" == soup.a.namespace

    def test_svg_tags_have_namespace(self):
        markup = '<svg><circle/></svg>'
        soup = self.soup(markup)
        namespace = "http://www.w3.org/2000/svg"
        assert namespace == soup.svg.namespace
        assert namespace == soup.circle.namespace


    def test_mathml_tags_have_namespace(self):
        markup = '<math><msqrt>5</msqrt></math>'
        soup = self.soup(markup)
        namespace = 'http://www.w3.org/1998/Math/MathML'
        assert namespace == soup.math.namespace
        assert namespace == soup.msqrt.namespace

    def test_xml_declaration_becomes_comment(self):
        markup = '<?xml version="1.0" encoding="utf-8"?><html></html>'
        soup = self.soup(markup)
        assert isinstance(soup.contents[0], Comment)
        assert soup.contents[0] == '?xml version="1.0" encoding="utf-8"?'
        assert "html" == soup.contents[0].next_element.name
