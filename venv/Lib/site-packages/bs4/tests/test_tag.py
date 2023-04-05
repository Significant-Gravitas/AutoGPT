import warnings
from bs4.element import (
    Comment,
    NavigableString,
)
from . import SoupTest

class TestTag(SoupTest):
    """Test various methods of Tag which aren't so complicated they
    need their own classes.
    """

    def test__should_pretty_print(self):
        # Test the rules about when a tag should be pretty-printed.
        tag = self.soup("").new_tag("a_tag")

        # No list of whitespace-preserving tags -> pretty-print
        tag._preserve_whitespace_tags = None
        assert True == tag._should_pretty_print(0)

        # List exists but tag is not on the list -> pretty-print
        tag.preserve_whitespace_tags = ["some_other_tag"]
        assert True == tag._should_pretty_print(1)

        # Indent level is None -> don't pretty-print
        assert False == tag._should_pretty_print(None)
        
        # Tag is on the whitespace-preserving list -> don't pretty-print
        tag.preserve_whitespace_tags = ["some_other_tag", "a_tag"]
        assert False == tag._should_pretty_print(1)
    
    def test_len(self):
        """The length of a Tag is its number of children."""
        soup = self.soup("<top>1<b>2</b>3</top>")

        # The BeautifulSoup object itself contains one element: the
        # <top> tag.
        assert len(soup.contents) == 1
        assert len(soup) == 1

        # The <top> tag contains three elements: the text node "1", the
        # <b> tag, and the text node "3".
        assert len(soup.top) == 3
        assert len(soup.top.contents) == 3

    def test_member_access_invokes_find(self):
        """Accessing a Python member .foo invokes find('foo')"""
        soup = self.soup('<b><i></i></b>')
        assert soup.b == soup.find('b')
        assert soup.b.i == soup.find('b').find('i')
        assert soup.a == None

    def test_deprecated_member_access(self):
        soup = self.soup('<b><i></i></b>')
        with warnings.catch_warnings(record=True) as w:
            tag = soup.bTag
        assert soup.b == tag
        assert '.bTag is deprecated, use .find("b") instead. If you really were looking for a tag called bTag, use .find("bTag")' == str(w[0].message)

    def test_has_attr(self):
        """has_attr() checks for the presence of an attribute.

        Please note note: has_attr() is different from
        __in__. has_attr() checks the tag's attributes and __in__
        checks the tag's chidlren.
        """
        soup = self.soup("<foo attr='bar'>")
        assert soup.foo.has_attr('attr')
        assert not soup.foo.has_attr('attr2')

    def test_attributes_come_out_in_alphabetical_order(self):
        markup = '<b a="1" z="5" m="3" f="2" y="4"></b>'
        self.assertSoupEquals(markup, '<b a="1" f="2" m="3" y="4" z="5"></b>')

    def test_string(self):
        # A Tag that contains only a text node makes that node
        # available as .string.
        soup = self.soup("<b>foo</b>")
        assert soup.b.string == 'foo'

    def test_empty_tag_has_no_string(self):
        # A Tag with no children has no .stirng.
        soup = self.soup("<b></b>")
        assert soup.b.string == None

    def test_tag_with_multiple_children_has_no_string(self):
        # A Tag with no children has no .string.
        soup = self.soup("<a>foo<b></b><b></b></b>")
        assert soup.b.string == None

        soup = self.soup("<a>foo<b></b>bar</b>")
        assert soup.b.string == None

        # Even if all the children are strings, due to trickery,
        # it won't work--but this would be a good optimization.
        soup = self.soup("<a>foo</b>")
        soup.a.insert(1, "bar")
        assert soup.a.string == None

    def test_tag_with_recursive_string_has_string(self):
        # A Tag with a single child which has a .string inherits that
        # .string.
        soup = self.soup("<a><b>foo</b></a>")
        assert soup.a.string == "foo"
        assert soup.string == "foo"

    def test_lack_of_string(self):
        """Only a Tag containing a single text node has a .string."""
        soup = self.soup("<b>f<i>e</i>o</b>")
        assert soup.b.string is None

        soup = self.soup("<b></b>")
        assert soup.b.string is None

    def test_all_text(self):
        """Tag.text and Tag.get_text(sep=u"") -> all child text, concatenated"""
        soup = self.soup("<a>a<b>r</b>   <r> t </r></a>")
        assert soup.a.text == "ar  t "
        assert soup.a.get_text(strip=True) == "art"
        assert soup.a.get_text(",") == "a,r, , t "
        assert soup.a.get_text(",", strip=True) == "a,r,t"

    def test_get_text_ignores_special_string_containers(self):
        soup = self.soup("foo<!--IGNORE-->bar")
        assert soup.get_text() == "foobar"

        assert soup.get_text(types=(NavigableString, Comment)) == "fooIGNOREbar"
        assert soup.get_text(types=None) == "fooIGNOREbar"

        soup = self.soup("foo<style>CSS</style><script>Javascript</script>bar")
        assert soup.get_text() == "foobar"
        
    def test_all_strings_ignores_special_string_containers(self):
        soup = self.soup("foo<!--IGNORE-->bar")
        assert ['foo', 'bar'] == list(soup.strings)

        soup = self.soup("foo<style>CSS</style><script>Javascript</script>bar")
        assert ['foo', 'bar'] == list(soup.strings)

    def test_string_methods_inside_special_string_container_tags(self):
        # Strings inside tags like <script> are generally ignored by
        # methods like get_text, because they're not what humans
        # consider 'text'. But if you call get_text on the <script>
        # tag itself, those strings _are_ considered to be 'text',
        # because there's nothing else you might be looking for.
        
        style = self.soup("<div>a<style>Some CSS</style></div>")
        template = self.soup("<div>a<template><p>Templated <b>text</b>.</p><!--With a comment.--></template></div>")
        script = self.soup("<div>a<script><!--a comment-->Some text</script></div>")
        
        assert style.div.get_text() == "a"
        assert list(style.div.strings) == ["a"]
        assert style.div.style.get_text() == "Some CSS"
        assert list(style.div.style.strings) == ['Some CSS']
        
        # The comment is not picked up here. That's because it was
        # parsed into a Comment object, which is not considered
        # interesting by template.strings.
        assert template.div.get_text() == "a"
        assert list(template.div.strings) == ["a"]
        assert template.div.template.get_text() == "Templated text."
        assert list(template.div.template.strings) == ["Templated ", "text", "."]

        # The comment is included here, because it didn't get parsed
        # into a Comment object--it's part of the Script string.
        assert script.div.get_text() == "a"
        assert list(script.div.strings) == ["a"]
        assert script.div.script.get_text() == "<!--a comment-->Some text"
        assert list(script.div.script.strings) == ['<!--a comment-->Some text']


class TestMultiValuedAttributes(SoupTest):
    """Test the behavior of multi-valued attributes like 'class'.

    The values of such attributes are always presented as lists.
    """

    def test_single_value_becomes_list(self):
        soup = self.soup("<a class='foo'>")
        assert ["foo"] ==soup.a['class']

    def test_multiple_values_becomes_list(self):
        soup = self.soup("<a class='foo bar'>")
        assert ["foo", "bar"] == soup.a['class']

    def test_multiple_values_separated_by_weird_whitespace(self):
        soup = self.soup("<a class='foo\tbar\nbaz'>")
        assert ["foo", "bar", "baz"] ==soup.a['class']

    def test_attributes_joined_into_string_on_output(self):
        soup = self.soup("<a class='foo\tbar'>")
        assert b'<a class="foo bar"></a>' == soup.a.encode()

    def test_get_attribute_list(self):
        soup = self.soup("<a id='abc def'>")
        assert ['abc def'] == soup.a.get_attribute_list('id')
        
    def test_accept_charset(self):
        soup = self.soup('<form accept-charset="ISO-8859-1 UTF-8">')
        assert ['ISO-8859-1', 'UTF-8'] == soup.form['accept-charset']

    def test_cdata_attribute_applying_only_to_one_tag(self):
        data = '<a accept-charset="ISO-8859-1 UTF-8"></a>'
        soup = self.soup(data)
        # We saw in another test that accept-charset is a cdata-list
        # attribute for the <form> tag. But it's not a cdata-list
        # attribute for any other tag.
        assert 'ISO-8859-1 UTF-8' == soup.a['accept-charset']

    def test_customization(self):
        # It's possible to change which attributes of which tags
        # are treated as multi-valued attributes.
        #
        # Here, 'id' is a multi-valued attribute and 'class' is not.
        #
        # TODO: This code is in the builder and should be tested there.
        soup = self.soup(
            '<a class="foo" id="bar">', multi_valued_attributes={ '*' : 'id' }
        )
        assert soup.a['class'] == 'foo'
        assert soup.a['id'] == ['bar']
