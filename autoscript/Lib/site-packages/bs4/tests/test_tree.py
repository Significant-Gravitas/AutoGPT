# -*- coding: utf-8 -*-
"""Tests for Beautiful Soup's tree traversal methods.

The tree traversal methods are the main advantage of using Beautiful
Soup over just using a parser.

Different parsers will build different Beautiful Soup trees given the
same markup, but all Beautiful Soup trees can be traversed with the
methods tested here.
"""

from pdb import set_trace
import pytest
import re
import warnings
from bs4 import BeautifulSoup
from bs4.builder import (
    builder_registry,
    HTMLParserTreeBuilder,
)
from bs4.element import (
    CData,
    Comment,
    Declaration,
    Doctype,
    Formatter,
    NavigableString,
    Script,
    SoupStrainer,
    Stylesheet,
    Tag,
    TemplateString,
)
from . import (
    SoupTest,
)

class TestFind(SoupTest):
    """Basic tests of the find() method.

    find() just calls find_all() with limit=1, so it's not tested all
    that thouroughly here.
    """

    def test_find_tag(self):
        soup = self.soup("<a>1</a><b>2</b><a>3</a><b>4</b>")
        assert soup.find("b").string == "2"

    def test_unicode_text_find(self):
        soup = self.soup('<h1>Räksmörgås</h1>')
        assert soup.find(string='Räksmörgås') == 'Räksmörgås'

    def test_unicode_attribute_find(self):
        soup = self.soup('<h1 id="Räksmörgås">here it is</h1>')
        str(soup)
        assert "here it is" == soup.find(id='Räksmörgås').text


    def test_find_everything(self):
        """Test an optimization that finds all tags."""
        soup = self.soup("<a>foo</a><b>bar</b>")
        assert 2 == len(soup.find_all())

    def test_find_everything_with_name(self):
        """Test an optimization that finds all tags with a given name."""
        soup = self.soup("<a>foo</a><b>bar</b><a>baz</a>")
        assert 2 == len(soup.find_all('a'))

class TestFindAll(SoupTest):
    """Basic tests of the find_all() method."""

    def test_find_all_text_nodes(self):
        """You can search the tree for text nodes."""
        soup = self.soup("<html>Foo<b>bar</b>\xbb</html>")
        # Exact match.
        assert soup.find_all(string="bar") == ["bar"]

       
        # Match any of a number of strings.
        assert soup.find_all(string=["Foo", "bar"]) == ["Foo", "bar"]
        # Match a regular expression.
        assert soup.find_all(string=re.compile('.*')) == ["Foo", "bar", '\xbb']
        # Match anything.
        assert soup.find_all(string=True) == ["Foo", "bar", '\xbb']

    def test_find_all_limit(self):
        """You can limit the number of items returned by find_all."""
        soup = self.soup("<a>1</a><a>2</a><a>3</a><a>4</a><a>5</a>")
        self.assert_selects(soup.find_all('a', limit=3), ["1", "2", "3"])
        self.assert_selects(soup.find_all('a', limit=1), ["1"])
        self.assert_selects(
            soup.find_all('a', limit=10), ["1", "2", "3", "4", "5"])

        # A limit of 0 means no limit.
        self.assert_selects(
            soup.find_all('a', limit=0), ["1", "2", "3", "4", "5"])

    def test_calling_a_tag_is_calling_findall(self):
        soup = self.soup("<a>1</a><b>2<a id='foo'>3</a></b>")
        self.assert_selects(soup('a', limit=1), ["1"])
        self.assert_selects(soup.b(id="foo"), ["3"])

    def test_find_all_with_self_referential_data_structure_does_not_cause_infinite_recursion(self):
        soup = self.soup("<a></a>")
        # Create a self-referential list.
        l = []
        l.append(l)

        # Without special code in _normalize_search_value, this would cause infinite
        # recursion.
        assert [] == soup.find_all(l)

    def test_find_all_resultset(self):
        """All find_all calls return a ResultSet"""
        soup = self.soup("<a></a>")
        result = soup.find_all("a")
        assert hasattr(result, "source")

        result = soup.find_all(True)
        assert hasattr(result, "source")

        result = soup.find_all(string="foo")
        assert hasattr(result, "source")


class TestFindAllBasicNamespaces(SoupTest):

    def test_find_by_namespaced_name(self):
        soup = self.soup('<mathml:msqrt>4</mathml:msqrt><a svg:fill="red">')
        assert "4" == soup.find("mathml:msqrt").string
        assert "a" == soup.find(attrs= { "svg:fill" : "red" }).name


class TestFindAllByName(SoupTest):
    """Test ways of finding tags by tag name."""

    def setup_method(self):
        self.tree =  self.soup("""<a>First tag.</a>
                                  <b>Second tag.</b>
                                  <c>Third <a>Nested tag.</a> tag.</c>""")

    def test_find_all_by_tag_name(self):
        # Find all the <a> tags.
        self.assert_selects(
            self.tree.find_all('a'), ['First tag.', 'Nested tag.'])

    def test_find_all_by_name_and_text(self):
        self.assert_selects(
            self.tree.find_all('a', string='First tag.'), ['First tag.'])

        self.assert_selects(
            self.tree.find_all('a', string=True), ['First tag.', 'Nested tag.'])

        self.assert_selects(
            self.tree.find_all('a', string=re.compile("tag")),
            ['First tag.', 'Nested tag.'])


    def test_find_all_on_non_root_element(self):
        # You can call find_all on any node, not just the root.
        self.assert_selects(self.tree.c.find_all('a'), ['Nested tag.'])

    def test_calling_element_invokes_find_all(self):
        self.assert_selects(self.tree('a'), ['First tag.', 'Nested tag.'])

    def test_find_all_by_tag_strainer(self):
        self.assert_selects(
            self.tree.find_all(SoupStrainer('a')),
            ['First tag.', 'Nested tag.'])

    def test_find_all_by_tag_names(self):
        self.assert_selects(
            self.tree.find_all(['a', 'b']),
            ['First tag.', 'Second tag.', 'Nested tag.'])

    def test_find_all_by_tag_dict(self):
        self.assert_selects(
            self.tree.find_all({'a' : True, 'b' : True}),
            ['First tag.', 'Second tag.', 'Nested tag.'])

    def test_find_all_by_tag_re(self):
        self.assert_selects(
            self.tree.find_all(re.compile('^[ab]$')),
            ['First tag.', 'Second tag.', 'Nested tag.'])

    def test_find_all_with_tags_matching_method(self):
        # You can define an oracle method that determines whether
        # a tag matches the search.
        def id_matches_name(tag):
            return tag.name == tag.get('id')

        tree = self.soup("""<a id="a">Match 1.</a>
                            <a id="1">Does not match.</a>
                            <b id="b">Match 2.</a>""")

        self.assert_selects(
            tree.find_all(id_matches_name), ["Match 1.", "Match 2."])

    def test_find_with_multi_valued_attribute(self):
        soup = self.soup(
            "<div class='a b'>1</div><div class='a c'>2</div><div class='a d'>3</div>"
        )
        r1 = soup.find('div', 'a d');
        r2 = soup.find('div', re.compile(r'a d'));
        r3, r4 = soup.find_all('div', ['a b', 'a d']);
        assert '3' == r1.string
        assert '3' == r2.string
        assert '1' == r3.string
        assert '3' == r4.string

        
class TestFindAllByAttribute(SoupTest):

    def test_find_all_by_attribute_name(self):
        # You can pass in keyword arguments to find_all to search by
        # attribute.
        tree = self.soup("""
                         <a id="first">Matching a.</a>
                         <a id="second">
                          Non-matching <b id="first">Matching b.</b>a.
                         </a>""")
        self.assert_selects(tree.find_all(id='first'),
                           ["Matching a.", "Matching b."])

    def test_find_all_by_utf8_attribute_value(self):
        peace = "םולש".encode("utf8")
        data = '<a title="םולש"></a>'.encode("utf8")
        soup = self.soup(data)
        assert [soup.a] == soup.find_all(title=peace)
        assert [soup.a] == soup.find_all(title=peace.decode("utf8"))
        assert [soup.a], soup.find_all(title=[peace, "something else"])

    def test_find_all_by_attribute_dict(self):
        # You can pass in a dictionary as the argument 'attrs'. This
        # lets you search for attributes like 'name' (a fixed argument
        # to find_all) and 'class' (a reserved word in Python.)
        tree = self.soup("""
                         <a name="name1" class="class1">Name match.</a>
                         <a name="name2" class="class2">Class match.</a>
                         <a name="name3" class="class3">Non-match.</a>
                         <name1>A tag called 'name1'.</name1>
                         """)

        # This doesn't do what you want.
        self.assert_selects(tree.find_all(name='name1'),
                           ["A tag called 'name1'."])
        # This does what you want.
        self.assert_selects(tree.find_all(attrs={'name' : 'name1'}),
                           ["Name match."])

        self.assert_selects(tree.find_all(attrs={'class' : 'class2'}),
                           ["Class match."])

    def test_find_all_by_class(self):
        tree = self.soup("""
                         <a class="1">Class 1.</a>
                         <a class="2">Class 2.</a>
                         <b class="1">Class 1.</b>
                         <c class="3 4">Class 3 and 4.</c>
                         """)

        # Passing in the class_ keyword argument will search against
        # the 'class' attribute.
        self.assert_selects(tree.find_all('a', class_='1'), ['Class 1.'])
        self.assert_selects(tree.find_all('c', class_='3'), ['Class 3 and 4.'])
        self.assert_selects(tree.find_all('c', class_='4'), ['Class 3 and 4.'])

        # Passing in a string to 'attrs' will also search the CSS class.
        self.assert_selects(tree.find_all('a', '1'), ['Class 1.'])
        self.assert_selects(tree.find_all(attrs='1'), ['Class 1.', 'Class 1.'])
        self.assert_selects(tree.find_all('c', '3'), ['Class 3 and 4.'])
        self.assert_selects(tree.find_all('c', '4'), ['Class 3 and 4.'])

    def test_find_by_class_when_multiple_classes_present(self):
        tree = self.soup("<gar class='foo bar'>Found it</gar>")

        f = tree.find_all("gar", class_=re.compile("o"))
        self.assert_selects(f, ["Found it"])

        f = tree.find_all("gar", class_=re.compile("a"))
        self.assert_selects(f, ["Found it"])

        # If the search fails to match the individual strings "foo" and "bar",
        # it will be tried against the combined string "foo bar".
        f = tree.find_all("gar", class_=re.compile("o b"))
        self.assert_selects(f, ["Found it"])

    def test_find_all_with_non_dictionary_for_attrs_finds_by_class(self):
        soup = self.soup("<a class='bar'>Found it</a>")

        self.assert_selects(soup.find_all("a", re.compile("ba")), ["Found it"])

        def big_attribute_value(value):
            return len(value) > 3

        self.assert_selects(soup.find_all("a", big_attribute_value), [])

        def small_attribute_value(value):
            return len(value) <= 3

        self.assert_selects(
            soup.find_all("a", small_attribute_value), ["Found it"])

    def test_find_all_with_string_for_attrs_finds_multiple_classes(self):
        soup = self.soup('<a class="foo bar"></a><a class="foo"></a>')
        a, a2 = soup.find_all("a")
        assert [a, a2], soup.find_all("a", "foo")
        assert [a], soup.find_all("a", "bar")

        # If you specify the class as a string that contains a
        # space, only that specific value will be found.
        assert [a] == soup.find_all("a", class_="foo bar")
        assert [a] == soup.find_all("a", "foo bar")
        assert [] == soup.find_all("a", "bar foo")

    def test_find_all_by_attribute_soupstrainer(self):
        tree = self.soup("""
                         <a id="first">Match.</a>
                         <a id="second">Non-match.</a>""")

        strainer = SoupStrainer(attrs={'id' : 'first'})
        self.assert_selects(tree.find_all(strainer), ['Match.'])

    def test_find_all_with_missing_attribute(self):
        # You can pass in None as the value of an attribute to find_all.
        # This will match tags that do not have that attribute set.
        tree = self.soup("""<a id="1">ID present.</a>
                            <a>No ID present.</a>
                            <a id="">ID is empty.</a>""")
        self.assert_selects(tree.find_all('a', id=None), ["No ID present."])

    def test_find_all_with_defined_attribute(self):
        # You can pass in None as the value of an attribute to find_all.
        # This will match tags that have that attribute set to any value.
        tree = self.soup("""<a id="1">ID present.</a>
                            <a>No ID present.</a>
                            <a id="">ID is empty.</a>""")
        self.assert_selects(
            tree.find_all(id=True), ["ID present.", "ID is empty."])

    def test_find_all_with_numeric_attribute(self):
        # If you search for a number, it's treated as a string.
        tree = self.soup("""<a id=1>Unquoted attribute.</a>
                            <a id="1">Quoted attribute.</a>""")

        expected = ["Unquoted attribute.", "Quoted attribute."]
        self.assert_selects(tree.find_all(id=1), expected)
        self.assert_selects(tree.find_all(id="1"), expected)

    def test_find_all_with_list_attribute_values(self):
        # You can pass a list of attribute values instead of just one,
        # and you'll get tags that match any of the values.
        tree = self.soup("""<a id="1">1</a>
                            <a id="2">2</a>
                            <a id="3">3</a>
                            <a>No ID.</a>""")
        self.assert_selects(tree.find_all(id=["1", "3", "4"]),
                           ["1", "3"])

    def test_find_all_with_regular_expression_attribute_value(self):
        # You can pass a regular expression as an attribute value, and
        # you'll get tags whose values for that attribute match the
        # regular expression.
        tree = self.soup("""<a id="a">One a.</a>
                            <a id="aa">Two as.</a>
                            <a id="ab">Mixed as and bs.</a>
                            <a id="b">One b.</a>
                            <a>No ID.</a>""")

        self.assert_selects(tree.find_all(id=re.compile("^a+$")),
                           ["One a.", "Two as."])

    def test_find_by_name_and_containing_string(self):
        soup = self.soup("<b>foo</b><b>bar</b><a>foo</a>")
        a = soup.a

        assert [a] == soup.find_all("a", string="foo")
        assert [] == soup.find_all("a", string="bar")

    def test_find_by_name_and_containing_string_when_string_is_buried(self):
        soup = self.soup("<a>foo</a><a><b><c>foo</c></b></a>")
        assert soup.find_all("a") == soup.find_all("a", string="foo")

    def test_find_by_attribute_and_containing_string(self):
        soup = self.soup('<b id="1">foo</b><a id="2">foo</a>')
        a = soup.a

        assert [a] == soup.find_all(id=2, string="foo")
        assert [] == soup.find_all(id=1, string="bar")


class TestSmooth(SoupTest):
    """Test Tag.smooth."""

    def test_smooth(self):
        soup = self.soup("<div>a</div>")
        div = soup.div
        div.append("b")
        div.append("c")
        div.append(Comment("Comment 1"))
        div.append(Comment("Comment 2"))
        div.append("d")
        builder = self.default_builder()
        span = Tag(soup, builder, 'span')
        span.append('1')
        span.append('2')
        div.append(span)

        # At this point the tree has a bunch of adjacent
        # NavigableStrings. This is normal, but it has no meaning in
        # terms of HTML, so we may want to smooth things out for
        # output.

        # Since the <span> tag has two children, its .string is None.
        assert None == div.span.string

        assert 7 == len(div.contents)
        div.smooth()
        assert 5 == len(div.contents)

        # The three strings at the beginning of div.contents have been
        # merged into on string.
        #
        assert 'abc' == div.contents[0]

        # The call is recursive -- the <span> tag was also smoothed.
        assert '12' == div.span.string

        # The two comments have _not_ been merged, even though
        # comments are strings. Merging comments would change the
        # meaning of the HTML.
        assert 'Comment 1' == div.contents[1]
        assert 'Comment 2' == div.contents[2]


class TestIndex(SoupTest):
    """Test Tag.index"""
    def test_index(self):
        tree = self.soup("""<div>
                            <a>Identical</a>
                            <b>Not identical</b>
                            <a>Identical</a>

                            <c><d>Identical with child</d></c>
                            <b>Also not identical</b>
                            <c><d>Identical with child</d></c>
                            </div>""")
        div = tree.div
        for i, element in enumerate(div.contents):
            assert i == div.index(element)
        with pytest.raises(ValueError):
            tree.index(1)


class TestParentOperations(SoupTest):
    """Test navigation and searching through an element's parents."""

    def setup_method(self):
        self.tree = self.soup('''<ul id="empty"></ul>
                                 <ul id="top">
                                  <ul id="middle">
                                   <ul id="bottom">
                                    <b>Start here</b>
                                   </ul>
                                  </ul>''')
        self.start = self.tree.b


    def test_parent(self):
        assert self.start.parent['id'] == 'bottom'
        assert self.start.parent.parent['id'] == 'middle'
        assert self.start.parent.parent.parent['id'] == 'top'

    def test_parent_of_top_tag_is_soup_object(self):
        top_tag = self.tree.contents[0]
        assert top_tag.parent == self.tree

    def test_soup_object_has_no_parent(self):
        assert None == self.tree.parent

    def test_find_parents(self):
        self.assert_selects_ids(
            self.start.find_parents('ul'), ['bottom', 'middle', 'top'])
        self.assert_selects_ids(
            self.start.find_parents('ul', id="middle"), ['middle'])

    def test_find_parent(self):
        assert self.start.find_parent('ul')['id'] == 'bottom'
        assert self.start.find_parent('ul', id='top')['id'] == 'top'

    def test_parent_of_text_element(self):
        text = self.tree.find(string="Start here")
        assert text.parent.name == 'b'

    def test_text_element_find_parent(self):
        text = self.tree.find(string="Start here")
        assert text.find_parent('ul')['id'] == 'bottom'

    def test_parent_generator(self):
        parents = [parent['id'] for parent in self.start.parents
                   if parent is not None and 'id' in parent.attrs]
        assert parents, ['bottom', 'middle' == 'top']


class ProximityTest(SoupTest):

    def setup_method(self):
        self.tree = self.soup(
            '<html id="start"><head></head><body><b id="1">One</b><b id="2">Two</b><b id="3">Three</b></body></html>')


class TestNextOperations(ProximityTest):

    def setup_method(self):
        super(TestNextOperations, self).setup_method()
        self.start = self.tree.b

    def test_next(self):
        assert self.start.next_element == "One"
        assert self.start.next_element.next_element['id'] == "2"

    def test_next_of_last_item_is_none(self):
        last = self.tree.find(string="Three")
        assert last.next_element == None

    def test_next_of_root_is_none(self):
        # The document root is outside the next/previous chain.
        assert self.tree.next_element == None

    def test_find_all_next(self):
        self.assert_selects(self.start.find_all_next('b'), ["Two", "Three"])
        self.start.find_all_next(id=3)
        self.assert_selects(self.start.find_all_next(id=3), ["Three"])

    def test_find_next(self):
        assert self.start.find_next('b')['id'] == '2'
        assert self.start.find_next(string="Three") == "Three"

    def test_find_next_for_text_element(self):
        text = self.tree.find(string="One")
        assert text.find_next("b").string == "Two"
        self.assert_selects(text.find_all_next("b"), ["Two", "Three"])

    def test_next_generator(self):
        start = self.tree.find(string="Two")
        successors = [node for node in start.next_elements]
        # There are two successors: the final <b> tag and its text contents.
        tag, contents = successors
        assert tag['id'] == '3'
        assert contents == "Three"

class TestPreviousOperations(ProximityTest):

    def setup_method(self):
        super(TestPreviousOperations, self).setup_method()
        self.end = self.tree.find(string="Three")

    def test_previous(self):
        assert self.end.previous_element['id'] == "3"
        assert self.end.previous_element.previous_element == "Two"

    def test_previous_of_first_item_is_none(self):
        first = self.tree.find('html')
        assert first.previous_element == None

    def test_previous_of_root_is_none(self):
        # The document root is outside the next/previous chain.
        assert self.tree.previous_element == None

    def test_find_all_previous(self):
        # The <b> tag containing the "Three" node is the predecessor
        # of the "Three" node itself, which is why "Three" shows up
        # here.
        self.assert_selects(
            self.end.find_all_previous('b'), ["Three", "Two", "One"])
        self.assert_selects(self.end.find_all_previous(id=1), ["One"])

    def test_find_previous(self):
        assert self.end.find_previous('b')['id'] == '3'
        assert self.end.find_previous(string="One") == "One"

    def test_find_previous_for_text_element(self):
        text = self.tree.find(string="Three")
        assert text.find_previous("b").string == "Three"
        self.assert_selects(
            text.find_all_previous("b"), ["Three", "Two", "One"])

    def test_previous_generator(self):
        start = self.tree.find(string="One")
        predecessors = [node for node in start.previous_elements]

        # There are four predecessors: the <b> tag containing "One"
        # the <body> tag, the <head> tag, and the <html> tag.
        b, body, head, html = predecessors
        assert b['id'] == '1'
        assert body.name == "body"
        assert head.name == "head"
        assert html.name == "html"


class SiblingTest(SoupTest):

    def setup_method(self):
        markup = '''<html>
                    <span id="1">
                     <span id="1.1"></span>
                    </span>
                    <span id="2">
                     <span id="2.1"></span>
                    </span>
                    <span id="3">
                     <span id="3.1"></span>
                    </span>
                    <span id="4"></span>
                    </html>'''
        # All that whitespace looks good but makes the tests more
        # difficult. Get rid of it.
        markup = re.compile(r"\n\s*").sub("", markup)
        self.tree = self.soup(markup)


class TestNextSibling(SiblingTest):

    def setup_method(self):
        super(TestNextSibling, self).setup_method()
        self.start = self.tree.find(id="1")

    def test_next_sibling_of_root_is_none(self):
        assert self.tree.next_sibling == None

    def test_next_sibling(self):
        assert self.start.next_sibling['id'] == '2'
        assert self.start.next_sibling.next_sibling['id'] == '3'

        # Note the difference between next_sibling and next_element.
        assert self.start.next_element['id'] == '1.1'

    def test_next_sibling_may_not_exist(self):
        assert self.tree.html.next_sibling == None

        nested_span = self.tree.find(id="1.1")
        assert nested_span.next_sibling == None

        last_span = self.tree.find(id="4")
        assert last_span.next_sibling == None

    def test_find_next_sibling(self):
        assert self.start.find_next_sibling('span')['id'] == '2'

    def test_next_siblings(self):
        self.assert_selects_ids(self.start.find_next_siblings("span"),
                              ['2', '3', '4'])

        self.assert_selects_ids(self.start.find_next_siblings(id='3'), ['3'])

    def test_next_sibling_for_text_element(self):
        soup = self.soup("Foo<b>bar</b>baz")
        start = soup.find(string="Foo")
        assert start.next_sibling.name == 'b'
        assert start.next_sibling.next_sibling == 'baz'

        self.assert_selects(start.find_next_siblings('b'), ['bar'])
        assert start.find_next_sibling(string="baz") == "baz"
        assert start.find_next_sibling(string="nonesuch") == None


class TestPreviousSibling(SiblingTest):

    def setup_method(self):
        super(TestPreviousSibling, self).setup_method()
        self.end = self.tree.find(id="4")

    def test_previous_sibling_of_root_is_none(self):
        assert self.tree.previous_sibling == None

    def test_previous_sibling(self):
        assert self.end.previous_sibling['id'] == '3'
        assert self.end.previous_sibling.previous_sibling['id'] == '2'

        # Note the difference between previous_sibling and previous_element.
        assert self.end.previous_element['id'] == '3.1'

    def test_previous_sibling_may_not_exist(self):
        assert self.tree.html.previous_sibling == None

        nested_span = self.tree.find(id="1.1")
        assert nested_span.previous_sibling == None

        first_span = self.tree.find(id="1")
        assert first_span.previous_sibling == None

    def test_find_previous_sibling(self):
        assert self.end.find_previous_sibling('span')['id'] == '3'

    def test_previous_siblings(self):
        self.assert_selects_ids(self.end.find_previous_siblings("span"),
                              ['3', '2', '1'])

        self.assert_selects_ids(self.end.find_previous_siblings(id='1'), ['1'])

    def test_previous_sibling_for_text_element(self):
        soup = self.soup("Foo<b>bar</b>baz")
        start = soup.find(string="baz")
        assert start.previous_sibling.name == 'b'
        assert start.previous_sibling.previous_sibling == 'Foo'

        self.assert_selects(start.find_previous_siblings('b'), ['bar'])
        assert start.find_previous_sibling(string="Foo") == "Foo"
        assert start.find_previous_sibling(string="nonesuch") == None


class TestTreeModification(SoupTest):

    def test_attribute_modification(self):
        soup = self.soup('<a id="1"></a>')
        soup.a['id'] = 2
        assert soup.decode() == self.document_for('<a id="2"></a>')
        del(soup.a['id'])
        assert soup.decode() == self.document_for('<a></a>')
        soup.a['id2'] = 'foo'
        assert soup.decode() == self.document_for('<a id2="foo"></a>')

    def test_new_tag_creation(self):
        builder = builder_registry.lookup('html')()
        soup = self.soup("<body></body>", builder=builder)
        a = Tag(soup, builder, 'a')
        ol = Tag(soup, builder, 'ol')
        a['href'] = 'http://foo.com/'
        soup.body.insert(0, a)
        soup.body.insert(1, ol)
        assert soup.body.encode() == b'<body><a href="http://foo.com/"></a><ol></ol></body>'

    def test_append_to_contents_moves_tag(self):
        doc = """<p id="1">Don't leave me <b>here</b>.</p>
                <p id="2">Don\'t leave!</p>"""
        soup = self.soup(doc)
        second_para = soup.find(id='2')
        bold = soup.b

        # Move the <b> tag to the end of the second paragraph.
        soup.find(id='2').append(soup.b)

        # The <b> tag is now a child of the second paragraph.
        assert bold.parent == second_para

        assert soup.decode() == self.document_for(
                '<p id="1">Don\'t leave me .</p>\n'
                '<p id="2">Don\'t leave!<b>here</b></p>'
        )

    def test_replace_with_returns_thing_that_was_replaced(self):
        text = "<a></a><b><c></c></b>"
        soup = self.soup(text)
        a = soup.a
        new_a = a.replace_with(soup.c)
        assert a == new_a

    def test_unwrap_returns_thing_that_was_replaced(self):
        text = "<a><b></b><c></c></a>"
        soup = self.soup(text)
        a = soup.a
        new_a = a.unwrap()
        assert a == new_a

    def test_replace_with_and_unwrap_give_useful_exception_when_tag_has_no_parent(self):
        soup = self.soup("<a><b>Foo</b></a><c>Bar</c>")
        a = soup.a
        a.extract()
        assert None == a.parent
        with pytest.raises(ValueError):
            a.unwrap()
        with pytest.raises(ValueError):
            a.replace_with(soup.c)

    def test_replace_tag_with_itself(self):
        text = "<a><b></b><c>Foo<d></d></c></a><a><e></e></a>"
        soup = self.soup(text)
        c = soup.c
        soup.c.replace_with(c)
        assert soup.decode() == self.document_for(text)

    def test_replace_tag_with_its_parent_raises_exception(self):
        text = "<a><b></b></a>"
        soup = self.soup(text)
        with pytest.raises(ValueError):
            soup.b.replace_with(soup.a)

    def test_insert_tag_into_itself_raises_exception(self):
        text = "<a><b></b></a>"
        soup = self.soup(text)
        with pytest.raises(ValueError):
            soup.a.insert(0, soup.a)

    def test_insert_beautifulsoup_object_inserts_children(self):
        """Inserting one BeautifulSoup object into another actually inserts all
        of its children -- you'll never combine BeautifulSoup objects.
        """
        soup = self.soup("<p>And now, a word:</p><p>And we're back.</p>")
        
        text = "<p>p2</p><p>p3</p>"
        to_insert = self.soup(text)
        soup.insert(1, to_insert)

        for i in soup.descendants:
            assert not isinstance(i, BeautifulSoup)
        
        p1, p2, p3, p4 = list(soup.children)
        assert "And now, a word:" == p1.string
        assert "p2" == p2.string
        assert "p3" == p3.string
        assert "And we're back." == p4.string
        
        
    def test_replace_with_maintains_next_element_throughout(self):
        soup = self.soup('<p><a>one</a><b>three</b></p>')
        a = soup.a
        b = a.contents[0]
        # Make it so the <a> tag has two text children.
        a.insert(1, "two")

        # Now replace each one with the empty string.
        left, right = a.contents
        left.replaceWith('')
        right.replaceWith('')

        # The <b> tag is still connected to the tree.
        assert "three" == soup.b.string

    def test_replace_final_node(self):
        soup = self.soup("<b>Argh!</b>")
        soup.find(string="Argh!").replace_with("Hooray!")
        new_text = soup.find(string="Hooray!")
        b = soup.b
        assert new_text.previous_element == b
        assert new_text.parent == b
        assert new_text.previous_element.next_element == new_text
        assert new_text.next_element == None

    def test_consecutive_text_nodes(self):
        # A builder should never create two consecutive text nodes,
        # but if you insert one next to another, Beautiful Soup will
        # handle it correctly.
        soup = self.soup("<a><b>Argh!</b><c></c></a>")
        soup.b.insert(1, "Hooray!")

        assert soup.decode() == self.document_for(
            "<a><b>Argh!Hooray!</b><c></c></a>"
        )

        new_text = soup.find(string="Hooray!")
        assert new_text.previous_element == "Argh!"
        assert new_text.previous_element.next_element == new_text

        assert new_text.previous_sibling == "Argh!"
        assert new_text.previous_sibling.next_sibling == new_text

        assert new_text.next_sibling == None
        assert new_text.next_element == soup.c

    def test_insert_string(self):
        soup = self.soup("<a></a>")
        soup.a.insert(0, "bar")
        soup.a.insert(0, "foo")
        # The string were added to the tag.
        assert ["foo", "bar"] == soup.a.contents
        # And they were converted to NavigableStrings.
        assert soup.a.contents[0].next_element == "bar"

    def test_insert_tag(self):
        builder = self.default_builder()
        soup = self.soup(
            "<a><b>Find</b><c>lady!</c><d></d></a>", builder=builder)
        magic_tag = Tag(soup, builder, 'magictag')
        magic_tag.insert(0, "the")
        soup.a.insert(1, magic_tag)

        assert soup.decode() == self.document_for(
                "<a><b>Find</b><magictag>the</magictag><c>lady!</c><d></d></a>"
        )

        # Make sure all the relationships are hooked up correctly.
        b_tag = soup.b
        assert b_tag.next_sibling == magic_tag
        assert magic_tag.previous_sibling == b_tag

        find = b_tag.find(string="Find")
        assert find.next_element == magic_tag
        assert magic_tag.previous_element == find

        c_tag = soup.c
        assert magic_tag.next_sibling == c_tag
        assert c_tag.previous_sibling == magic_tag

        the = magic_tag.find(string="the")
        assert the.parent == magic_tag
        assert the.next_element == c_tag
        assert c_tag.previous_element == the

    def test_append_child_thats_already_at_the_end(self):
        data = "<a><b></b></a>"
        soup = self.soup(data)
        soup.a.append(soup.b)
        assert data == soup.decode()

    def test_extend(self):
        data = "<a><b><c><d><e><f><g></g></f></e></d></c></b></a>"
        soup = self.soup(data)
        l = [soup.g, soup.f, soup.e, soup.d, soup.c, soup.b]
        soup.a.extend(l)
        assert "<a><g></g><f></f><e></e><d></d><c></c><b></b></a>" == soup.decode()

    @pytest.mark.parametrize(
        "get_tags", [lambda tag: tag, lambda tag: tag.contents]
    )
    def test_extend_with_another_tags_contents(self, get_tags):
        data = '<body><div id="d1"><a>1</a><a>2</a><a>3</a><a>4</a></div><div id="d2"></div></body>'
        soup = self.soup(data)
        d1 = soup.find('div', id='d1')
        d2 = soup.find('div', id='d2')
        tags = get_tags(d1)
        d2.extend(tags)
        assert '<div id="d1"></div>' == d1.decode()
        assert '<div id="d2"><a>1</a><a>2</a><a>3</a><a>4</a></div>' == d2.decode()
        
    def test_move_tag_to_beginning_of_parent(self):
        data = "<a><b></b><c></c><d></d></a>"
        soup = self.soup(data)
        soup.a.insert(0, soup.d)
        assert "<a><d></d><b></b><c></c></a>" == soup.decode()

    def test_insert_works_on_empty_element_tag(self):
        # This is a little strange, since most HTML parsers don't allow
        # markup like this to come through. But in general, we don't
        # know what the parser would or wouldn't have allowed, so
        # I'm letting this succeed for now.
        soup = self.soup("<br/>")
        soup.br.insert(1, "Contents")
        assert str(soup.br) == "<br>Contents</br>"

    def test_insert_before(self):
        soup = self.soup("<a>foo</a><b>bar</b>")
        soup.b.insert_before("BAZ")
        soup.a.insert_before("QUUX")
        assert soup.decode() == self.document_for(
            "QUUX<a>foo</a>BAZ<b>bar</b>"
        )

        soup.a.insert_before(soup.b)
        assert soup.decode() == self.document_for("QUUX<b>bar</b><a>foo</a>BAZ")

        # Can't insert an element before itself.
        b = soup.b
        with pytest.raises(ValueError):
            b.insert_before(b)

        # Can't insert before if an element has no parent.
        b.extract()
        with pytest.raises(ValueError):
            b.insert_before("nope")

        # Can insert an identical element
        soup = self.soup("<a>")
        soup.a.insert_before(soup.new_tag("a"))

        # TODO: OK but what happens?
        
    def test_insert_multiple_before(self):
        soup = self.soup("<a>foo</a><b>bar</b>")
        soup.b.insert_before("BAZ", " ", "QUUX")
        soup.a.insert_before("QUUX", " ", "BAZ")
        assert soup.decode() == self.document_for(
            "QUUX BAZ<a>foo</a>BAZ QUUX<b>bar</b>"
        )

        soup.a.insert_before(soup.b, "FOO")
        assert soup.decode() == self.document_for(
            "QUUX BAZ<b>bar</b>FOO<a>foo</a>BAZ QUUX"
        )

    def test_insert_after(self):
        soup = self.soup("<a>foo</a><b>bar</b>")
        soup.b.insert_after("BAZ")
        soup.a.insert_after("QUUX")
        assert soup.decode() == self.document_for(
            "<a>foo</a>QUUX<b>bar</b>BAZ"
        )
        soup.b.insert_after(soup.a)
        assert soup.decode() == self.document_for("QUUX<b>bar</b><a>foo</a>BAZ")

        # Can't insert an element after itself.
        b = soup.b
        with pytest.raises(ValueError):
            b.insert_after(b)

        # Can't insert after if an element has no parent.
        b.extract()
        with pytest.raises(ValueError):
            b.insert_after("nope")

        # Can insert an identical element
        soup = self.soup("<a>")
        soup.a.insert_before(soup.new_tag("a"))

        # TODO: OK but what does it look like?
        
    def test_insert_multiple_after(self):
        soup = self.soup("<a>foo</a><b>bar</b>")
        soup.b.insert_after("BAZ", " ", "QUUX")
        soup.a.insert_after("QUUX", " ", "BAZ")
        assert soup.decode() == self.document_for(
            "<a>foo</a>QUUX BAZ<b>bar</b>BAZ QUUX"
        )
        soup.b.insert_after(soup.a, "FOO ")
        assert soup.decode() == self.document_for(
            "QUUX BAZ<b>bar</b><a>foo</a>FOO BAZ QUUX"
        )

    def test_insert_after_raises_exception_if_after_has_no_meaning(self):
        soup = self.soup("")
        tag = soup.new_tag("a")
        string = soup.new_string("")
        with pytest.raises(ValueError):
            string.insert_after(tag)
        with pytest.raises(NotImplementedError):
            soup.insert_after(tag)
        with pytest.raises(ValueError):
            tag.insert_after(tag)

    def test_insert_before_raises_notimplementederror_if_before_has_no_meaning(self):
        soup = self.soup("")
        tag = soup.new_tag("a")
        string = soup.new_string("")
        with pytest.raises(ValueError):
            string.insert_before(tag)
        with pytest.raises(NotImplementedError):
            soup.insert_before(tag)
        with pytest.raises(ValueError):
            tag.insert_before(tag)

    def test_replace_with(self):
        soup = self.soup(
                "<p>There's <b>no</b> business like <b>show</b> business</p>")
        no, show = soup.find_all('b')
        show.replace_with(no)
        assert soup.decode() == self.document_for(
            "<p>There's  business like <b>no</b> business</p>"
        )

        assert show.parent == None
        assert no.parent == soup.p
        assert no.next_element == "no"
        assert no.next_sibling == " business"

    def test_replace_with_errors(self):
        # Can't replace a tag that's not part of a tree.
        a_tag = Tag(name="a")
        with pytest.raises(ValueError):
            a_tag.replace_with("won't work")

        # Can't replace a tag with its parent.
        a_tag = self.soup("<a><b></b></a>").a
        with pytest.raises(ValueError):
            a_tag.b.replace_with(a_tag)

        # Or with a list that includes its parent.
        with pytest.raises(ValueError):
            a_tag.b.replace_with("string1", a_tag, "string2")
        
    def test_replace_with_multiple(self):
        data = "<a><b></b><c></c></a>"
        soup = self.soup(data)
        d_tag = soup.new_tag("d")
        d_tag.string = "Text In D Tag"
        e_tag = soup.new_tag("e")
        f_tag = soup.new_tag("f")
        a_string = "Random Text"
        soup.c.replace_with(d_tag, e_tag, a_string, f_tag)
        assert soup.decode() == "<a><b></b><d>Text In D Tag</d><e></e>Random Text<f></f></a>"
        assert soup.b.next_element == d_tag
        assert d_tag.string.next_element==e_tag
        assert e_tag.next_element.string == a_string
        assert e_tag.next_element.next_element == f_tag
        
    def test_replace_first_child(self):
        data = "<a><b></b><c></c></a>"
        soup = self.soup(data)
        soup.b.replace_with(soup.c)
        assert "<a><c></c></a>" == soup.decode()

    def test_replace_last_child(self):
        data = "<a><b></b><c></c></a>"
        soup = self.soup(data)
        soup.c.replace_with(soup.b)
        assert "<a><b></b></a>" == soup.decode()

    def test_nested_tag_replace_with(self):
        soup = self.soup(
            """<a>We<b>reserve<c>the</c><d>right</d></b></a><e>to<f>refuse</f><g>service</g></e>""")

        # Replace the entire <b> tag and its contents ("reserve the
        # right") with the <f> tag ("refuse").
        remove_tag = soup.b
        move_tag = soup.f
        remove_tag.replace_with(move_tag)

        assert soup.decode() == self.document_for(
            "<a>We<f>refuse</f></a><e>to<g>service</g></e>"
        )

        # The <b> tag is now an orphan.
        assert remove_tag.parent == None
        assert remove_tag.find(string="right").next_element == None
        assert remove_tag.previous_element == None
        assert remove_tag.next_sibling == None
        assert remove_tag.previous_sibling == None

        # The <f> tag is now connected to the <a> tag.
        assert move_tag.parent == soup.a
        assert move_tag.previous_element == "We"
        assert move_tag.next_element.next_element == soup.e
        assert move_tag.next_sibling == None

        # The gap where the <f> tag used to be has been mended, and
        # the word "to" is now connected to the <g> tag.
        to_text = soup.find(string="to")
        g_tag = soup.g
        assert to_text.next_element == g_tag
        assert to_text.next_sibling == g_tag
        assert g_tag.previous_element == to_text
        assert g_tag.previous_sibling == to_text

    def test_unwrap(self):
        tree = self.soup("""
            <p>Unneeded <em>formatting</em> is unneeded</p>
            """)
        tree.em.unwrap()
        assert tree.em == None
        assert tree.p.text == "Unneeded formatting is unneeded"

    def test_wrap(self):
        soup = self.soup("I wish I was bold.")
        value = soup.string.wrap(soup.new_tag("b"))
        assert value.decode() == "<b>I wish I was bold.</b>"
        assert soup.decode() == self.document_for("<b>I wish I was bold.</b>")

    def test_wrap_extracts_tag_from_elsewhere(self):
        soup = self.soup("<b></b>I wish I was bold.")
        soup.b.next_sibling.wrap(soup.b)
        assert soup.decode() == self.document_for("<b>I wish I was bold.</b>")

    def test_wrap_puts_new_contents_at_the_end(self):
        soup = self.soup("<b>I like being bold.</b>I wish I was bold.")
        soup.b.next_sibling.wrap(soup.b)
        assert 2 == len(soup.b.contents)
        assert soup.decode() == self.document_for(
            "<b>I like being bold.I wish I was bold.</b>"
        )

    def test_extract(self):
        soup = self.soup(
            '<html><body>Some content. <div id="nav">Nav crap</div> More content.</body></html>')

        assert len(soup.body.contents) == 3
        extracted = soup.find(id="nav").extract()

        assert soup.decode() == "<html><body>Some content.  More content.</body></html>"
        assert extracted.decode() == '<div id="nav">Nav crap</div>'

        # The extracted tag is now an orphan.
        assert len(soup.body.contents) == 2
        assert extracted.parent == None
        assert extracted.previous_element == None
        assert extracted.next_element.next_element == None

        # The gap where the extracted tag used to be has been mended.
        content_1 = soup.find(string="Some content. ")
        content_2 = soup.find(string=" More content.")
        assert content_1.next_element == content_2
        assert content_1.next_sibling == content_2
        assert content_2.previous_element == content_1
        assert content_2.previous_sibling == content_1

    def test_extract_distinguishes_between_identical_strings(self):
        soup = self.soup("<a>foo</a><b>bar</b>")
        foo_1 = soup.a.string
        bar_1 = soup.b.string
        foo_2 = soup.new_string("foo")
        bar_2 = soup.new_string("bar")
        soup.a.append(foo_2)
        soup.b.append(bar_2)

        # Now there are two identical strings in the <a> tag, and two
        # in the <b> tag. Let's remove the first "foo" and the second
        # "bar".
        foo_1.extract()
        bar_2.extract()
        assert foo_2 == soup.a.string
        assert bar_2 == soup.b.string

    def test_extract_multiples_of_same_tag(self):
        soup = self.soup("""
<html>
<head>
<script>foo</script>
</head>
<body>
 <script>bar</script>
 <a></a>
</body>
<script>baz</script>
</html>""")
        [soup.script.extract() for i in soup.find_all("script")]
        assert "<body>\n\n<a></a>\n</body>" == str(soup.body)


    def test_extract_works_when_element_is_surrounded_by_identical_strings(self):
        soup = self.soup(
 '<html>\n'
 '<body>hi</body>\n'
 '</html>')
        soup.find('body').extract()
        assert None == soup.find('body')


    def test_clear(self):
        """Tag.clear()"""
        soup = self.soup("<p><a>String <em>Italicized</em></a> and another</p>")
        # clear using extract()
        a = soup.a
        soup.p.clear()
        assert len(soup.p.contents) == 0
        assert hasattr(a, "contents")

        # clear using decompose()
        em = a.em
        a.clear(decompose=True)
        assert 0 == len(em.contents)

       
    def test_decompose(self):
        # Test PageElement.decompose() and PageElement.decomposed
        soup = self.soup("<p><a>String <em>Italicized</em></a></p><p>Another para</p>")
        p1, p2 = soup.find_all('p')
        a = p1.a
        text = p1.em.string
        for i in [p1, p2, a, text]:
            assert False == i.decomposed

        # This sets p1 and everything beneath it to decomposed.
        p1.decompose()
        for i in [p1, a, text]:
            assert True == i.decomposed
        # p2 is unaffected.
        assert False == p2.decomposed
            
    def test_string_set(self):
        """Tag.string = 'string'"""
        soup = self.soup("<a></a> <b><c></c></b>")
        soup.a.string = "foo"
        assert soup.a.contents == ["foo"]
        soup.b.string = "bar"
        assert soup.b.contents == ["bar"]

    def test_string_set_does_not_affect_original_string(self):
        soup = self.soup("<a><b>foo</b><c>bar</c>")
        soup.b.string = soup.c.string
        assert soup.a.encode() == b"<a><b>bar</b><c>bar</c></a>"

    def test_set_string_preserves_class_of_string(self):
        soup = self.soup("<a></a>")
        cdata = CData("foo")
        soup.a.string = cdata
        assert isinstance(soup.a.string, CData)


class TestDeprecatedArguments(SoupTest):

    @pytest.mark.parametrize(
        "method_name", [
            "find", "find_all", "find_parent", "find_parents",
            "find_next", "find_all_next", "find_previous",
            "find_all_previous", "find_next_sibling", "find_next_siblings",
            "find_previous_sibling", "find_previous_siblings",
        ]
    )
    def test_find_type_method_string(self, method_name):
        soup = self.soup("<a>some</a><b>markup</b>")
        method = getattr(soup.b, method_name)
        with warnings.catch_warnings(record=True) as w:
            method(text='markup')
            [warning] = w
            assert warning.filename == __file__
            msg = str(warning.message)
            assert msg == "The 'text' argument to find()-type methods is deprecated. Use 'string' instead."

    def test_soupstrainer_constructor_string(self):
        with warnings.catch_warnings(record=True) as w:
            strainer = SoupStrainer(text="text")
            assert strainer.text == 'text'
            [warning] = w
            msg = str(warning.message)
            assert warning.filename == __file__
            assert msg == "The 'text' argument to the SoupStrainer constructor is deprecated. Use 'string' instead."

