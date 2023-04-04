import pytest

from bs4.element import Tag
from bs4.formatter import (
    Formatter,
    HTMLFormatter,
    XMLFormatter,
)
from . import SoupTest

class TestFormatter(SoupTest):

    def test_default_attributes(self):
        # Test the default behavior of Formatter.attributes().
        formatter = Formatter()
        tag = Tag(name="tag")
        tag['b'] = 1
        tag['a'] = 2

        # Attributes come out sorted by name. In Python 3, attributes
        # normally come out of a dictionary in the order they were
        # added.
        assert [('a', 2), ('b', 1)] == formatter.attributes(tag)

        # This works even if Tag.attrs is None, though this shouldn't
        # normally happen.
        tag.attrs = None
        assert [] == formatter.attributes(tag)

        assert ' ' == formatter.indent
        
    def test_sort_attributes(self):
        # Test the ability to override Formatter.attributes() to,
        # e.g., disable the normal sorting of attributes.
        class UnsortedFormatter(Formatter):
            def attributes(self, tag):
                self.called_with = tag
                for k, v in sorted(tag.attrs.items()):
                    if k == 'ignore':
                        continue
                    yield k,v

        soup = self.soup('<p cval="1" aval="2" ignore="ignored"></p>')
        formatter = UnsortedFormatter()
        decoded = soup.decode(formatter=formatter)

        # attributes() was called on the <p> tag. It filtered out one
        # attribute and sorted the other two.
        assert formatter.called_with == soup.p
        assert '<p aval="2" cval="1"></p>' == decoded

    def test_empty_attributes_are_booleans(self):
        # Test the behavior of empty_attributes_are_booleans as well
        # as which Formatters have it enabled.
        
        for name in ('html', 'minimal', None):
            formatter = HTMLFormatter.REGISTRY[name]
            assert False == formatter.empty_attributes_are_booleans

        formatter = XMLFormatter.REGISTRY[None]
        assert False == formatter.empty_attributes_are_booleans

        formatter = HTMLFormatter.REGISTRY['html5']
        assert True == formatter.empty_attributes_are_booleans

        # Verify that the constructor sets the value.
        formatter = Formatter(empty_attributes_are_booleans=True)
        assert True == formatter.empty_attributes_are_booleans

        # Now demonstrate what it does to markup.
        for markup in (
                "<option selected></option>",
                '<option selected=""></option>'
        ):
            soup = self.soup(markup)
            for formatter in ('html', 'minimal', 'xml', None):
                assert b'<option selected=""></option>' == soup.option.encode(formatter='html')
                assert b'<option selected></option>' == soup.option.encode(formatter='html5')

    @pytest.mark.parametrize(
        "indent,expect",
        [
            (None, '<a>\n<b>\ntext\n</b>\n</a>'),
            (-1, '<a>\n<b>\ntext\n</b>\n</a>'),
            (0, '<a>\n<b>\ntext\n</b>\n</a>'),
            ("", '<a>\n<b>\ntext\n</b>\n</a>'),

            (1, '<a>\n <b>\n  text\n </b>\n</a>'),
            (2, '<a>\n  <b>\n    text\n  </b>\n</a>'),

            ("\t", '<a>\n\t<b>\n\t\ttext\n\t</b>\n</a>'),
            ('abc', '<a>\nabc<b>\nabcabctext\nabc</b>\n</a>'),

            # Some invalid inputs -- the default behavior is used.
            (object(), '<a>\n <b>\n  text\n </b>\n</a>'),
            (b'bytes', '<a>\n <b>\n  text\n </b>\n</a>'),
        ]
    )
    def test_indent(self, indent, expect):
        # Pretty-print a tree with a Formatter set to
        # indent in a certain way and verify the results.
        soup = self.soup("<a><b>text</b></a>")
        formatter = Formatter(indent=indent)
        assert soup.prettify(formatter=formatter) == expect

        # Pretty-printing only happens with prettify(), not
        # encode().
        assert soup.encode(formatter=formatter) != expect
        
    def test_default_indent_value(self):
        formatter = Formatter()
        assert formatter.indent == ' '

