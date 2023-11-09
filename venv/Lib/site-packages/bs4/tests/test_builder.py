import pytest
from unittest.mock import patch
from bs4.builder import DetectsXMLParsedAsHTML

class TestDetectsXMLParsedAsHTML(object):

    @pytest.mark.parametrize(
        "markup,looks_like_xml",
        [("No xml declaration", False),
         ("<html>obviously HTML</html", False),
         ("<?xml ><html>Actually XHTML</html>", False),
         ("<?xml>            <    html>Tricky XHTML</html>", False),
         ("<?xml ><no-html-tag>", True),
        ]
    )
    def test_warn_if_markup_looks_like_xml(self, markup, looks_like_xml):
        # Test of our ability to guess at whether markup looks XML-ish
        # _and_ not HTML-ish.
        with patch('bs4.builder.DetectsXMLParsedAsHTML._warn') as mock:
            for data in markup, markup.encode('utf8'):
                result = DetectsXMLParsedAsHTML.warn_if_markup_looks_like_xml(
                    data
                )
                assert result == looks_like_xml
                if looks_like_xml:
                    assert mock.called
                else:
                    assert not mock.called
                mock.reset_mock()
