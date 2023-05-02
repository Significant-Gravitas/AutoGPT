import os
import sys
import unittest

from bs4 import BeautifulSoup

sys.path.append(os.path.abspath("../scripts"))

from browse import extract_hyperlinks


class TestBrowseLinks(unittest.TestCase):
    """Unit tests for the browse module functions that extract hyperlinks."""

    def test_extract_hyperlinks(self):
        """Test the extract_hyperlinks function with a simple HTML body."""
        body = """
        <body>
        <a href="https://google.com">Google</a>
        <a href="foo.html">Foo</a>
        <div>Some other crap</div>
        </body>
        """
        soup = BeautifulSoup(body, "html.parser")
        links = extract_hyperlinks(soup, "http://example.com")
        self.assertEqual(
            links,
            [("Google", "https://google.com"), ("Foo", "http://example.com/foo.html")],
        )
