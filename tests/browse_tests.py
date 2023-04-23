import os
import sys
import unittest

from bs4 import BeautifulSoup

sys.path.append(os.path.abspath("../scripts"))

from browse import extract_hyperlinks


class TestBrowseLinks(unittest.TestCase):
    def test_extract_hyperlinks(self):
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
