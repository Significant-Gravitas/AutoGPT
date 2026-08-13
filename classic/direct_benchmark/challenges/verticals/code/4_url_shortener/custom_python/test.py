# pyright: reportMissingImports=false
import unittest

from url_shortener import retrieve_url, shorten_url


class TestURLShortener(unittest.TestCase):
    def test_url_retrieval(self):
        # Shorten the URL to get its shortened form
        shortened_url = shorten_url("https://www.example.com")

        # Retrieve the original URL using the shortened URL directly
        retrieved_url = retrieve_url(shortened_url)

        self.assertEqual(
            retrieved_url,
            "https://www.example.com",
            "Retrieved URL does not match the original!",
        )


if __name__ == "__main__":
    unittest.main()
