import unittest

from autogpt.processing.text import split_text


class TestSplitText(unittest.TestCase):
    def test_empty_string(self):
        text = ""
        result = list(split_text(text))
        self.assertEqual(result, [])

    def test_no_split_required(self):
        text = "This is a short text that doesn't require any splitting based on the given maximum length."
        result = list(split_text(text, max_length=100))
        self.assertEqual(result, [text])

    def test_split_required(self):
        text = (
            "This is a longer piece of text that requires splitting based on the given maximum length.\n"
            "It contains multiple lines that will need to be divided into smaller chunks."
        )
        result = list(split_text(text, max_length=50))
        self.assertEqual(
            result,
            [
                "This is a longer piece of text that requires",
                "splitting based on the given maximum length.",
                "It contains multiple lines that will need to be",
                "divided into smaller chunks.",
            ],
        )

    def test_long_paragraph_split(self):
        text = (
            "This is a very long paragraph that needs to be split into smaller chunks. "
            * 10
        )
        result = list(split_text(text, max_length=50))
        expected = [
            "This is a very long paragraph that needs to be",
            "split into smaller chunks. This is a very long",
            "paragraph that needs to be split into smaller",
            "chunks. This is a very long paragraph that needs",
            "to be split into smaller chunks. This is a very",
            "long paragraph that needs to be split into smaller",
            "chunks. This is a very long paragraph that needs",
            "to be split into smaller chunks. This is a very",
            "long paragraph that needs to be split into smaller",
            "chunks. This is a very long paragraph that needs",
            "to be split into smaller chunks. This is a very",
            "long paragraph that needs to be split into smaller",
            "chunks. This is a very long paragraph that needs",
            "to be split into smaller chunks. This is a very",
            "long paragraph that needs to be split into smaller",
            "chunks.",
        ]
        self.assertEqual(result, expected)

    def test_split_with_whitespace(self):
        text = (
            "This is a test\n\nwith extra whitespace and a longer text that might require splitting "
            "based on the given maximum length."
        )
        result = list(split_text(text, max_length=50))
        self.assertEqual(
            result,
            [
                "This is a test",
                "with extra whitespace and a longer text that might",
                "require splitting based on the given maximum",
                "length.",
            ],
        )

    def test_value_error(self):
        text = (
            "This is a very long paragraph that needs to be split into smaller chunks."
        )
        with self.assertRaises(ValueError):
            list(split_text(text, max_length=0))

    def test_max_length_larger_than_text(self):
        text = (
            "This is a text that has a length smaller than the provided maximum length."
        )
        result = list(split_text(text, max_length=10000))
        self.assertEqual(result, [text])


if __name__ == "__main__":
    unittest.main()
