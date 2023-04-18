import unittest
from autogpt.processing.text import split_text

class TestSplitText(unittest.TestCase):
    def check_chunk_lengths(self, chunks, max_length):
        for chunk in chunks:
            self.assertLessEqual(len(chunk), max_length)

    def test_empty_string(self):
        text = ""
        result = list(split_text(text))
        self.check_chunk_lengths(result, 8192)

    def test_no_split_required(self):
        text = "Hello, world!"
        max_length = 20
        result = list(split_text(text, max_length=max_length))
        self.check_chunk_lengths(result, max_length)

    def test_split_required(self):
        text = "Hello, world!\nHow are you today?"
        max_length = 15
        result = list(split_text(text, max_length=max_length))
        self.check_chunk_lengths(result, max_length)

    def test_long_paragraph_split(self):
        text = "This is a very long paragraph that needs to be split into smaller chunks."
        max_length = 10
        result = list(split_text(text, max_length=max_length))
        self.check_chunk_lengths(result, max_length)

    def test_split_with_whitespace(self):
        text = "This is a test\n\nwith extra whitespace."
        max_length = 15
        result = list(split_text(text, max_length=max_length))
        self.check_chunk_lengths(result, max_length)

    def test_value_error(self):
        text = "This is a very long paragraph that needs to be split into smaller chunks."
        with self.assertRaises(ValueError):
            list(split_text(text, max_length=0))

    def test_max_length_larger_than_text(self):
        text = "Hello, world!"
        max_length = 10000
        result = list(split_text(text, max_length=max_length))
        self.check_chunk_lengths(result, max_length)

if __name__ == "__main__":
    unittest.main()
