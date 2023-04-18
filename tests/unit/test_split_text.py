import unittest
from autogpt.processing.text import split_long_paragraph


class TestSplitLongParagraph(unittest.TestCase):
    def test_split_long_paragraph(self):
        paragraph = "This is a sample paragraph to test the functionality of the split_long_paragraph function. It should split this paragraph into smaller chunks of text based on the specified maximum length, ensuring that the splits occur at whitespace or other non-word characters."

        # Test case 1: max_length >= len(paragraph)
        result = split_long_paragraph(paragraph, max_length=300)
        self.assertEqual(result, [paragraph])

        # Test case 2: max_length < len(paragraph) and splits on whitespace
        max_length = 26
        result = split_long_paragraph(paragraph, max_length=max_length)
        for chunk in result:
            self.assertLessEqual(len(chunk), max_length)

        # Test case 3: max_length is small, but still larger than the longest word
        max_length = 10
        result = split_long_paragraph(paragraph, max_length=max_length)
        for chunk in result:
            self.assertLessEqual(len(chunk), max_length)


if __name__ == "__main__":
    unittest.main()
