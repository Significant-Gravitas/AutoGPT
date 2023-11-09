# -*- coding: utf-8 -*-
import unittest
from gtts.tokenizer.tokenizer_cases import (
    tone_marks,
    period_comma,
    colon,
    other_punctuation,
    legacy_all_punctuation,
)
from gtts.tokenizer import Tokenizer, symbols


class TestPreTokenizerCases(unittest.TestCase):
    def test_tone_marks(self):
        t = Tokenizer([tone_marks])
        _in = "Lorem? Ipsum!"
        _out = ["Lorem?", "Ipsum!"]
        self.assertEqual(t.run(_in), _out)

    def test_period_comma(self):
        t = Tokenizer([period_comma])
        _in = "Hello, it's 24.5 degrees in the U.K. today. $20,000,000."
        _out = ["Hello", "it's 24.5 degrees in the U.K. today", "$20,000,000."]
        self.assertEqual(t.run(_in), _out)

    def test_colon(self):
        t = Tokenizer([colon])
        _in = "It's now 6:30 which means: morning missing:space"
        _out = ["It's now 6:30 which means", " morning missing", "space"]
        self.assertEqual(t.run(_in), _out)

    def test_other_punctuation(self):
        # String of the unique 'other punctuations'
        other_punc_str = "".join(
            set(symbols.ALL_PUNC)
            - set(symbols.TONE_MARKS)
            - set(symbols.PERIOD_COMMA)
            - set(symbols.COLON)
        )

        t = Tokenizer([other_punctuation])
        self.assertEqual(len(t.run(other_punc_str)) - 1, len(other_punc_str))

    def test_legacy_all_punctuation(self):
        t = Tokenizer([legacy_all_punctuation])
        self.assertEqual(len(t.run(symbols.ALL_PUNC)) - 1, len(symbols.ALL_PUNC))


if __name__ == "__main__":
    unittest.main()
