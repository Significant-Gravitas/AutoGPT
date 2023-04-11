# -*- coding: utf-8 -*-
import unittest
from gtts.tokenizer.pre_processors import (
    tone_marks,
    end_of_line,
    abbreviations,
    word_sub,
)


class TestPreProcessors(unittest.TestCase):
    def test_tone_marks(self):
        _in = "lorem!ipsum?"
        _out = "lorem! ipsum? "
        self.assertEqual(tone_marks(_in), _out)

    def test_end_of_line(self):
        _in = """test-
ing"""
        _out = "testing"
        self.assertEqual(end_of_line(_in), _out)

    def test_abbreviations(self):
        _in = "jr. sr. dr."
        _out = "jr sr dr"
        self.assertEqual(abbreviations(_in), _out)

    def test_word_sub(self):
        _in = "Esq. Bacon"
        _out = "Esquire Bacon"
        self.assertEqual(word_sub(_in), _out)


if __name__ == "__main__":
    unittest.main()
