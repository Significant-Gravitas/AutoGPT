import sys
from unittest import TestCase

import dirtyjson
from dirtyjson.compat import unichr, text_type


class TestUnicode(TestCase):
    def test_big_unicode_decode(self):
        uc = u'z\U0001d120x'
        self.assertEqual(dirtyjson.loads('"' + uc + '"'), uc)
        self.assertEqual(dirtyjson.loads('"z\\ud834\\udd20x"'), uc)

    def test_unicode_decode(self):
        for i in range(0, 0xd7ff):
            uc = unichr(i)
            s = '"\\u%04x"' % (i,)
            self.assertEqual(dirtyjson.loads(s), uc)

    def test_default_encoding(self):
        self.assertEqual(dirtyjson.loads(u'{"a": "\xe9"}'.encode('utf-8')),
                         {'a': u'\xe9'})

    def test_unicode_preservation(self):
        self.assertEqual(type(dirtyjson.loads(u'""')), text_type)
        self.assertEqual(type(dirtyjson.loads(u'"a"')), text_type)
        self.assertEqual(type(dirtyjson.loads(u'["a"]')[0]), text_type)

    def test_invalid_escape_sequences(self):
        # incomplete escape sequence
        self.assertRaises(dirtyjson.Error, dirtyjson.loads, '"\\u"')
        self.assertRaises(dirtyjson.Error, dirtyjson.loads, '"\\u1"')
        self.assertRaises(dirtyjson.Error, dirtyjson.loads, '"\\u12"')
        self.assertRaises(dirtyjson.Error, dirtyjson.loads, '"\\u123"')
        # invalid escape sequence
        self.assertRaises(dirtyjson.Error, dirtyjson.loads, '"\\u123x"')
        self.assertRaises(dirtyjson.Error, dirtyjson.loads, '"\\u12x4"')
        self.assertRaises(dirtyjson.Error, dirtyjson.loads, '"\\u1x34"')
        self.assertRaises(dirtyjson.Error, dirtyjson.loads, '"\\ux234"')
        if sys.maxunicode > 65535:
            # invalid escape sequence for low surrogate
            self.assertRaises(dirtyjson.Error, dirtyjson.loads, '"\\ud800\\u"')
            self.assertRaises(dirtyjson.Error, dirtyjson.loads, '"\\ud800\\u0"')
            self.assertRaises(dirtyjson.Error, dirtyjson.loads, '"\\ud800\\u00"')
            self.assertRaises(dirtyjson.Error, dirtyjson.loads, '"\\ud800\\u000"')
            self.assertRaises(dirtyjson.Error, dirtyjson.loads, '"\\ud800\\u000x"')
            self.assertRaises(dirtyjson.Error, dirtyjson.loads, '"\\ud800\\u00x0"')
            self.assertRaises(dirtyjson.Error, dirtyjson.loads, '"\\ud800\\u0x00"')
            self.assertRaises(dirtyjson.Error, dirtyjson.loads, '"\\ud800\\ux000"')
