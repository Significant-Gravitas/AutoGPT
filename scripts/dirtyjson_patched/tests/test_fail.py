import sys
from unittest import TestCase

import dirtyjson

# 2007-10-05
JSONDOCS = [
    # http://json.org/JSON_checker/test/fail2.json
    '["Unclosed array"',
    # http://json.org/JSON_checker/test/fail5.json
    '["double extra comma",,]',
    # http://json.org/JSON_checker/test/fail6.json
    '[   , "<-- missing value"]',
    # http://json.org/JSON_checker/test/fail11.json
    '{"Illegal expression": 1 + 2}',
    # http://json.org/JSON_checker/test/fail12.json
    '{"Illegal invocation": alert()}',
    # http://json.org/JSON_checker/test/fail15.json
    '["Illegal backslash escape: \\x15"]',
    # http://json.org/JSON_checker/test/fail16.json
    '[\\naked]',
    # http://json.org/JSON_checker/test/fail17.json
    '["Illegal backslash escape: \\017"]',
    # http://json.org/JSON_checker/test/fail19.json
    '{"Missing colon" null}',
    # http://json.org/JSON_checker/test/fail20.json
    '{"Double colon":: null}',
    # http://json.org/JSON_checker/test/fail21.json
    '{"Comma instead of colon", null}',
    # http://json.org/JSON_checker/test/fail22.json
    '["Colon instead of comma": false]',
    # http://json.org/JSON_checker/test/fail23.json
    '["Bad value", truth]',
    # http://json.org/JSON_checker/test/fail26.json
    '["tab\\   character\\   in\\  string\\  "]',
    # http://json.org/JSON_checker/test/fail28.json
    '["line\\\nbreak"]',
    # http://json.org/JSON_checker/test/fail29.json
    '[0e]',
    # http://json.org/JSON_checker/test/fail30.json
    '[0e+]',
    # http://json.org/JSON_checker/test/fail31.json
    '[0e+-1]',
    # http://json.org/JSON_checker/test/fail32.json
    '{"Comma instead if closing brace": true,',
    # http://json.org/JSON_checker/test/fail33.json
    '["mismatch"}',
    # misc based on coverage
    '{',
    '{]',
    '{"foo": "bar"]',
    '{"foo": "bar"',
    'nul',
    'nulx',
    '-',
    '-x',
    '-e',
    '-e0',
    '-Infinite',
    '-Inf',
    'Infinit',
    'Infinite',
    'NaM',
    'NuN',
    'falsy',
    'fal',
    'trug',
    'tru',
]


class TestFail(TestCase):
    def test_failures(self):
        for idx, doc in enumerate(JSONDOCS):
            idx += 1
            try:
                dirtyjson.loads(doc)
            except dirtyjson.Error:
                pass
            else:
                self.fail("Expected failure for fail%d.json: %r" % (idx, doc))

    def test_array_decoder_issue46(self):
        # http://code.google.com/p/dirtyjson/issues/detail?id=46
        for doc in [u'[,]', '[,]']:
            try:
                dirtyjson.loads(doc)
            except dirtyjson.Error:
                e = sys.exc_info()[1]
                self.assertEqual(e.pos, 1)
                self.assertEqual(e.lineno, 1)
                self.assertEqual(e.colno, 2)
            except Exception:
                e = sys.exc_info()[1]
                self.fail("Unexpected exception raised %r %s" % (e, e))
            else:
                self.fail("Unexpected success parsing '[,]'")

    def test_truncated_input(self):
        test_cases = [
            ('', 'Expecting value', 0),
            ('[', "Expecting value or ']'", 1),
            ('[42', "Expecting ',' delimiter", 3),
            ('[42,', 'Expecting value', 4),
            ('["', 'Unterminated string starting at', 1),
            ('["spam', 'Unterminated string starting at', 1),
            ('["spam"', "Expecting ',' delimiter", 7),
            ('["spam",', 'Expecting value', 8),
            ('{', 'Expecting property name', 1),
            ('{"', 'Unterminated string starting at', 1),
            ('{"spam', 'Unterminated string starting at', 1),
            ('{"spam"', "Expecting ':' delimiter", 7),
            ('{"spam":', 'Expecting value', 8),
            ('{"spam":42', "Expecting ',' delimiter", 10),
            ('{"spam":42,', 'Expecting property name',
             11),
            ('"', 'Unterminated string starting at', 0),
            ('"spam', 'Unterminated string starting at', 0),
            ('[,', "Expecting value", 1),
        ]
        for data, msg, idx in test_cases:
            try:
                dirtyjson.loads(data)
            except dirtyjson.Error:
                e = sys.exc_info()[1]
                self.assertEqual(
                    e.msg[:len(msg)],
                    msg,
                    "%r doesn't start with %r for %r" % (e.msg, msg, data))
                self.assertEqual(
                    e.pos, idx,
                    "pos %r != %r for %r" % (e.pos, idx, data))
            except Exception:
                e = sys.exc_info()[1]
                self.fail("Unexpected exception raised %r %s" % (e, e))
            else:
                self.fail("Unexpected success parsing '%r'" % (data,))
