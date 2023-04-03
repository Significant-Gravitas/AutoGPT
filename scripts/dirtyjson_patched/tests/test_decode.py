from __future__ import absolute_import
import decimal
from unittest import TestCase

import dirtyjson


class TestDecode(TestCase):
    if not hasattr(TestCase, 'assertIs'):
        def assertIs(self, expr1, expr2, msg=None):
            self.assertTrue(expr1 is expr2, msg or '%r is %r' % (expr1, expr2))

    def test_decimal(self):
        rval = dirtyjson.loads('1.1', parse_float=decimal.Decimal)
        self.assertTrue(isinstance(rval, decimal.Decimal))
        self.assertEqual(rval, decimal.Decimal('1.1'))

    def test_float(self):
        rval = dirtyjson.loads('1', parse_int=float)
        self.assertTrue(isinstance(rval, float))
        self.assertEqual(rval, 1.0)

    def test_decoder_optimizations(self):
        rval = dirtyjson.loads('{   "key"    :    "value"    ,  "k":"v"    }')
        self.assertEqual(rval, {"key": "value", "k": "v"})

    def test_empty_objects(self):
        s = '{}'
        self.assertEqual(dirtyjson.loads(s), eval(s))
        s = '[]'
        self.assertEqual(dirtyjson.loads(s), eval(s))
        s = '""'
        self.assertEqual(dirtyjson.loads(s), eval(s))

    def check_keys_reuse(self, source, loads):
        rval = loads(source)
        (a, b), (c, d) = sorted(rval[0]), sorted(rval[1])
        self.assertIs(a, c)
        self.assertIs(b, d)

    def test_keys_reuse_str(self):
        s = u'[{"a_key": 1, "b_\xe9": 2}, {"a_key": 3, "b_\xe9": 4}]'.encode('utf8')
        self.check_keys_reuse(s, dirtyjson.loads)

    def test_keys_reuse_unicode(self):
        s = u'[{"a_key": 1, "b_\xe9": 2}, {"a_key": 3, "b_\xe9": 4}]'
        self.check_keys_reuse(s, dirtyjson.loads)

    def test_empty_strings(self):
        self.assertEqual(dirtyjson.loads('""'), "")
        self.assertEqual(dirtyjson.loads(u'""'), u"")
        self.assertEqual(dirtyjson.loads('[""]'), [""])
        self.assertEqual(dirtyjson.loads(u'[""]'), [u""])

    def test_empty_strings_with_single_quotes(self):
        self.assertEqual(dirtyjson.loads("''"), "")
        self.assertEqual(dirtyjson.loads(u"''"), u"")
        self.assertEqual(dirtyjson.loads("['']"), [""])
        self.assertEqual(dirtyjson.loads(u"['']"), [u""])

    def test_object_keys(self):
        result = {"key": "value", "k": "v"}
        rval = dirtyjson.loads("""{"key": "value", "k": "v"}""")
        self.assertEqual(rval, result)
        rval = dirtyjson.loads("""{'key': 'value', 'k': 'v'}""")
        self.assertEqual(rval, result)
        rval = dirtyjson.loads("""{key: 'value', k: 'v'}""")
        self.assertEqual(rval, result)
        rval = dirtyjson.loads("""{key: 'value', k: 'v',}""")
        self.assertEqual(rval, result)

    def test_not_at_beginning(self):
        s = """
// here are some comments
var a = 1; // here is a line of regular JS

var b = {test: 1, 'aack': 0x80, "bar": [1, 2, 3]};
console.log(b);
"""
        first_object_index = s.index('{')

        rval = dirtyjson.loads(s, start_index=first_object_index)
        self.assertEqual(rval, {'test': 1, 'aack': 0x80, 'bar': [1, 2, 3]})

        rval = dirtyjson.loads(s, start_index=first_object_index + 1, search_for_first_object=True)
        self.assertEqual(rval, [1, 2, 3])

        rval = dirtyjson.loads(s, search_for_first_object=True)
        self.assertEqual(rval, {'test': 1, 'aack': 0x80, 'bar': [1, 2, 3]})

    def test_ignore_single_line_comments(self):
        s = """
// here are some comments
{
    // comments inside too
    test: 1,    // and at the end of lines
    'aack': 0x80,
    "bar": [ // even inside arrays
        1,
        2,
        3, // and after trailing commas
    ],
    more: { // and inside objects
        once: true,
        twice: false,
        three_times3: null // and at the end
    }
}
console.log(b);
"""
        rval = dirtyjson.loads(s)
        self.assertEqual(rval, {'test': 1, 'aack': 0x80, 'bar': [1, 2, 3], 'more': {'once': True, 'twice': False, 'three_times3': None}})

    def test_ignore_inline_comments(self):
        s = """
/* here are some comments
 * that should all be skipped
 * right up until the terminator */ {
    /* comments inside too */
    test: 1,    /* and at the end of lines */
    'aack': 0x80,
    "bar": [ // even inside arrays
        1,
        2,
        3, // and after trailing commas
    ],
    /* comment this block out
    more: { // and inside objects
        once: true,
        twice: false,
        three_times3: null // and at the end
    } */
}
console.log(b);
"""
        rval = dirtyjson.loads(s)
        self.assertEqual(rval, {'test': 1, 'aack': 0x80, 'bar': [1, 2, 3]})
