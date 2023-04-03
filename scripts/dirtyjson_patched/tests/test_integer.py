from unittest import TestCase
from dirtyjson.compat import StringIO

import dirtyjson


class TestInteger(TestCase):
    NUMS = ("1", 1), ("10", 10), ("077", 63), ("-1000", -1000), ("0x40", 64), ("-0x40", -64)

    def loads(self, s, **kw):
        sio = StringIO(s)
        res = dirtyjson.loads(s, **kw)
        self.assertEqual(res, dirtyjson.load(sio, **kw))
        return res

    def test_decimal_decode(self):
        for s, n in self.NUMS:
            self.assertEqual(self.loads(s), n)
