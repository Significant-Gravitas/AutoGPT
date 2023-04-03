from decimal import Decimal
from unittest import TestCase
from dirtyjson.compat import StringIO

import dirtyjson


class TestDecimal(TestCase):
    NUMS = "1.0", "10.00", "1.1", "1234567890.1234567890", "500"

    def loads(self, s, **kw):
        sio = StringIO(s)
        res = dirtyjson.loads(s, **kw)
        self.assertEqual(res, dirtyjson.load(sio, **kw))
        return res

    def test_decimal_decode(self):
        for s in self.NUMS:
            self.assertEqual(self.loads(s, parse_float=Decimal), Decimal(s))
