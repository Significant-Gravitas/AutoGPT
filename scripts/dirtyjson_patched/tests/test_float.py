import math
import json
from unittest import TestCase
import dirtyjson
from dirtyjson.compat import long_type, text_type
from dirtyjson.loader import NaN, PosInf, NegInf


class TestFloat(TestCase):
    def test_degenerates_allow(self):
        for inf in (PosInf, NegInf):
            self.assertEqual(dirtyjson.loads(json.dumps(inf)), inf)
        # Python 2.5 doesn't have math.isnan
        nan = dirtyjson.loads(json.dumps(NaN))
        self.assertTrue((0 + nan) != nan)

    def test_floats(self):
        for num in [1617161771.7650001, math.pi, math.pi ** 100,
                    math.pi ** -100, 3.1]:
            self.assertEqual(dirtyjson.loads(json.dumps(num)), num)
            self.assertEqual(dirtyjson.loads(text_type(json.dumps(num))), num)

    def test_ints(self):
        for num in [1, long_type(1), 1 << 32, 1 << 64]:
            self.assertEqual(dirtyjson.loads(json.dumps(num)), num)
            self.assertEqual(dirtyjson.loads(text_type(json.dumps(num))), num)
