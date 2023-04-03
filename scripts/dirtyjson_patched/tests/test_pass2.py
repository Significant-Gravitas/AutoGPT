from unittest import TestCase
import json
import dirtyjson

# from http://json.org/JSON_checker/test/pass2.json
JSON = r'''
[[[[[[[[[[[[[[[[[[["Not too deep"]]]]]]]]]]]]]]]]]]]
'''


class TestPass2(TestCase):
    def test_parse(self):
        # test in/out equivalence and parsing
        res = dirtyjson.loads(JSON)
        out = json.dumps(res)
        self.assertEqual(res, dirtyjson.loads(out))
