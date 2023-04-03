import sys
from unittest import TestCase

import dirtyjson
from dirtyjson.compat import u, b


class TestErrors(TestCase):
    def test_scan_error(self):
        for t in (u, b):
            try:
                dirtyjson.loads(t('{"asdf": "'))
            except dirtyjson.Error:
                err = sys.exc_info()[1]
            else:
                self.fail('Expected JSONDecodeError')
            self.assertEqual(err.lineno, 1)
            self.assertEqual(err.colno, 10)
