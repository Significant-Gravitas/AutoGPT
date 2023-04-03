from __future__ import absolute_import
import unittest
import doctest


def additional_tests(suite=None):
    import dirtyjson
    import dirtyjson.loader
    if suite is None:
        suite = unittest.TestSuite()
    for mod in (dirtyjson, dirtyjson.loader):
        suite.addTest(doctest.DocTestSuite(mod))
    suite.addTest(doctest.DocFileSuite('../../index.rst'))
    return suite


def all_tests_suite():
    suite = unittest.TestLoader().loadTestsFromNames([
        'dirtyjson.tests.test_decimal',
        'dirtyjson.tests.test_decode',
        'dirtyjson.tests.test_errors',
        'dirtyjson.tests.test_fail',
        'dirtyjson.tests.test_float',
        'dirtyjson.tests.test_integer',
        'dirtyjson.tests.test_pass1',
        'dirtyjson.tests.test_pass2',
        'dirtyjson.tests.test_pass3',
        'dirtyjson.tests.test_unicode',
    ])
    suite = additional_tests(suite)
    return unittest.TestSuite([suite])


def main():
    runner = unittest.TextTestRunner(verbosity=1 + sys.argv.count('-v'))
    suite = all_tests_suite()
    raise SystemExit(not runner.run(suite).wasSuccessful())


if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
