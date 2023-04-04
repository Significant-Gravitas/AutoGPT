"Test harness for doctests."

# TODO: Pretty sure this isn't used and should be deleted.

# pylint: disable-msg=E0611,W0142

__metaclass__ = type
__all__ = [
    'additional_tests',
    ]

import atexit
import doctest
import os
#from pkg_resources import (
#    resource_filename, resource_exists, resource_listdir, cleanup_resources)
import unittest

DOCTEST_FLAGS = (
    doctest.ELLIPSIS |
    doctest.NORMALIZE_WHITESPACE |
    doctest.REPORT_NDIFF)


# def additional_tests():
#     "Run the doc tests (README.txt and docs/*, if any exist)"
#     doctest_files = [
#         os.path.abspath(resource_filename('bs4', 'README.txt'))]
#     if resource_exists('bs4', 'docs'):
#         for name in resource_listdir('bs4', 'docs'):
#             if name.endswith('.txt'):
#                 doctest_files.append(
#                     os.path.abspath(
#                         resource_filename('bs4', 'docs/%s' % name)))
#     kwargs = dict(module_relative=False, optionflags=DOCTEST_FLAGS)
#     atexit.register(cleanup_resources)
#     return unittest.TestSuite((
#         doctest.DocFileSuite(*doctest_files, **kwargs)))
