"""
This module contains compatibility helpers for Python 2/3 interoperability.

It mainly exists because their are certain incompatibilities in the Python
syntax that can only be solved by conditionally importing different functions.
"""
import sys
from lxml.etree import tostring

if sys.version_info[0] == 2:
    bytes_ = str
    str_ = unicode
    def tostring_(s):
        return tostring(s, encoding='utf-8').decode('utf-8')

elif sys.version_info[0] == 3:
    bytes_ = bytes
    str_ = str
    def tostring_(s):
        return tostring(s, encoding='utf-8')
