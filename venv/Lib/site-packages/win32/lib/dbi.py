"""
Skeleton replacement for removed dbi module.
Use of objects created by this module should be replaced with native Python objects.
Dates are now returned as datetime.datetime objects, but will still accept PyTime
objects also.
Raw data for binary fields should be passed as buffer objects for Python 2.x,
and memoryview objects in Py3k.
"""

import warnings

warnings.warn(
    "dbi module is obsolete, code should now use native python datetime and buffer/memoryview objects",
    DeprecationWarning,
)

import datetime

dbDate = dbiDate = datetime.datetime

try:
    dbRaw = dbiRaw = buffer
except NameError:
    dbRaw = dbiRaw = memoryview

# type names are still exported by odbc module
from odbc import *
