#
# This file is part of pyasn1 software.
#
# Copyright (c) 2005-2019, Ilya Etingof <etingof@gmail.com>
# License: http://snmplabs.com/pyasn1/license.html
#
import time
from datetime import datetime
from sys import version_info

__all__ = ['strptime']


if version_info[:2] <= (2, 4):

    def strptime(text, dateFormat):
        return datetime(*(time.strptime(text, dateFormat)[0:6]))

else:

    def strptime(text, dateFormat):
        return datetime.strptime(text, dateFormat)
