#
# This file is part of pyasn1 software.
#
# Copyright (c) 2005-2019, Ilya Etingof <etingof@gmail.com>
# License: http://snmplabs.com/pyasn1/license.html
#
from sys import version_info

if version_info[0:2] < (2, 6):
    def bin(value):
        bitstring = []

        if value > 0:
            prefix = '0b'
        elif value < 0:
            prefix = '-0b'
            value = abs(value)
        else:
            prefix = '0b0'

        while value:
            if value & 1 == 1:
                bitstring.append('1')
            else:
                bitstring.append('0')

            value >>= 1

        bitstring.reverse()

        return prefix + ''.join(bitstring)
else:
    bin = bin
