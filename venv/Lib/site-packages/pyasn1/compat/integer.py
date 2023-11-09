#
# This file is part of pyasn1 software.
#
# Copyright (c) 2005-2019, Ilya Etingof <etingof@gmail.com>
# License: http://snmplabs.com/pyasn1/license.html
#
import sys

try:
    import platform

    implementation = platform.python_implementation()

except (ImportError, AttributeError):
    implementation = 'CPython'

from pyasn1.compat.octets import oct2int, null, ensureString

if sys.version_info[0:2] < (3, 2) or implementation != 'CPython':
    from binascii import a2b_hex, b2a_hex

    if sys.version_info[0] > 2:
        long = int

    def from_bytes(octets, signed=False):
        if not octets:
            return 0

        value = long(b2a_hex(ensureString(octets)), 16)

        if signed and oct2int(octets[0]) & 0x80:
            return value - (1 << len(octets) * 8)

        return value

    def to_bytes(value, signed=False, length=0):
        if value < 0:
            if signed:
                bits = bitLength(value)

                # two's complement form
                maxValue = 1 << bits
                valueToEncode = (value + maxValue) % maxValue

            else:
                raise OverflowError('can\'t convert negative int to unsigned')
        elif value == 0 and length == 0:
            return null
        else:
            bits = 0
            valueToEncode = value

        hexValue = hex(valueToEncode)[2:]
        if hexValue.endswith('L'):
            hexValue = hexValue[:-1]

        if len(hexValue) & 1:
            hexValue = '0' + hexValue

        # padding may be needed for two's complement encoding
        if value != valueToEncode or length:
            hexLength = len(hexValue) * 4

            padLength = max(length, bits)

            if padLength > hexLength:
                hexValue = '00' * ((padLength - hexLength - 1) // 8 + 1) + hexValue
            elif length and hexLength - length > 7:
                raise OverflowError('int too big to convert')

        firstOctet = int(hexValue[:2], 16)

        if signed:
            if firstOctet & 0x80:
                if value >= 0:
                    hexValue = '00' + hexValue
            elif value < 0:
                hexValue = 'ff' + hexValue

        octets_value = a2b_hex(hexValue)

        return octets_value

    def bitLength(number):
        # bits in unsigned number
        hexValue = hex(abs(number))
        bits = len(hexValue) - 2
        if hexValue.endswith('L'):
            bits -= 1
        if bits & 1:
            bits += 1
        bits *= 4
        # TODO: strip lhs zeros
        return bits

else:

    def from_bytes(octets, signed=False):
        return int.from_bytes(bytes(octets), 'big', signed=signed)

    def to_bytes(value, signed=False, length=0):
        length = max(value.bit_length(), length)

        if signed and length % 8 == 0:
            length += 1

        return value.to_bytes(length // 8 + (length % 8 and 1 or 0), 'big', signed=signed)

    def bitLength(number):
        return int(number).bit_length()
