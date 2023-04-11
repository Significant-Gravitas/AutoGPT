#
# The Python Imaging Library.
# $Id$
#
# Binary input/output support routines.
#
# Copyright (c) 1997-2003 by Secret Labs AB
# Copyright (c) 1995-2003 by Fredrik Lundh
# Copyright (c) 2012 by Brian Crowell
#
# See the README file for information on usage and redistribution.
#


"""Binary input/output support routines."""


from struct import pack, unpack_from


def i8(c):
    return c if c.__class__ is int else c[0]


def o8(i):
    return bytes((i & 255,))


# Input, le = little endian, be = big endian
def i16le(c, o=0):
    """
    Converts a 2-bytes (16 bits) string to an unsigned integer.

    :param c: string containing bytes to convert
    :param o: offset of bytes to convert in string
    """
    return unpack_from("<H", c, o)[0]


def si16le(c, o=0):
    """
    Converts a 2-bytes (16 bits) string to a signed integer.

    :param c: string containing bytes to convert
    :param o: offset of bytes to convert in string
    """
    return unpack_from("<h", c, o)[0]


def si16be(c, o=0):
    """
    Converts a 2-bytes (16 bits) string to a signed integer, big endian.

    :param c: string containing bytes to convert
    :param o: offset of bytes to convert in string
    """
    return unpack_from(">h", c, o)[0]


def i32le(c, o=0):
    """
    Converts a 4-bytes (32 bits) string to an unsigned integer.

    :param c: string containing bytes to convert
    :param o: offset of bytes to convert in string
    """
    return unpack_from("<I", c, o)[0]


def si32le(c, o=0):
    """
    Converts a 4-bytes (32 bits) string to a signed integer.

    :param c: string containing bytes to convert
    :param o: offset of bytes to convert in string
    """
    return unpack_from("<i", c, o)[0]


def i16be(c, o=0):
    return unpack_from(">H", c, o)[0]


def i32be(c, o=0):
    return unpack_from(">I", c, o)[0]


# Output, le = little endian, be = big endian
def o16le(i):
    return pack("<H", i)


def o32le(i):
    return pack("<I", i)


def o16be(i):
    return pack(">H", i)


def o32be(i):
    return pack(">I", i)
