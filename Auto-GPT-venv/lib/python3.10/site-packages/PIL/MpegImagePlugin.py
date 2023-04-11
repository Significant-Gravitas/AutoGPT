#
# The Python Imaging Library.
# $Id$
#
# MPEG file handling
#
# History:
#       95-09-09 fl     Created
#
# Copyright (c) Secret Labs AB 1997.
# Copyright (c) Fredrik Lundh 1995.
#
# See the README file for information on usage and redistribution.
#


from . import Image, ImageFile
from ._binary import i8

#
# Bitstream parser


class BitStream:
    def __init__(self, fp):
        self.fp = fp
        self.bits = 0
        self.bitbuffer = 0

    def next(self):
        return i8(self.fp.read(1))

    def peek(self, bits):
        while self.bits < bits:
            c = self.next()
            if c < 0:
                self.bits = 0
                continue
            self.bitbuffer = (self.bitbuffer << 8) + c
            self.bits += 8
        return self.bitbuffer >> (self.bits - bits) & (1 << bits) - 1

    def skip(self, bits):
        while self.bits < bits:
            self.bitbuffer = (self.bitbuffer << 8) + i8(self.fp.read(1))
            self.bits += 8
        self.bits = self.bits - bits

    def read(self, bits):
        v = self.peek(bits)
        self.bits = self.bits - bits
        return v


##
# Image plugin for MPEG streams.  This plugin can identify a stream,
# but it cannot read it.


class MpegImageFile(ImageFile.ImageFile):
    format = "MPEG"
    format_description = "MPEG"

    def _open(self):
        s = BitStream(self.fp)

        if s.read(32) != 0x1B3:
            msg = "not an MPEG file"
            raise SyntaxError(msg)

        self.mode = "RGB"
        self._size = s.read(12), s.read(12)


# --------------------------------------------------------------------
# Registry stuff

Image.register_open(MpegImageFile.format, MpegImageFile)

Image.register_extensions(MpegImageFile.format, [".mpg", ".mpeg"])

Image.register_mime(MpegImageFile.format, "video/mpeg")
