#
# THIS IS WORK IN PROGRESS
#
# The Python Imaging Library
# $Id$
#
# portable compiled font file parser
#
# history:
# 1997-08-19 fl   created
# 2003-09-13 fl   fixed loading of unicode fonts
#
# Copyright (c) 1997-2003 by Secret Labs AB.
# Copyright (c) 1997-2003 by Fredrik Lundh.
#
# See the README file for information on usage and redistribution.
#

import io

from . import FontFile, Image
from ._binary import i8
from ._binary import i16be as b16
from ._binary import i16le as l16
from ._binary import i32be as b32
from ._binary import i32le as l32

# --------------------------------------------------------------------
# declarations

PCF_MAGIC = 0x70636601  # "\x01fcp"

PCF_PROPERTIES = 1 << 0
PCF_ACCELERATORS = 1 << 1
PCF_METRICS = 1 << 2
PCF_BITMAPS = 1 << 3
PCF_INK_METRICS = 1 << 4
PCF_BDF_ENCODINGS = 1 << 5
PCF_SWIDTHS = 1 << 6
PCF_GLYPH_NAMES = 1 << 7
PCF_BDF_ACCELERATORS = 1 << 8

BYTES_PER_ROW = [
    lambda bits: ((bits + 7) >> 3),
    lambda bits: ((bits + 15) >> 3) & ~1,
    lambda bits: ((bits + 31) >> 3) & ~3,
    lambda bits: ((bits + 63) >> 3) & ~7,
]


def sz(s, o):
    return s[o : s.index(b"\0", o)]


class PcfFontFile(FontFile.FontFile):
    """Font file plugin for the X11 PCF format."""

    name = "name"

    def __init__(self, fp, charset_encoding="iso8859-1"):
        self.charset_encoding = charset_encoding

        magic = l32(fp.read(4))
        if magic != PCF_MAGIC:
            msg = "not a PCF file"
            raise SyntaxError(msg)

        super().__init__()

        count = l32(fp.read(4))
        self.toc = {}
        for i in range(count):
            type = l32(fp.read(4))
            self.toc[type] = l32(fp.read(4)), l32(fp.read(4)), l32(fp.read(4))

        self.fp = fp

        self.info = self._load_properties()

        metrics = self._load_metrics()
        bitmaps = self._load_bitmaps(metrics)
        encoding = self._load_encoding()

        #
        # create glyph structure

        for ch, ix in enumerate(encoding):
            if ix is not None:
                (
                    xsize,
                    ysize,
                    left,
                    right,
                    width,
                    ascent,
                    descent,
                    attributes,
                ) = metrics[ix]
                self.glyph[ch] = (
                    (width, 0),
                    (left, descent - ysize, xsize + left, descent),
                    (0, 0, xsize, ysize),
                    bitmaps[ix],
                )

    def _getformat(self, tag):
        format, size, offset = self.toc[tag]

        fp = self.fp
        fp.seek(offset)

        format = l32(fp.read(4))

        if format & 4:
            i16, i32 = b16, b32
        else:
            i16, i32 = l16, l32

        return fp, format, i16, i32

    def _load_properties(self):
        #
        # font properties

        properties = {}

        fp, format, i16, i32 = self._getformat(PCF_PROPERTIES)

        nprops = i32(fp.read(4))

        # read property description
        p = []
        for i in range(nprops):
            p.append((i32(fp.read(4)), i8(fp.read(1)), i32(fp.read(4))))
        if nprops & 3:
            fp.seek(4 - (nprops & 3), io.SEEK_CUR)  # pad

        data = fp.read(i32(fp.read(4)))

        for k, s, v in p:
            k = sz(data, k)
            if s:
                v = sz(data, v)
            properties[k] = v

        return properties

    def _load_metrics(self):
        #
        # font metrics

        metrics = []

        fp, format, i16, i32 = self._getformat(PCF_METRICS)

        append = metrics.append

        if (format & 0xFF00) == 0x100:
            # "compressed" metrics
            for i in range(i16(fp.read(2))):
                left = i8(fp.read(1)) - 128
                right = i8(fp.read(1)) - 128
                width = i8(fp.read(1)) - 128
                ascent = i8(fp.read(1)) - 128
                descent = i8(fp.read(1)) - 128
                xsize = right - left
                ysize = ascent + descent
                append((xsize, ysize, left, right, width, ascent, descent, 0))

        else:
            # "jumbo" metrics
            for i in range(i32(fp.read(4))):
                left = i16(fp.read(2))
                right = i16(fp.read(2))
                width = i16(fp.read(2))
                ascent = i16(fp.read(2))
                descent = i16(fp.read(2))
                attributes = i16(fp.read(2))
                xsize = right - left
                ysize = ascent + descent
                append((xsize, ysize, left, right, width, ascent, descent, attributes))

        return metrics

    def _load_bitmaps(self, metrics):
        #
        # bitmap data

        bitmaps = []

        fp, format, i16, i32 = self._getformat(PCF_BITMAPS)

        nbitmaps = i32(fp.read(4))

        if nbitmaps != len(metrics):
            msg = "Wrong number of bitmaps"
            raise OSError(msg)

        offsets = []
        for i in range(nbitmaps):
            offsets.append(i32(fp.read(4)))

        bitmap_sizes = []
        for i in range(4):
            bitmap_sizes.append(i32(fp.read(4)))

        # byteorder = format & 4  # non-zero => MSB
        bitorder = format & 8  # non-zero => MSB
        padindex = format & 3

        bitmapsize = bitmap_sizes[padindex]
        offsets.append(bitmapsize)

        data = fp.read(bitmapsize)

        pad = BYTES_PER_ROW[padindex]
        mode = "1;R"
        if bitorder:
            mode = "1"

        for i in range(nbitmaps):
            xsize, ysize = metrics[i][:2]
            b, e = offsets[i : i + 2]
            bitmaps.append(
                Image.frombytes("1", (xsize, ysize), data[b:e], "raw", mode, pad(xsize))
            )

        return bitmaps

    def _load_encoding(self):
        fp, format, i16, i32 = self._getformat(PCF_BDF_ENCODINGS)

        first_col, last_col = i16(fp.read(2)), i16(fp.read(2))
        first_row, last_row = i16(fp.read(2)), i16(fp.read(2))

        i16(fp.read(2))  # default

        nencoding = (last_col - first_col + 1) * (last_row - first_row + 1)

        # map character code to bitmap index
        encoding = [None] * min(256, nencoding)

        encoding_offsets = [i16(fp.read(2)) for _ in range(nencoding)]

        for i in range(first_col, len(encoding)):
            try:
                encoding_offset = encoding_offsets[
                    ord(bytearray([i]).decode(self.charset_encoding))
                ]
                if encoding_offset != 0xFFFF:
                    encoding[i] = encoding_offset
            except UnicodeDecodeError:
                # character is not supported in selected encoding
                pass

        return encoding
