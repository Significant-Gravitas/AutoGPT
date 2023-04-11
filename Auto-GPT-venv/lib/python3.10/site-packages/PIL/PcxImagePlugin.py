#
# The Python Imaging Library.
# $Id$
#
# PCX file handling
#
# This format was originally used by ZSoft's popular PaintBrush
# program for the IBM PC.  It is also supported by many MS-DOS and
# Windows applications, including the Windows PaintBrush program in
# Windows 3.
#
# history:
# 1995-09-01 fl   Created
# 1996-05-20 fl   Fixed RGB support
# 1997-01-03 fl   Fixed 2-bit and 4-bit support
# 1999-02-03 fl   Fixed 8-bit support (broken in 1.0b1)
# 1999-02-07 fl   Added write support
# 2002-06-09 fl   Made 2-bit and 4-bit support a bit more robust
# 2002-07-30 fl   Seek from to current position, not beginning of file
# 2003-06-03 fl   Extract DPI settings (info["dpi"])
#
# Copyright (c) 1997-2003 by Secret Labs AB.
# Copyright (c) 1995-2003 by Fredrik Lundh.
#
# See the README file for information on usage and redistribution.
#

import io
import logging

from . import Image, ImageFile, ImagePalette
from ._binary import i16le as i16
from ._binary import o8
from ._binary import o16le as o16

logger = logging.getLogger(__name__)


def _accept(prefix):
    return prefix[0] == 10 and prefix[1] in [0, 2, 3, 5]


##
# Image plugin for Paintbrush images.


class PcxImageFile(ImageFile.ImageFile):
    format = "PCX"
    format_description = "Paintbrush"

    def _open(self):
        # header
        s = self.fp.read(128)
        if not _accept(s):
            msg = "not a PCX file"
            raise SyntaxError(msg)

        # image
        bbox = i16(s, 4), i16(s, 6), i16(s, 8) + 1, i16(s, 10) + 1
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            msg = "bad PCX image size"
            raise SyntaxError(msg)
        logger.debug("BBox: %s %s %s %s", *bbox)

        # format
        version = s[1]
        bits = s[3]
        planes = s[65]
        provided_stride = i16(s, 66)
        logger.debug(
            "PCX version %s, bits %s, planes %s, stride %s",
            version,
            bits,
            planes,
            provided_stride,
        )

        self.info["dpi"] = i16(s, 12), i16(s, 14)

        if bits == 1 and planes == 1:
            mode = rawmode = "1"

        elif bits == 1 and planes in (2, 4):
            mode = "P"
            rawmode = "P;%dL" % planes
            self.palette = ImagePalette.raw("RGB", s[16:64])

        elif version == 5 and bits == 8 and planes == 1:
            mode = rawmode = "L"
            # FIXME: hey, this doesn't work with the incremental loader !!!
            self.fp.seek(-769, io.SEEK_END)
            s = self.fp.read(769)
            if len(s) == 769 and s[0] == 12:
                # check if the palette is linear greyscale
                for i in range(256):
                    if s[i * 3 + 1 : i * 3 + 4] != o8(i) * 3:
                        mode = rawmode = "P"
                        break
                if mode == "P":
                    self.palette = ImagePalette.raw("RGB", s[1:])
            self.fp.seek(128)

        elif version == 5 and bits == 8 and planes == 3:
            mode = "RGB"
            rawmode = "RGB;L"

        else:
            msg = "unknown PCX mode"
            raise OSError(msg)

        self.mode = mode
        self._size = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # Don't trust the passed in stride.
        # Calculate the approximate position for ourselves.
        # CVE-2020-35653
        stride = (self._size[0] * bits + 7) // 8

        # While the specification states that this must be even,
        # not all images follow this
        if provided_stride != stride:
            stride += stride % 2

        bbox = (0, 0) + self.size
        logger.debug("size: %sx%s", *self.size)

        self.tile = [("pcx", bbox, self.fp.tell(), (rawmode, planes * stride))]


# --------------------------------------------------------------------
# save PCX files


SAVE = {
    # mode: (version, bits, planes, raw mode)
    "1": (2, 1, 1, "1"),
    "L": (5, 8, 1, "L"),
    "P": (5, 8, 1, "P"),
    "RGB": (5, 8, 3, "RGB;L"),
}


def _save(im, fp, filename):
    try:
        version, bits, planes, rawmode = SAVE[im.mode]
    except KeyError as e:
        msg = f"Cannot save {im.mode} images as PCX"
        raise ValueError(msg) from e

    # bytes per plane
    stride = (im.size[0] * bits + 7) // 8
    # stride should be even
    stride += stride % 2
    # Stride needs to be kept in sync with the PcxEncode.c version.
    # Ideally it should be passed in in the state, but the bytes value
    # gets overwritten.

    logger.debug(
        "PcxImagePlugin._save: xwidth: %d, bits: %d, stride: %d",
        im.size[0],
        bits,
        stride,
    )

    # under windows, we could determine the current screen size with
    # "Image.core.display_mode()[1]", but I think that's overkill...

    screen = im.size

    dpi = 100, 100

    # PCX header
    fp.write(
        o8(10)
        + o8(version)
        + o8(1)
        + o8(bits)
        + o16(0)
        + o16(0)
        + o16(im.size[0] - 1)
        + o16(im.size[1] - 1)
        + o16(dpi[0])
        + o16(dpi[1])
        + b"\0" * 24
        + b"\xFF" * 24
        + b"\0"
        + o8(planes)
        + o16(stride)
        + o16(1)
        + o16(screen[0])
        + o16(screen[1])
        + b"\0" * 54
    )

    assert fp.tell() == 128

    ImageFile._save(im, fp, [("pcx", (0, 0) + im.size, 0, (rawmode, bits * planes))])

    if im.mode == "P":
        # colour palette
        fp.write(o8(12))
        palette = im.im.getpalette("RGB", "RGB")
        palette += b"\x00" * (768 - len(palette))
        fp.write(palette)  # 768 bytes
    elif im.mode == "L":
        # greyscale palette
        fp.write(o8(12))
        for i in range(256):
            fp.write(o8(i) * 3)


# --------------------------------------------------------------------
# registry


Image.register_open(PcxImageFile.format, PcxImageFile, _accept)
Image.register_save(PcxImageFile.format, _save)

Image.register_extension(PcxImageFile.format, ".pcx")

Image.register_mime(PcxImageFile.format, "image/x-pcx")
