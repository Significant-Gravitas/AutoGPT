#
# The Python Imaging Library.
# $Id$
#
# TGA file handling
#
# History:
# 95-09-01 fl   created (reads 24-bit files only)
# 97-01-04 fl   support more TGA versions, including compressed images
# 98-07-04 fl   fixed orientation and alpha layer bugs
# 98-09-11 fl   fixed orientation for runlength decoder
#
# Copyright (c) Secret Labs AB 1997-98.
# Copyright (c) Fredrik Lundh 1995-97.
#
# See the README file for information on usage and redistribution.
#


import warnings

from . import Image, ImageFile, ImagePalette
from ._binary import i16le as i16
from ._binary import o8
from ._binary import o16le as o16

#
# --------------------------------------------------------------------
# Read RGA file


MODES = {
    # map imagetype/depth to rawmode
    (1, 8): "P",
    (3, 1): "1",
    (3, 8): "L",
    (3, 16): "LA",
    (2, 16): "BGR;5",
    (2, 24): "BGR",
    (2, 32): "BGRA",
}


##
# Image plugin for Targa files.


class TgaImageFile(ImageFile.ImageFile):
    format = "TGA"
    format_description = "Targa"

    def _open(self):
        # process header
        s = self.fp.read(18)

        id_len = s[0]

        colormaptype = s[1]
        imagetype = s[2]

        depth = s[16]

        flags = s[17]

        self._size = i16(s, 12), i16(s, 14)

        # validate header fields
        if (
            colormaptype not in (0, 1)
            or self.size[0] <= 0
            or self.size[1] <= 0
            or depth not in (1, 8, 16, 24, 32)
        ):
            msg = "not a TGA file"
            raise SyntaxError(msg)

        # image mode
        if imagetype in (3, 11):
            self.mode = "L"
            if depth == 1:
                self.mode = "1"  # ???
            elif depth == 16:
                self.mode = "LA"
        elif imagetype in (1, 9):
            self.mode = "P"
        elif imagetype in (2, 10):
            self.mode = "RGB"
            if depth == 32:
                self.mode = "RGBA"
        else:
            msg = "unknown TGA mode"
            raise SyntaxError(msg)

        # orientation
        orientation = flags & 0x30
        self._flip_horizontally = orientation in [0x10, 0x30]
        if orientation in [0x20, 0x30]:
            orientation = 1
        elif orientation in [0, 0x10]:
            orientation = -1
        else:
            msg = "unknown TGA orientation"
            raise SyntaxError(msg)

        self.info["orientation"] = orientation

        if imagetype & 8:
            self.info["compression"] = "tga_rle"

        if id_len:
            self.info["id_section"] = self.fp.read(id_len)

        if colormaptype:
            # read palette
            start, size, mapdepth = i16(s, 3), i16(s, 5), s[7]
            if mapdepth == 16:
                self.palette = ImagePalette.raw(
                    "BGR;15", b"\0" * 2 * start + self.fp.read(2 * size)
                )
            elif mapdepth == 24:
                self.palette = ImagePalette.raw(
                    "BGR", b"\0" * 3 * start + self.fp.read(3 * size)
                )
            elif mapdepth == 32:
                self.palette = ImagePalette.raw(
                    "BGRA", b"\0" * 4 * start + self.fp.read(4 * size)
                )

        # setup tile descriptor
        try:
            rawmode = MODES[(imagetype & 7, depth)]
            if imagetype & 8:
                # compressed
                self.tile = [
                    (
                        "tga_rle",
                        (0, 0) + self.size,
                        self.fp.tell(),
                        (rawmode, orientation, depth),
                    )
                ]
            else:
                self.tile = [
                    (
                        "raw",
                        (0, 0) + self.size,
                        self.fp.tell(),
                        (rawmode, 0, orientation),
                    )
                ]
        except KeyError:
            pass  # cannot decode

    def load_end(self):
        if self._flip_horizontally:
            self.im = self.im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)


#
# --------------------------------------------------------------------
# Write TGA file


SAVE = {
    "1": ("1", 1, 0, 3),
    "L": ("L", 8, 0, 3),
    "LA": ("LA", 16, 0, 3),
    "P": ("P", 8, 1, 1),
    "RGB": ("BGR", 24, 0, 2),
    "RGBA": ("BGRA", 32, 0, 2),
}


def _save(im, fp, filename):
    try:
        rawmode, bits, colormaptype, imagetype = SAVE[im.mode]
    except KeyError as e:
        msg = f"cannot write mode {im.mode} as TGA"
        raise OSError(msg) from e

    if "rle" in im.encoderinfo:
        rle = im.encoderinfo["rle"]
    else:
        compression = im.encoderinfo.get("compression", im.info.get("compression"))
        rle = compression == "tga_rle"
    if rle:
        imagetype += 8

    id_section = im.encoderinfo.get("id_section", im.info.get("id_section", ""))
    id_len = len(id_section)
    if id_len > 255:
        id_len = 255
        id_section = id_section[:255]
        warnings.warn("id_section has been trimmed to 255 characters")

    if colormaptype:
        palette = im.im.getpalette("RGB", "BGR")
        colormaplength, colormapentry = len(palette) // 3, 24
    else:
        colormaplength, colormapentry = 0, 0

    if im.mode in ("LA", "RGBA"):
        flags = 8
    else:
        flags = 0

    orientation = im.encoderinfo.get("orientation", im.info.get("orientation", -1))
    if orientation > 0:
        flags = flags | 0x20

    fp.write(
        o8(id_len)
        + o8(colormaptype)
        + o8(imagetype)
        + o16(0)  # colormapfirst
        + o16(colormaplength)
        + o8(colormapentry)
        + o16(0)
        + o16(0)
        + o16(im.size[0])
        + o16(im.size[1])
        + o8(bits)
        + o8(flags)
    )

    if id_section:
        fp.write(id_section)

    if colormaptype:
        fp.write(palette)

    if rle:
        ImageFile._save(
            im, fp, [("tga_rle", (0, 0) + im.size, 0, (rawmode, orientation))]
        )
    else:
        ImageFile._save(
            im, fp, [("raw", (0, 0) + im.size, 0, (rawmode, 0, orientation))]
        )

    # write targa version 2 footer
    fp.write(b"\000" * 8 + b"TRUEVISION-XFILE." + b"\000")


#
# --------------------------------------------------------------------
# Registry


Image.register_open(TgaImageFile.format, TgaImageFile)
Image.register_save(TgaImageFile.format, _save)

Image.register_extensions(TgaImageFile.format, [".tga", ".icb", ".vda", ".vst"])

Image.register_mime(TgaImageFile.format, "image/x-tga")
