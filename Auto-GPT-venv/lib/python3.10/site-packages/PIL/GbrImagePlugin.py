#
# The Python Imaging Library
#
# load a GIMP brush file
#
# History:
#       96-03-14 fl     Created
#       16-01-08 es     Version 2
#
# Copyright (c) Secret Labs AB 1997.
# Copyright (c) Fredrik Lundh 1996.
# Copyright (c) Eric Soroos 2016.
#
# See the README file for information on usage and redistribution.
#
#
# See https://github.com/GNOME/gimp/blob/mainline/devel-docs/gbr.txt for
# format documentation.
#
# This code Interprets version 1 and 2 .gbr files.
# Version 1 files are obsolete, and should not be used for new
#   brushes.
# Version 2 files are saved by GIMP v2.8 (at least)
# Version 3 files have a format specifier of 18 for 16bit floats in
#   the color depth field. This is currently unsupported by Pillow.

from . import Image, ImageFile
from ._binary import i32be as i32


def _accept(prefix):
    return len(prefix) >= 8 and i32(prefix, 0) >= 20 and i32(prefix, 4) in (1, 2)


##
# Image plugin for the GIMP brush format.


class GbrImageFile(ImageFile.ImageFile):
    format = "GBR"
    format_description = "GIMP brush file"

    def _open(self):
        header_size = i32(self.fp.read(4))
        if header_size < 20:
            msg = "not a GIMP brush"
            raise SyntaxError(msg)
        version = i32(self.fp.read(4))
        if version not in (1, 2):
            msg = f"Unsupported GIMP brush version: {version}"
            raise SyntaxError(msg)

        width = i32(self.fp.read(4))
        height = i32(self.fp.read(4))
        color_depth = i32(self.fp.read(4))
        if width <= 0 or height <= 0:
            msg = "not a GIMP brush"
            raise SyntaxError(msg)
        if color_depth not in (1, 4):
            msg = f"Unsupported GIMP brush color depth: {color_depth}"
            raise SyntaxError(msg)

        if version == 1:
            comment_length = header_size - 20
        else:
            comment_length = header_size - 28
            magic_number = self.fp.read(4)
            if magic_number != b"GIMP":
                msg = "not a GIMP brush, bad magic number"
                raise SyntaxError(msg)
            self.info["spacing"] = i32(self.fp.read(4))

        comment = self.fp.read(comment_length)[:-1]

        if color_depth == 1:
            self.mode = "L"
        else:
            self.mode = "RGBA"

        self._size = width, height

        self.info["comment"] = comment

        # Image might not be small
        Image._decompression_bomb_check(self.size)

        # Data is an uncompressed block of w * h * bytes/pixel
        self._data_size = width * height * color_depth

    def load(self):
        if not self.im:
            self.im = Image.core.new(self.mode, self.size)
            self.frombytes(self.fp.read(self._data_size))
        return Image.Image.load(self)


#
# registry


Image.register_open(GbrImageFile.format, GbrImageFile, _accept)
Image.register_extension(GbrImageFile.format, ".gbr")
