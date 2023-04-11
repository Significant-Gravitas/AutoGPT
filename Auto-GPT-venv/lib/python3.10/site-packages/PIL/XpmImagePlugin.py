#
# The Python Imaging Library.
# $Id$
#
# XPM File handling
#
# History:
# 1996-12-29 fl   Created
# 2001-02-17 fl   Use 're' instead of 'regex' (Python 2.1) (0.7)
#
# Copyright (c) Secret Labs AB 1997-2001.
# Copyright (c) Fredrik Lundh 1996-2001.
#
# See the README file for information on usage and redistribution.
#


import re

from . import Image, ImageFile, ImagePalette
from ._binary import o8

# XPM header
xpm_head = re.compile(b'"([0-9]*) ([0-9]*) ([0-9]*) ([0-9]*)')


def _accept(prefix):
    return prefix[:9] == b"/* XPM */"


##
# Image plugin for X11 pixel maps.


class XpmImageFile(ImageFile.ImageFile):
    format = "XPM"
    format_description = "X11 Pixel Map"

    def _open(self):
        if not _accept(self.fp.read(9)):
            msg = "not an XPM file"
            raise SyntaxError(msg)

        # skip forward to next string
        while True:
            s = self.fp.readline()
            if not s:
                msg = "broken XPM file"
                raise SyntaxError(msg)
            m = xpm_head.match(s)
            if m:
                break

        self._size = int(m.group(1)), int(m.group(2))

        pal = int(m.group(3))
        bpp = int(m.group(4))

        if pal > 256 or bpp != 1:
            msg = "cannot read this XPM file"
            raise ValueError(msg)

        #
        # load palette description

        palette = [b"\0\0\0"] * 256

        for _ in range(pal):
            s = self.fp.readline()
            if s[-2:] == b"\r\n":
                s = s[:-2]
            elif s[-1:] in b"\r\n":
                s = s[:-1]

            c = s[1]
            s = s[2:-2].split()

            for i in range(0, len(s), 2):
                if s[i] == b"c":
                    # process colour key
                    rgb = s[i + 1]
                    if rgb == b"None":
                        self.info["transparency"] = c
                    elif rgb[:1] == b"#":
                        # FIXME: handle colour names (see ImagePalette.py)
                        rgb = int(rgb[1:], 16)
                        palette[c] = (
                            o8((rgb >> 16) & 255) + o8((rgb >> 8) & 255) + o8(rgb & 255)
                        )
                    else:
                        # unknown colour
                        msg = "cannot read this XPM file"
                        raise ValueError(msg)
                    break

            else:
                # missing colour key
                msg = "cannot read this XPM file"
                raise ValueError(msg)

        self.mode = "P"
        self.palette = ImagePalette.raw("RGB", b"".join(palette))

        self.tile = [("raw", (0, 0) + self.size, self.fp.tell(), ("P", 0, 1))]

    def load_read(self, bytes):
        #
        # load all image data in one chunk

        xsize, ysize = self.size

        s = [None] * ysize

        for i in range(ysize):
            s[i] = self.fp.readline()[1 : xsize + 1].ljust(xsize)

        return b"".join(s)


#
# Registry


Image.register_open(XpmImageFile.format, XpmImageFile, _accept)

Image.register_extension(XpmImageFile.format, ".xpm")

Image.register_mime(XpmImageFile.format, "image/xpm")
