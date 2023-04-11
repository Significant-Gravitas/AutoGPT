#
# The Python Imaging Library.
# $Id$
#
# XBM File handling
#
# History:
# 1995-09-08 fl   Created
# 1996-11-01 fl   Added save support
# 1997-07-07 fl   Made header parser more tolerant
# 1997-07-22 fl   Fixed yet another parser bug
# 2001-02-17 fl   Use 're' instead of 'regex' (Python 2.1) (0.4)
# 2001-05-13 fl   Added hotspot handling (based on code from Bernhard Herzog)
# 2004-02-24 fl   Allow some whitespace before first #define
#
# Copyright (c) 1997-2004 by Secret Labs AB
# Copyright (c) 1996-1997 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#

import re

from . import Image, ImageFile

# XBM header
xbm_head = re.compile(
    rb"\s*#define[ \t]+.*_width[ \t]+(?P<width>[0-9]+)[\r\n]+"
    b"#define[ \t]+.*_height[ \t]+(?P<height>[0-9]+)[\r\n]+"
    b"(?P<hotspot>"
    b"#define[ \t]+[^_]*_x_hot[ \t]+(?P<xhot>[0-9]+)[\r\n]+"
    b"#define[ \t]+[^_]*_y_hot[ \t]+(?P<yhot>[0-9]+)[\r\n]+"
    b")?"
    rb"[\000-\377]*_bits\[]"
)


def _accept(prefix):
    return prefix.lstrip()[:7] == b"#define"


##
# Image plugin for X11 bitmaps.


class XbmImageFile(ImageFile.ImageFile):
    format = "XBM"
    format_description = "X11 Bitmap"

    def _open(self):
        m = xbm_head.match(self.fp.read(512))

        if not m:
            msg = "not a XBM file"
            raise SyntaxError(msg)

        xsize = int(m.group("width"))
        ysize = int(m.group("height"))

        if m.group("hotspot"):
            self.info["hotspot"] = (int(m.group("xhot")), int(m.group("yhot")))

        self.mode = "1"
        self._size = xsize, ysize

        self.tile = [("xbm", (0, 0) + self.size, m.end(), None)]


def _save(im, fp, filename):
    if im.mode != "1":
        msg = f"cannot write mode {im.mode} as XBM"
        raise OSError(msg)

    fp.write(f"#define im_width {im.size[0]}\n".encode("ascii"))
    fp.write(f"#define im_height {im.size[1]}\n".encode("ascii"))

    hotspot = im.encoderinfo.get("hotspot")
    if hotspot:
        fp.write(f"#define im_x_hot {hotspot[0]}\n".encode("ascii"))
        fp.write(f"#define im_y_hot {hotspot[1]}\n".encode("ascii"))

    fp.write(b"static char im_bits[] = {\n")

    ImageFile._save(im, fp, [("xbm", (0, 0) + im.size, 0, None)])

    fp.write(b"};\n")


Image.register_open(XbmImageFile.format, XbmImageFile, _accept)
Image.register_save(XbmImageFile.format, _save)

Image.register_extension(XbmImageFile.format, ".xbm")

Image.register_mime(XbmImageFile.format, "image/xbm")
