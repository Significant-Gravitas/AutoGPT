#
# The Python Imaging Library.
# $Id$
#
# XV Thumbnail file handler by Charles E. "Gene" Cash
# (gcash@magicnet.net)
#
# see xvcolor.c and xvbrowse.c in the sources to John Bradley's XV,
# available from ftp://ftp.cis.upenn.edu/pub/xv/
#
# history:
# 98-08-15 cec  created (b/w only)
# 98-12-09 cec  added color palette
# 98-12-28 fl   added to PIL (with only a few very minor modifications)
#
# To do:
# FIXME: make save work (this requires quantization support)
#

from . import Image, ImageFile, ImagePalette
from ._binary import o8

_MAGIC = b"P7 332"

# standard color palette for thumbnails (RGB332)
PALETTE = b""
for r in range(8):
    for g in range(8):
        for b in range(4):
            PALETTE = PALETTE + (
                o8((r * 255) // 7) + o8((g * 255) // 7) + o8((b * 255) // 3)
            )


def _accept(prefix):
    return prefix[:6] == _MAGIC


##
# Image plugin for XV thumbnail images.


class XVThumbImageFile(ImageFile.ImageFile):
    format = "XVThumb"
    format_description = "XV thumbnail image"

    def _open(self):
        # check magic
        if not _accept(self.fp.read(6)):
            msg = "not an XV thumbnail file"
            raise SyntaxError(msg)

        # Skip to beginning of next line
        self.fp.readline()

        # skip info comments
        while True:
            s = self.fp.readline()
            if not s:
                msg = "Unexpected EOF reading XV thumbnail file"
                raise SyntaxError(msg)
            if s[0] != 35:  # ie. when not a comment: '#'
                break

        # parse header line (already read)
        s = s.strip().split()

        self.mode = "P"
        self._size = int(s[0]), int(s[1])

        self.palette = ImagePalette.raw("RGB", PALETTE)

        self.tile = [("raw", (0, 0) + self.size, self.fp.tell(), (self.mode, 0, 1))]


# --------------------------------------------------------------------

Image.register_open(XVThumbImageFile.format, XVThumbImageFile, _accept)
