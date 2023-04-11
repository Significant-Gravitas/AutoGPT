#
# The Python Imaging Library.
# $Id$
#
# Basic McIdas support for PIL
#
# History:
# 1997-05-05 fl  Created (8-bit images only)
# 2009-03-08 fl  Added 16/32-bit support.
#
# Thanks to Richard Jones and Craig Swank for specs and samples.
#
# Copyright (c) Secret Labs AB 1997.
# Copyright (c) Fredrik Lundh 1997.
#
# See the README file for information on usage and redistribution.
#

import struct

from . import Image, ImageFile


def _accept(s):
    return s[:8] == b"\x00\x00\x00\x00\x00\x00\x00\x04"


##
# Image plugin for McIdas area images.


class McIdasImageFile(ImageFile.ImageFile):
    format = "MCIDAS"
    format_description = "McIdas area file"

    def _open(self):
        # parse area file directory
        s = self.fp.read(256)
        if not _accept(s) or len(s) != 256:
            msg = "not an McIdas area file"
            raise SyntaxError(msg)

        self.area_descriptor_raw = s
        self.area_descriptor = w = [0] + list(struct.unpack("!64i", s))

        # get mode
        if w[11] == 1:
            mode = rawmode = "L"
        elif w[11] == 2:
            # FIXME: add memory map support
            mode = "I"
            rawmode = "I;16B"
        elif w[11] == 4:
            # FIXME: add memory map support
            mode = "I"
            rawmode = "I;32B"
        else:
            msg = "unsupported McIdas format"
            raise SyntaxError(msg)

        self.mode = mode
        self._size = w[10], w[9]

        offset = w[34] + w[15]
        stride = w[15] + w[10] * w[11] * w[14]

        self.tile = [("raw", (0, 0) + self.size, offset, (rawmode, stride, 1))]


# --------------------------------------------------------------------
# registry

Image.register_open(McIdasImageFile.format, McIdasImageFile, _accept)

# no default extension
