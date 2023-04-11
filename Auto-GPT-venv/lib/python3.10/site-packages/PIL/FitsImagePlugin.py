#
# The Python Imaging Library
# $Id$
#
# FITS file handling
#
# Copyright (c) 1998-2003 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#

import math

from . import Image, ImageFile


def _accept(prefix):
    return prefix[:6] == b"SIMPLE"


class FitsImageFile(ImageFile.ImageFile):
    format = "FITS"
    format_description = "FITS"

    def _open(self):
        headers = {}
        while True:
            header = self.fp.read(80)
            if not header:
                msg = "Truncated FITS file"
                raise OSError(msg)
            keyword = header[:8].strip()
            if keyword == b"END":
                break
            value = header[8:].split(b"/")[0].strip()
            if value.startswith(b"="):
                value = value[1:].strip()
            if not headers and (not _accept(keyword) or value != b"T"):
                msg = "Not a FITS file"
                raise SyntaxError(msg)
            headers[keyword] = value

        naxis = int(headers[b"NAXIS"])
        if naxis == 0:
            msg = "No image data"
            raise ValueError(msg)
        elif naxis == 1:
            self._size = 1, int(headers[b"NAXIS1"])
        else:
            self._size = int(headers[b"NAXIS1"]), int(headers[b"NAXIS2"])

        number_of_bits = int(headers[b"BITPIX"])
        if number_of_bits == 8:
            self.mode = "L"
        elif number_of_bits == 16:
            self.mode = "I"
            # rawmode = "I;16S"
        elif number_of_bits == 32:
            self.mode = "I"
        elif number_of_bits in (-32, -64):
            self.mode = "F"
            # rawmode = "F" if number_of_bits == -32 else "F;64F"

        offset = math.ceil(self.fp.tell() / 2880) * 2880
        self.tile = [("raw", (0, 0) + self.size, offset, (self.mode, 0, -1))]


# --------------------------------------------------------------------
# Registry

Image.register_open(FitsImageFile.format, FitsImageFile, _accept)

Image.register_extensions(FitsImageFile.format, [".fit", ".fits"])
