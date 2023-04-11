#
# The Python Imaging Library
# $Id$
#
# WMF stub codec
#
# history:
# 1996-12-14 fl   Created
# 2004-02-22 fl   Turned into a stub driver
# 2004-02-23 fl   Added EMF support
#
# Copyright (c) Secret Labs AB 1997-2004.  All rights reserved.
# Copyright (c) Fredrik Lundh 1996.
#
# See the README file for information on usage and redistribution.
#
# WMF/EMF reference documentation:
# https://winprotocoldoc.blob.core.windows.net/productionwindowsarchives/MS-WMF/[MS-WMF].pdf
# http://wvware.sourceforge.net/caolan/index.html
# http://wvware.sourceforge.net/caolan/ora-wmf.html

from . import Image, ImageFile
from ._binary import i16le as word
from ._binary import si16le as short
from ._binary import si32le as _long

_handler = None


def register_handler(handler):
    """
    Install application-specific WMF image handler.

    :param handler: Handler object.
    """
    global _handler
    _handler = handler


if hasattr(Image.core, "drawwmf"):
    # install default handler (windows only)

    class WmfHandler:
        def open(self, im):
            im.mode = "RGB"
            self.bbox = im.info["wmf_bbox"]

        def load(self, im):
            im.fp.seek(0)  # rewind
            return Image.frombytes(
                "RGB",
                im.size,
                Image.core.drawwmf(im.fp.read(), im.size, self.bbox),
                "raw",
                "BGR",
                (im.size[0] * 3 + 3) & -4,
                -1,
            )

    register_handler(WmfHandler())

#
# --------------------------------------------------------------------
# Read WMF file


def _accept(prefix):
    return (
        prefix[:6] == b"\xd7\xcd\xc6\x9a\x00\x00" or prefix[:4] == b"\x01\x00\x00\x00"
    )


##
# Image plugin for Windows metafiles.


class WmfStubImageFile(ImageFile.StubImageFile):
    format = "WMF"
    format_description = "Windows Metafile"

    def _open(self):
        self._inch = None

        # check placable header
        s = self.fp.read(80)

        if s[:6] == b"\xd7\xcd\xc6\x9a\x00\x00":
            # placeable windows metafile

            # get units per inch
            self._inch = word(s, 14)

            # get bounding box
            x0 = short(s, 6)
            y0 = short(s, 8)
            x1 = short(s, 10)
            y1 = short(s, 12)

            # normalize size to 72 dots per inch
            self.info["dpi"] = 72
            size = (
                (x1 - x0) * self.info["dpi"] // self._inch,
                (y1 - y0) * self.info["dpi"] // self._inch,
            )

            self.info["wmf_bbox"] = x0, y0, x1, y1

            # sanity check (standard metafile header)
            if s[22:26] != b"\x01\x00\t\x00":
                msg = "Unsupported WMF file format"
                raise SyntaxError(msg)

        elif s[:4] == b"\x01\x00\x00\x00" and s[40:44] == b" EMF":
            # enhanced metafile

            # get bounding box
            x0 = _long(s, 8)
            y0 = _long(s, 12)
            x1 = _long(s, 16)
            y1 = _long(s, 20)

            # get frame (in 0.01 millimeter units)
            frame = _long(s, 24), _long(s, 28), _long(s, 32), _long(s, 36)

            size = x1 - x0, y1 - y0

            # calculate dots per inch from bbox and frame
            xdpi = 2540.0 * (x1 - y0) / (frame[2] - frame[0])
            ydpi = 2540.0 * (y1 - y0) / (frame[3] - frame[1])

            self.info["wmf_bbox"] = x0, y0, x1, y1

            if xdpi == ydpi:
                self.info["dpi"] = xdpi
            else:
                self.info["dpi"] = xdpi, ydpi

        else:
            msg = "Unsupported file format"
            raise SyntaxError(msg)

        self.mode = "RGB"
        self._size = size

        loader = self._load()
        if loader:
            loader.open(self)

    def _load(self):
        return _handler

    def load(self, dpi=None):
        if dpi is not None and self._inch is not None:
            self.info["dpi"] = dpi
            x0, y0, x1, y1 = self.info["wmf_bbox"]
            self._size = (
                (x1 - x0) * self.info["dpi"] // self._inch,
                (y1 - y0) * self.info["dpi"] // self._inch,
            )
        return super().load()


def _save(im, fp, filename):
    if _handler is None or not hasattr(_handler, "save"):
        msg = "WMF save handler not installed"
        raise OSError(msg)
    _handler.save(im, fp, filename)


#
# --------------------------------------------------------------------
# Registry stuff


Image.register_open(WmfStubImageFile.format, WmfStubImageFile, _accept)
Image.register_save(WmfStubImageFile.format, _save)

Image.register_extensions(WmfStubImageFile.format, [".wmf", ".emf"])
