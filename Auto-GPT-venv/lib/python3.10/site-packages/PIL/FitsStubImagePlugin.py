#
# The Python Imaging Library
# $Id$
#
# FITS stub adapter
#
# Copyright (c) 1998-2003 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#

from . import FitsImagePlugin, Image, ImageFile
from ._deprecate import deprecate

_handler = None


def register_handler(handler):
    """
    Install application-specific FITS image handler.

    :param handler: Handler object.
    """
    global _handler
    _handler = handler

    deprecate(
        "FitsStubImagePlugin",
        10,
        action="FITS images can now be read without "
        "a handler through FitsImagePlugin instead",
    )

    # Override FitsImagePlugin with this handler
    # for backwards compatibility
    try:
        Image.ID.remove(FITSStubImageFile.format)
    except ValueError:
        pass

    Image.register_open(
        FITSStubImageFile.format, FITSStubImageFile, FitsImagePlugin._accept
    )


class FITSStubImageFile(ImageFile.StubImageFile):
    format = FitsImagePlugin.FitsImageFile.format
    format_description = FitsImagePlugin.FitsImageFile.format_description

    def _open(self):
        offset = self.fp.tell()

        im = FitsImagePlugin.FitsImageFile(self.fp)
        self._size = im.size
        self.mode = im.mode
        self.tile = []

        self.fp.seek(offset)

        loader = self._load()
        if loader:
            loader.open(self)

    def _load(self):
        return _handler


def _save(im, fp, filename):
    msg = "FITS save handler not installed"
    raise OSError(msg)


# --------------------------------------------------------------------
# Registry

Image.register_save(FITSStubImageFile.format, _save)
