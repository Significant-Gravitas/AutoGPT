#
# The Python Imaging Library
# $Id$
#
# BUFR stub adapter
#
# Copyright (c) 1996-2003 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#

from . import Image, ImageFile

_handler = None


def register_handler(handler):
    """
    Install application-specific BUFR image handler.

    :param handler: Handler object.
    """
    global _handler
    _handler = handler


# --------------------------------------------------------------------
# Image adapter


def _accept(prefix):
    return prefix[:4] == b"BUFR" or prefix[:4] == b"ZCZC"


class BufrStubImageFile(ImageFile.StubImageFile):
    format = "BUFR"
    format_description = "BUFR"

    def _open(self):
        offset = self.fp.tell()

        if not _accept(self.fp.read(4)):
            msg = "Not a BUFR file"
            raise SyntaxError(msg)

        self.fp.seek(offset)

        # make something up
        self.mode = "F"
        self._size = 1, 1

        loader = self._load()
        if loader:
            loader.open(self)

    def _load(self):
        return _handler


def _save(im, fp, filename):
    if _handler is None or not hasattr(_handler, "save"):
        msg = "BUFR save handler not installed"
        raise OSError(msg)
    _handler.save(im, fp, filename)


# --------------------------------------------------------------------
# Registry

Image.register_open(BufrStubImageFile.format, BufrStubImageFile, _accept)
Image.register_save(BufrStubImageFile.format, _save)

Image.register_extension(BufrStubImageFile.format, ".bufr")
