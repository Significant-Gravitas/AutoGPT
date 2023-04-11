#
# The Python Imaging Library.
# $Id$
#
# DCX file handling
#
# DCX is a container file format defined by Intel, commonly used
# for fax applications.  Each DCX file consists of a directory
# (a list of file offsets) followed by a set of (usually 1-bit)
# PCX files.
#
# History:
# 1995-09-09 fl   Created
# 1996-03-20 fl   Properly derived from PcxImageFile.
# 1998-07-15 fl   Renamed offset attribute to avoid name clash
# 2002-07-30 fl   Fixed file handling
#
# Copyright (c) 1997-98 by Secret Labs AB.
# Copyright (c) 1995-96 by Fredrik Lundh.
#
# See the README file for information on usage and redistribution.
#

from . import Image
from ._binary import i32le as i32
from .PcxImagePlugin import PcxImageFile

MAGIC = 0x3ADE68B1  # QUIZ: what's this value, then?


def _accept(prefix):
    return len(prefix) >= 4 and i32(prefix) == MAGIC


##
# Image plugin for the Intel DCX format.


class DcxImageFile(PcxImageFile):
    format = "DCX"
    format_description = "Intel DCX"
    _close_exclusive_fp_after_loading = False

    def _open(self):
        # Header
        s = self.fp.read(4)
        if not _accept(s):
            msg = "not a DCX file"
            raise SyntaxError(msg)

        # Component directory
        self._offset = []
        for i in range(1024):
            offset = i32(self.fp.read(4))
            if not offset:
                break
            self._offset.append(offset)

        self._fp = self.fp
        self.frame = None
        self.n_frames = len(self._offset)
        self.is_animated = self.n_frames > 1
        self.seek(0)

    def seek(self, frame):
        if not self._seek_check(frame):
            return
        self.frame = frame
        self.fp = self._fp
        self.fp.seek(self._offset[frame])
        PcxImageFile._open(self)

    def tell(self):
        return self.frame


Image.register_open(DcxImageFile.format, DcxImageFile, _accept)

Image.register_extension(DcxImageFile.format, ".dcx")
