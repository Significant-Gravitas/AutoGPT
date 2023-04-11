#
# Python Imaging Library
# $Id$
#
# stuff to read simple, teragon-style palette files
#
# History:
#       97-08-23 fl     Created
#
# Copyright (c) Secret Labs AB 1997.
# Copyright (c) Fredrik Lundh 1997.
#
# See the README file for information on usage and redistribution.
#

from ._binary import o8


class PaletteFile:
    """File handler for Teragon-style palette files."""

    rawmode = "RGB"

    def __init__(self, fp):
        self.palette = [(i, i, i) for i in range(256)]

        while True:
            s = fp.readline()

            if not s:
                break
            if s[:1] == b"#":
                continue
            if len(s) > 100:
                msg = "bad palette file"
                raise SyntaxError(msg)

            v = [int(x) for x in s.split()]
            try:
                [i, r, g, b] = v
            except ValueError:
                [i, r] = v
                g = b = r

            if 0 <= i <= 255:
                self.palette[i] = o8(r) + o8(g) + o8(b)

        self.palette = b"".join(self.palette)

    def getpalette(self):
        return self.palette, self.rawmode
