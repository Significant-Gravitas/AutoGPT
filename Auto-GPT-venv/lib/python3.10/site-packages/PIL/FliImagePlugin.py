#
# The Python Imaging Library.
# $Id$
#
# FLI/FLC file handling.
#
# History:
#       95-09-01 fl     Created
#       97-01-03 fl     Fixed parser, setup decoder tile
#       98-07-15 fl     Renamed offset attribute to avoid name clash
#
# Copyright (c) Secret Labs AB 1997-98.
# Copyright (c) Fredrik Lundh 1995-97.
#
# See the README file for information on usage and redistribution.
#

import os

from . import Image, ImageFile, ImagePalette
from ._binary import i16le as i16
from ._binary import i32le as i32
from ._binary import o8

#
# decoder


def _accept(prefix):
    return (
        len(prefix) >= 6
        and i16(prefix, 4) in [0xAF11, 0xAF12]
        and i16(prefix, 14) in [0, 3]  # flags
    )


##
# Image plugin for the FLI/FLC animation format.  Use the <b>seek</b>
# method to load individual frames.


class FliImageFile(ImageFile.ImageFile):
    format = "FLI"
    format_description = "Autodesk FLI/FLC Animation"
    _close_exclusive_fp_after_loading = False

    def _open(self):
        # HEAD
        s = self.fp.read(128)
        if not (_accept(s) and s[20:22] == b"\x00\x00"):
            msg = "not an FLI/FLC file"
            raise SyntaxError(msg)

        # frames
        self.n_frames = i16(s, 6)
        self.is_animated = self.n_frames > 1

        # image characteristics
        self.mode = "P"
        self._size = i16(s, 8), i16(s, 10)

        # animation speed
        duration = i32(s, 16)
        magic = i16(s, 4)
        if magic == 0xAF11:
            duration = (duration * 1000) // 70
        self.info["duration"] = duration

        # look for palette
        palette = [(a, a, a) for a in range(256)]

        s = self.fp.read(16)

        self.__offset = 128

        if i16(s, 4) == 0xF100:
            # prefix chunk; ignore it
            self.__offset = self.__offset + i32(s)
            s = self.fp.read(16)

        if i16(s, 4) == 0xF1FA:
            # look for palette chunk
            number_of_subchunks = i16(s, 6)
            chunk_size = None
            for _ in range(number_of_subchunks):
                if chunk_size is not None:
                    self.fp.seek(chunk_size - 6, os.SEEK_CUR)
                s = self.fp.read(6)
                chunk_type = i16(s, 4)
                if chunk_type in (4, 11):
                    self._palette(palette, 2 if chunk_type == 11 else 0)
                    break
                chunk_size = i32(s)
                if not chunk_size:
                    break

        palette = [o8(r) + o8(g) + o8(b) for (r, g, b) in palette]
        self.palette = ImagePalette.raw("RGB", b"".join(palette))

        # set things up to decode first frame
        self.__frame = -1
        self._fp = self.fp
        self.__rewind = self.fp.tell()
        self.seek(0)

    def _palette(self, palette, shift):
        # load palette

        i = 0
        for e in range(i16(self.fp.read(2))):
            s = self.fp.read(2)
            i = i + s[0]
            n = s[1]
            if n == 0:
                n = 256
            s = self.fp.read(n * 3)
            for n in range(0, len(s), 3):
                r = s[n] << shift
                g = s[n + 1] << shift
                b = s[n + 2] << shift
                palette[i] = (r, g, b)
                i += 1

    def seek(self, frame):
        if not self._seek_check(frame):
            return
        if frame < self.__frame:
            self._seek(0)

        for f in range(self.__frame + 1, frame + 1):
            self._seek(f)

    def _seek(self, frame):
        if frame == 0:
            self.__frame = -1
            self._fp.seek(self.__rewind)
            self.__offset = 128
        else:
            # ensure that the previous frame was loaded
            self.load()

        if frame != self.__frame + 1:
            msg = f"cannot seek to frame {frame}"
            raise ValueError(msg)
        self.__frame = frame

        # move to next frame
        self.fp = self._fp
        self.fp.seek(self.__offset)

        s = self.fp.read(4)
        if not s:
            raise EOFError

        framesize = i32(s)

        self.decodermaxblock = framesize
        self.tile = [("fli", (0, 0) + self.size, self.__offset, None)]

        self.__offset += framesize

    def tell(self):
        return self.__frame


#
# registry

Image.register_open(FliImageFile.format, FliImageFile, _accept)

Image.register_extensions(FliImageFile.format, [".fli", ".flc"])
