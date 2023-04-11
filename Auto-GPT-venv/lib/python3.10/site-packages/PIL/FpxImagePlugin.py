#
# THIS IS WORK IN PROGRESS
#
# The Python Imaging Library.
# $Id$
#
# FlashPix support for PIL
#
# History:
# 97-01-25 fl   Created (reads uncompressed RGB images only)
#
# Copyright (c) Secret Labs AB 1997.
# Copyright (c) Fredrik Lundh 1997.
#
# See the README file for information on usage and redistribution.
#
import olefile

from . import Image, ImageFile
from ._binary import i32le as i32

# we map from colour field tuples to (mode, rawmode) descriptors
MODES = {
    # opacity
    (0x00007FFE,): ("A", "L"),
    # monochrome
    (0x00010000,): ("L", "L"),
    (0x00018000, 0x00017FFE): ("RGBA", "LA"),
    # photo YCC
    (0x00020000, 0x00020001, 0x00020002): ("RGB", "YCC;P"),
    (0x00028000, 0x00028001, 0x00028002, 0x00027FFE): ("RGBA", "YCCA;P"),
    # standard RGB (NIFRGB)
    (0x00030000, 0x00030001, 0x00030002): ("RGB", "RGB"),
    (0x00038000, 0x00038001, 0x00038002, 0x00037FFE): ("RGBA", "RGBA"),
}


#
# --------------------------------------------------------------------


def _accept(prefix):
    return prefix[:8] == olefile.MAGIC


##
# Image plugin for the FlashPix images.


class FpxImageFile(ImageFile.ImageFile):
    format = "FPX"
    format_description = "FlashPix"

    def _open(self):
        #
        # read the OLE directory and see if this is a likely
        # to be a FlashPix file

        try:
            self.ole = olefile.OleFileIO(self.fp)
        except OSError as e:
            msg = "not an FPX file; invalid OLE file"
            raise SyntaxError(msg) from e

        if self.ole.root.clsid != "56616700-C154-11CE-8553-00AA00A1F95B":
            msg = "not an FPX file; bad root CLSID"
            raise SyntaxError(msg)

        self._open_index(1)

    def _open_index(self, index=1):
        #
        # get the Image Contents Property Set

        prop = self.ole.getproperties(
            [f"Data Object Store {index:06d}", "\005Image Contents"]
        )

        # size (highest resolution)

        self._size = prop[0x1000002], prop[0x1000003]

        size = max(self.size)
        i = 1
        while size > 64:
            size = size / 2
            i += 1
        self.maxid = i - 1

        # mode.  instead of using a single field for this, flashpix
        # requires you to specify the mode for each channel in each
        # resolution subimage, and leaves it to the decoder to make
        # sure that they all match.  for now, we'll cheat and assume
        # that this is always the case.

        id = self.maxid << 16

        s = prop[0x2000002 | id]

        colors = []
        bands = i32(s, 4)
        if bands > 4:
            msg = "Invalid number of bands"
            raise OSError(msg)
        for i in range(bands):
            # note: for now, we ignore the "uncalibrated" flag
            colors.append(i32(s, 8 + i * 4) & 0x7FFFFFFF)

        self.mode, self.rawmode = MODES[tuple(colors)]

        # load JPEG tables, if any
        self.jpeg = {}
        for i in range(256):
            id = 0x3000001 | (i << 16)
            if id in prop:
                self.jpeg[i] = prop[id]

        self._open_subimage(1, self.maxid)

    def _open_subimage(self, index=1, subimage=0):
        #
        # setup tile descriptors for a given subimage

        stream = [
            f"Data Object Store {index:06d}",
            f"Resolution {subimage:04d}",
            "Subimage 0000 Header",
        ]

        fp = self.ole.openstream(stream)

        # skip prefix
        fp.read(28)

        # header stream
        s = fp.read(36)

        size = i32(s, 4), i32(s, 8)
        # tilecount = i32(s, 12)
        tilesize = i32(s, 16), i32(s, 20)
        # channels = i32(s, 24)
        offset = i32(s, 28)
        length = i32(s, 32)

        if size != self.size:
            msg = "subimage mismatch"
            raise OSError(msg)

        # get tile descriptors
        fp.seek(28 + offset)
        s = fp.read(i32(s, 12) * length)

        x = y = 0
        xsize, ysize = size
        xtile, ytile = tilesize
        self.tile = []

        for i in range(0, len(s), length):
            x1 = min(xsize, x + xtile)
            y1 = min(ysize, y + ytile)

            compression = i32(s, i + 8)

            if compression == 0:
                self.tile.append(
                    (
                        "raw",
                        (x, y, x1, y1),
                        i32(s, i) + 28,
                        (self.rawmode,),
                    )
                )

            elif compression == 1:
                # FIXME: the fill decoder is not implemented
                self.tile.append(
                    (
                        "fill",
                        (x, y, x1, y1),
                        i32(s, i) + 28,
                        (self.rawmode, s[12:16]),
                    )
                )

            elif compression == 2:
                internal_color_conversion = s[14]
                jpeg_tables = s[15]
                rawmode = self.rawmode

                if internal_color_conversion:
                    # The image is stored as usual (usually YCbCr).
                    if rawmode == "RGBA":
                        # For "RGBA", data is stored as YCbCrA based on
                        # negative RGB. The following trick works around
                        # this problem :
                        jpegmode, rawmode = "YCbCrK", "CMYK"
                    else:
                        jpegmode = None  # let the decoder decide

                else:
                    # The image is stored as defined by rawmode
                    jpegmode = rawmode

                self.tile.append(
                    (
                        "jpeg",
                        (x, y, x1, y1),
                        i32(s, i) + 28,
                        (rawmode, jpegmode),
                    )
                )

                # FIXME: jpeg tables are tile dependent; the prefix
                # data must be placed in the tile descriptor itself!

                if jpeg_tables:
                    self.tile_prefix = self.jpeg[jpeg_tables]

            else:
                msg = "unknown/invalid compression"
                raise OSError(msg)

            x = x + xtile
            if x >= xsize:
                x, y = 0, y + ytile
                if y >= ysize:
                    break  # isn't really required

        self.stream = stream
        self.fp = None

    def load(self):
        if not self.fp:
            self.fp = self.ole.openstream(self.stream[:2] + ["Subimage 0000 Data"])

        return ImageFile.ImageFile.load(self)

    def close(self):
        self.ole.close()
        super().close()

    def __exit__(self, *args):
        self.ole.close()
        super().__exit__()


#
# --------------------------------------------------------------------


Image.register_open(FpxImageFile.format, FpxImageFile, _accept)

Image.register_extension(FpxImageFile.format, ".fpx")
