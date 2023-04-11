#
# The Python Imaging Library.
# $Id$
#
# SGI image file handling
#
# See "The SGI Image File Format (Draft version 0.97)", Paul Haeberli.
# <ftp://ftp.sgi.com/graphics/SGIIMAGESPEC>
#
#
# History:
# 2017-22-07 mb   Add RLE decompression
# 2016-16-10 mb   Add save method without compression
# 1995-09-10 fl   Created
#
# Copyright (c) 2016 by Mickael Bonfill.
# Copyright (c) 2008 by Karsten Hiddemann.
# Copyright (c) 1997 by Secret Labs AB.
# Copyright (c) 1995 by Fredrik Lundh.
#
# See the README file for information on usage and redistribution.
#


import os
import struct

from . import Image, ImageFile
from ._binary import i16be as i16
from ._binary import o8


def _accept(prefix):
    return len(prefix) >= 2 and i16(prefix) == 474


MODES = {
    (1, 1, 1): "L",
    (1, 2, 1): "L",
    (2, 1, 1): "L;16B",
    (2, 2, 1): "L;16B",
    (1, 3, 3): "RGB",
    (2, 3, 3): "RGB;16B",
    (1, 3, 4): "RGBA",
    (2, 3, 4): "RGBA;16B",
}


##
# Image plugin for SGI images.
class SgiImageFile(ImageFile.ImageFile):
    format = "SGI"
    format_description = "SGI Image File Format"

    def _open(self):
        # HEAD
        headlen = 512
        s = self.fp.read(headlen)

        if not _accept(s):
            msg = "Not an SGI image file"
            raise ValueError(msg)

        # compression : verbatim or RLE
        compression = s[2]

        # bpc : 1 or 2 bytes (8bits or 16bits)
        bpc = s[3]

        # dimension : 1, 2 or 3 (depending on xsize, ysize and zsize)
        dimension = i16(s, 4)

        # xsize : width
        xsize = i16(s, 6)

        # ysize : height
        ysize = i16(s, 8)

        # zsize : channels count
        zsize = i16(s, 10)

        # layout
        layout = bpc, dimension, zsize

        # determine mode from bits/zsize
        rawmode = ""
        try:
            rawmode = MODES[layout]
        except KeyError:
            pass

        if rawmode == "":
            msg = "Unsupported SGI image mode"
            raise ValueError(msg)

        self._size = xsize, ysize
        self.mode = rawmode.split(";")[0]
        if self.mode == "RGB":
            self.custom_mimetype = "image/rgb"

        # orientation -1 : scanlines begins at the bottom-left corner
        orientation = -1

        # decoder info
        if compression == 0:
            pagesize = xsize * ysize * bpc
            if bpc == 2:
                self.tile = [
                    ("SGI16", (0, 0) + self.size, headlen, (self.mode, 0, orientation))
                ]
            else:
                self.tile = []
                offset = headlen
                for layer in self.mode:
                    self.tile.append(
                        ("raw", (0, 0) + self.size, offset, (layer, 0, orientation))
                    )
                    offset += pagesize
        elif compression == 1:
            self.tile = [
                ("sgi_rle", (0, 0) + self.size, headlen, (rawmode, orientation, bpc))
            ]


def _save(im, fp, filename):
    if im.mode != "RGB" and im.mode != "RGBA" and im.mode != "L":
        msg = "Unsupported SGI image mode"
        raise ValueError(msg)

    # Get the keyword arguments
    info = im.encoderinfo

    # Byte-per-pixel precision, 1 = 8bits per pixel
    bpc = info.get("bpc", 1)

    if bpc not in (1, 2):
        msg = "Unsupported number of bytes per pixel"
        raise ValueError(msg)

    # Flip the image, since the origin of SGI file is the bottom-left corner
    orientation = -1
    # Define the file as SGI File Format
    magic_number = 474
    # Run-Length Encoding Compression - Unsupported at this time
    rle = 0

    # Number of dimensions (x,y,z)
    dim = 3
    # X Dimension = width / Y Dimension = height
    x, y = im.size
    if im.mode == "L" and y == 1:
        dim = 1
    elif im.mode == "L":
        dim = 2
    # Z Dimension: Number of channels
    z = len(im.mode)

    if dim == 1 or dim == 2:
        z = 1

    # assert we've got the right number of bands.
    if len(im.getbands()) != z:
        msg = f"incorrect number of bands in SGI write: {z} vs {len(im.getbands())}"
        raise ValueError(msg)

    # Minimum Byte value
    pinmin = 0
    # Maximum Byte value (255 = 8bits per pixel)
    pinmax = 255
    # Image name (79 characters max, truncated below in write)
    img_name = os.path.splitext(os.path.basename(filename))[0]
    img_name = img_name.encode("ascii", "ignore")
    # Standard representation of pixel in the file
    colormap = 0
    fp.write(struct.pack(">h", magic_number))
    fp.write(o8(rle))
    fp.write(o8(bpc))
    fp.write(struct.pack(">H", dim))
    fp.write(struct.pack(">H", x))
    fp.write(struct.pack(">H", y))
    fp.write(struct.pack(">H", z))
    fp.write(struct.pack(">l", pinmin))
    fp.write(struct.pack(">l", pinmax))
    fp.write(struct.pack("4s", b""))  # dummy
    fp.write(struct.pack("79s", img_name))  # truncates to 79 chars
    fp.write(struct.pack("s", b""))  # force null byte after img_name
    fp.write(struct.pack(">l", colormap))
    fp.write(struct.pack("404s", b""))  # dummy

    rawmode = "L"
    if bpc == 2:
        rawmode = "L;16B"

    for channel in im.split():
        fp.write(channel.tobytes("raw", rawmode, 0, orientation))

    if hasattr(fp, "flush"):
        fp.flush()


class SGI16Decoder(ImageFile.PyDecoder):
    _pulls_fd = True

    def decode(self, buffer):
        rawmode, stride, orientation = self.args
        pagesize = self.state.xsize * self.state.ysize
        zsize = len(self.mode)
        self.fd.seek(512)

        for band in range(zsize):
            channel = Image.new("L", (self.state.xsize, self.state.ysize))
            channel.frombytes(
                self.fd.read(2 * pagesize), "raw", "L;16B", stride, orientation
            )
            self.im.putband(channel.im, band)

        return -1, 0


#
# registry


Image.register_decoder("SGI16", SGI16Decoder)
Image.register_open(SgiImageFile.format, SgiImageFile, _accept)
Image.register_save(SgiImageFile.format, _save)
Image.register_mime(SgiImageFile.format, "image/sgi")

Image.register_extensions(SgiImageFile.format, [".bw", ".rgb", ".rgba", ".sgi"])

# End of file
