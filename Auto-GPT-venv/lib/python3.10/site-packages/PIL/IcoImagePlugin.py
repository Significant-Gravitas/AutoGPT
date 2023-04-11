#
# The Python Imaging Library.
# $Id$
#
# Windows Icon support for PIL
#
# History:
#       96-05-27 fl     Created
#
# Copyright (c) Secret Labs AB 1997.
# Copyright (c) Fredrik Lundh 1996.
#
# See the README file for information on usage and redistribution.
#

# This plugin is a refactored version of Win32IconImagePlugin by Bryan Davis
# <casadebender@gmail.com>.
# https://code.google.com/archive/p/casadebender/wikis/Win32IconImagePlugin.wiki
#
# Icon format references:
#   * https://en.wikipedia.org/wiki/ICO_(file_format)
#   * https://msdn.microsoft.com/en-us/library/ms997538.aspx


import warnings
from io import BytesIO
from math import ceil, log

from . import BmpImagePlugin, Image, ImageFile, PngImagePlugin
from ._binary import i16le as i16
from ._binary import i32le as i32
from ._binary import o8
from ._binary import o16le as o16
from ._binary import o32le as o32

#
# --------------------------------------------------------------------

_MAGIC = b"\0\0\1\0"


def _save(im, fp, filename):
    fp.write(_MAGIC)  # (2+2)
    bmp = im.encoderinfo.get("bitmap_format") == "bmp"
    sizes = im.encoderinfo.get(
        "sizes",
        [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    frames = []
    provided_ims = [im] + im.encoderinfo.get("append_images", [])
    width, height = im.size
    for size in sorted(set(sizes)):
        if size[0] > width or size[1] > height or size[0] > 256 or size[1] > 256:
            continue

        for provided_im in provided_ims:
            if provided_im.size != size:
                continue
            frames.append(provided_im)
            if bmp:
                bits = BmpImagePlugin.SAVE[provided_im.mode][1]
                bits_used = [bits]
                for other_im in provided_ims:
                    if other_im.size != size:
                        continue
                    bits = BmpImagePlugin.SAVE[other_im.mode][1]
                    if bits not in bits_used:
                        # Another image has been supplied for this size
                        # with a different bit depth
                        frames.append(other_im)
                        bits_used.append(bits)
            break
        else:
            # TODO: invent a more convenient method for proportional scalings
            frame = provided_im.copy()
            frame.thumbnail(size, Image.Resampling.LANCZOS, reducing_gap=None)
            frames.append(frame)
    fp.write(o16(len(frames)))  # idCount(2)
    offset = fp.tell() + len(frames) * 16
    for frame in frames:
        width, height = frame.size
        # 0 means 256
        fp.write(o8(width if width < 256 else 0))  # bWidth(1)
        fp.write(o8(height if height < 256 else 0))  # bHeight(1)

        bits, colors = BmpImagePlugin.SAVE[frame.mode][1:] if bmp else (32, 0)
        fp.write(o8(colors))  # bColorCount(1)
        fp.write(b"\0")  # bReserved(1)
        fp.write(b"\0\0")  # wPlanes(2)
        fp.write(o16(bits))  # wBitCount(2)

        image_io = BytesIO()
        if bmp:
            frame.save(image_io, "dib")

            if bits != 32:
                and_mask = Image.new("1", size)
                ImageFile._save(
                    and_mask, image_io, [("raw", (0, 0) + size, 0, ("1", 0, -1))]
                )
        else:
            frame.save(image_io, "png")
        image_io.seek(0)
        image_bytes = image_io.read()
        if bmp:
            image_bytes = image_bytes[:8] + o32(height * 2) + image_bytes[12:]
        bytes_len = len(image_bytes)
        fp.write(o32(bytes_len))  # dwBytesInRes(4)
        fp.write(o32(offset))  # dwImageOffset(4)
        current = fp.tell()
        fp.seek(offset)
        fp.write(image_bytes)
        offset = offset + bytes_len
        fp.seek(current)


def _accept(prefix):
    return prefix[:4] == _MAGIC


class IcoFile:
    def __init__(self, buf):
        """
        Parse image from file-like object containing ico file data
        """

        # check magic
        s = buf.read(6)
        if not _accept(s):
            msg = "not an ICO file"
            raise SyntaxError(msg)

        self.buf = buf
        self.entry = []

        # Number of items in file
        self.nb_items = i16(s, 4)

        # Get headers for each item
        for i in range(self.nb_items):
            s = buf.read(16)

            icon_header = {
                "width": s[0],
                "height": s[1],
                "nb_color": s[2],  # No. of colors in image (0 if >=8bpp)
                "reserved": s[3],
                "planes": i16(s, 4),
                "bpp": i16(s, 6),
                "size": i32(s, 8),
                "offset": i32(s, 12),
            }

            # See Wikipedia
            for j in ("width", "height"):
                if not icon_header[j]:
                    icon_header[j] = 256

            # See Wikipedia notes about color depth.
            # We need this just to differ images with equal sizes
            icon_header["color_depth"] = (
                icon_header["bpp"]
                or (
                    icon_header["nb_color"] != 0
                    and ceil(log(icon_header["nb_color"], 2))
                )
                or 256
            )

            icon_header["dim"] = (icon_header["width"], icon_header["height"])
            icon_header["square"] = icon_header["width"] * icon_header["height"]

            self.entry.append(icon_header)

        self.entry = sorted(self.entry, key=lambda x: x["color_depth"])
        # ICO images are usually squares
        # self.entry = sorted(self.entry, key=lambda x: x['width'])
        self.entry = sorted(self.entry, key=lambda x: x["square"])
        self.entry.reverse()

    def sizes(self):
        """
        Get a list of all available icon sizes and color depths.
        """
        return {(h["width"], h["height"]) for h in self.entry}

    def getentryindex(self, size, bpp=False):
        for i, h in enumerate(self.entry):
            if size == h["dim"] and (bpp is False or bpp == h["color_depth"]):
                return i
        return 0

    def getimage(self, size, bpp=False):
        """
        Get an image from the icon
        """
        return self.frame(self.getentryindex(size, bpp))

    def frame(self, idx):
        """
        Get an image from frame idx
        """

        header = self.entry[idx]

        self.buf.seek(header["offset"])
        data = self.buf.read(8)
        self.buf.seek(header["offset"])

        if data[:8] == PngImagePlugin._MAGIC:
            # png frame
            im = PngImagePlugin.PngImageFile(self.buf)
            Image._decompression_bomb_check(im.size)
        else:
            # XOR + AND mask bmp frame
            im = BmpImagePlugin.DibImageFile(self.buf)
            Image._decompression_bomb_check(im.size)

            # change tile dimension to only encompass XOR image
            im._size = (im.size[0], int(im.size[1] / 2))
            d, e, o, a = im.tile[0]
            im.tile[0] = d, (0, 0) + im.size, o, a

            # figure out where AND mask image starts
            bpp = header["bpp"]
            if 32 == bpp:
                # 32-bit color depth icon image allows semitransparent areas
                # PIL's DIB format ignores transparency bits, recover them.
                # The DIB is packed in BGRX byte order where X is the alpha
                # channel.

                # Back up to start of bmp data
                self.buf.seek(o)
                # extract every 4th byte (eg. 3,7,11,15,...)
                alpha_bytes = self.buf.read(im.size[0] * im.size[1] * 4)[3::4]

                # convert to an 8bpp grayscale image
                mask = Image.frombuffer(
                    "L",  # 8bpp
                    im.size,  # (w, h)
                    alpha_bytes,  # source chars
                    "raw",  # raw decoder
                    ("L", 0, -1),  # 8bpp inverted, unpadded, reversed
                )
            else:
                # get AND image from end of bitmap
                w = im.size[0]
                if (w % 32) > 0:
                    # bitmap row data is aligned to word boundaries
                    w += 32 - (im.size[0] % 32)

                # the total mask data is
                # padded row size * height / bits per char

                total_bytes = int((w * im.size[1]) / 8)
                and_mask_offset = header["offset"] + header["size"] - total_bytes

                self.buf.seek(and_mask_offset)
                mask_data = self.buf.read(total_bytes)

                # convert raw data to image
                mask = Image.frombuffer(
                    "1",  # 1 bpp
                    im.size,  # (w, h)
                    mask_data,  # source chars
                    "raw",  # raw decoder
                    ("1;I", int(w / 8), -1),  # 1bpp inverted, padded, reversed
                )

                # now we have two images, im is XOR image and mask is AND image

            # apply mask image as alpha channel
            im = im.convert("RGBA")
            im.putalpha(mask)

        return im


##
# Image plugin for Windows Icon files.


class IcoImageFile(ImageFile.ImageFile):
    """
    PIL read-only image support for Microsoft Windows .ico files.

    By default the largest resolution image in the file will be loaded. This
    can be changed by altering the 'size' attribute before calling 'load'.

    The info dictionary has a key 'sizes' that is a list of the sizes available
    in the icon file.

    Handles classic, XP and Vista icon formats.

    When saving, PNG compression is used. Support for this was only added in
    Windows Vista. If you are unable to view the icon in Windows, convert the
    image to "RGBA" mode before saving.

    This plugin is a refactored version of Win32IconImagePlugin by Bryan Davis
    <casadebender@gmail.com>.
    https://code.google.com/archive/p/casadebender/wikis/Win32IconImagePlugin.wiki
    """

    format = "ICO"
    format_description = "Windows Icon"

    def _open(self):
        self.ico = IcoFile(self.fp)
        self.info["sizes"] = self.ico.sizes()
        self.size = self.ico.entry[0]["dim"]
        self.load()

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        if value not in self.info["sizes"]:
            msg = "This is not one of the allowed sizes of this image"
            raise ValueError(msg)
        self._size = value

    def load(self):
        if self.im is not None and self.im.size == self.size:
            # Already loaded
            return Image.Image.load(self)
        im = self.ico.getimage(self.size)
        # if tile is PNG, it won't really be loaded yet
        im.load()
        self.im = im.im
        self.pyaccess = None
        self.mode = im.mode
        if im.size != self.size:
            warnings.warn("Image was not the expected size")

            index = self.ico.getentryindex(self.size)
            sizes = list(self.info["sizes"])
            sizes[index] = im.size
            self.info["sizes"] = set(sizes)

            self.size = im.size

    def load_seek(self):
        # Flag the ImageFile.Parser so that it
        # just does all the decode at the end.
        pass


#
# --------------------------------------------------------------------


Image.register_open(IcoImageFile.format, IcoImageFile, _accept)
Image.register_save(IcoImageFile.format, _save)
Image.register_extension(IcoImageFile.format, ".ico")

Image.register_mime(IcoImageFile.format, "image/x-icon")
