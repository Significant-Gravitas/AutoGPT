#
# The Python Imaging Library.
# $Id$
#
# macOS icns file decoder, based on icns.py by Bob Ippolito.
#
# history:
# 2004-10-09 fl   Turned into a PIL plugin; removed 2.3 dependencies.
# 2020-04-04      Allow saving on all operating systems.
#
# Copyright (c) 2004 by Bob Ippolito.
# Copyright (c) 2004 by Secret Labs.
# Copyright (c) 2004 by Fredrik Lundh.
# Copyright (c) 2014 by Alastair Houghton.
# Copyright (c) 2020 by Pan Jing.
#
# See the README file for information on usage and redistribution.
#

import io
import os
import struct
import sys

from PIL import Image, ImageFile, PngImagePlugin, features

enable_jpeg2k = features.check_codec("jpg_2000")
if enable_jpeg2k:
    from PIL import Jpeg2KImagePlugin

MAGIC = b"icns"
HEADERSIZE = 8


def nextheader(fobj):
    return struct.unpack(">4sI", fobj.read(HEADERSIZE))


def read_32t(fobj, start_length, size):
    # The 128x128 icon seems to have an extra header for some reason.
    (start, length) = start_length
    fobj.seek(start)
    sig = fobj.read(4)
    if sig != b"\x00\x00\x00\x00":
        msg = "Unknown signature, expecting 0x00000000"
        raise SyntaxError(msg)
    return read_32(fobj, (start + 4, length - 4), size)


def read_32(fobj, start_length, size):
    """
    Read a 32bit RGB icon resource.  Seems to be either uncompressed or
    an RLE packbits-like scheme.
    """
    (start, length) = start_length
    fobj.seek(start)
    pixel_size = (size[0] * size[2], size[1] * size[2])
    sizesq = pixel_size[0] * pixel_size[1]
    if length == sizesq * 3:
        # uncompressed ("RGBRGBGB")
        indata = fobj.read(length)
        im = Image.frombuffer("RGB", pixel_size, indata, "raw", "RGB", 0, 1)
    else:
        # decode image
        im = Image.new("RGB", pixel_size, None)
        for band_ix in range(3):
            data = []
            bytesleft = sizesq
            while bytesleft > 0:
                byte = fobj.read(1)
                if not byte:
                    break
                byte = byte[0]
                if byte & 0x80:
                    blocksize = byte - 125
                    byte = fobj.read(1)
                    for i in range(blocksize):
                        data.append(byte)
                else:
                    blocksize = byte + 1
                    data.append(fobj.read(blocksize))
                bytesleft -= blocksize
                if bytesleft <= 0:
                    break
            if bytesleft != 0:
                msg = f"Error reading channel [{repr(bytesleft)} left]"
                raise SyntaxError(msg)
            band = Image.frombuffer("L", pixel_size, b"".join(data), "raw", "L", 0, 1)
            im.im.putband(band.im, band_ix)
    return {"RGB": im}


def read_mk(fobj, start_length, size):
    # Alpha masks seem to be uncompressed
    start = start_length[0]
    fobj.seek(start)
    pixel_size = (size[0] * size[2], size[1] * size[2])
    sizesq = pixel_size[0] * pixel_size[1]
    band = Image.frombuffer("L", pixel_size, fobj.read(sizesq), "raw", "L", 0, 1)
    return {"A": band}


def read_png_or_jpeg2000(fobj, start_length, size):
    (start, length) = start_length
    fobj.seek(start)
    sig = fobj.read(12)
    if sig[:8] == b"\x89PNG\x0d\x0a\x1a\x0a":
        fobj.seek(start)
        im = PngImagePlugin.PngImageFile(fobj)
        Image._decompression_bomb_check(im.size)
        return {"RGBA": im}
    elif (
        sig[:4] == b"\xff\x4f\xff\x51"
        or sig[:4] == b"\x0d\x0a\x87\x0a"
        or sig == b"\x00\x00\x00\x0cjP  \x0d\x0a\x87\x0a"
    ):
        if not enable_jpeg2k:
            msg = (
                "Unsupported icon subimage format (rebuild PIL "
                "with JPEG 2000 support to fix this)"
            )
            raise ValueError(msg)
        # j2k, jpc or j2c
        fobj.seek(start)
        jp2kstream = fobj.read(length)
        f = io.BytesIO(jp2kstream)
        im = Jpeg2KImagePlugin.Jpeg2KImageFile(f)
        Image._decompression_bomb_check(im.size)
        if im.mode != "RGBA":
            im = im.convert("RGBA")
        return {"RGBA": im}
    else:
        msg = "Unsupported icon subimage format"
        raise ValueError(msg)


class IcnsFile:
    SIZES = {
        (512, 512, 2): [(b"ic10", read_png_or_jpeg2000)],
        (512, 512, 1): [(b"ic09", read_png_or_jpeg2000)],
        (256, 256, 2): [(b"ic14", read_png_or_jpeg2000)],
        (256, 256, 1): [(b"ic08", read_png_or_jpeg2000)],
        (128, 128, 2): [(b"ic13", read_png_or_jpeg2000)],
        (128, 128, 1): [
            (b"ic07", read_png_or_jpeg2000),
            (b"it32", read_32t),
            (b"t8mk", read_mk),
        ],
        (64, 64, 1): [(b"icp6", read_png_or_jpeg2000)],
        (32, 32, 2): [(b"ic12", read_png_or_jpeg2000)],
        (48, 48, 1): [(b"ih32", read_32), (b"h8mk", read_mk)],
        (32, 32, 1): [
            (b"icp5", read_png_or_jpeg2000),
            (b"il32", read_32),
            (b"l8mk", read_mk),
        ],
        (16, 16, 2): [(b"ic11", read_png_or_jpeg2000)],
        (16, 16, 1): [
            (b"icp4", read_png_or_jpeg2000),
            (b"is32", read_32),
            (b"s8mk", read_mk),
        ],
    }

    def __init__(self, fobj):
        """
        fobj is a file-like object as an icns resource
        """
        # signature : (start, length)
        self.dct = dct = {}
        self.fobj = fobj
        sig, filesize = nextheader(fobj)
        if not _accept(sig):
            msg = "not an icns file"
            raise SyntaxError(msg)
        i = HEADERSIZE
        while i < filesize:
            sig, blocksize = nextheader(fobj)
            if blocksize <= 0:
                msg = "invalid block header"
                raise SyntaxError(msg)
            i += HEADERSIZE
            blocksize -= HEADERSIZE
            dct[sig] = (i, blocksize)
            fobj.seek(blocksize, io.SEEK_CUR)
            i += blocksize

    def itersizes(self):
        sizes = []
        for size, fmts in self.SIZES.items():
            for fmt, reader in fmts:
                if fmt in self.dct:
                    sizes.append(size)
                    break
        return sizes

    def bestsize(self):
        sizes = self.itersizes()
        if not sizes:
            msg = "No 32bit icon resources found"
            raise SyntaxError(msg)
        return max(sizes)

    def dataforsize(self, size):
        """
        Get an icon resource as {channel: array}.  Note that
        the arrays are bottom-up like windows bitmaps and will likely
        need to be flipped or transposed in some way.
        """
        dct = {}
        for code, reader in self.SIZES[size]:
            desc = self.dct.get(code)
            if desc is not None:
                dct.update(reader(self.fobj, desc, size))
        return dct

    def getimage(self, size=None):
        if size is None:
            size = self.bestsize()
        if len(size) == 2:
            size = (size[0], size[1], 1)
        channels = self.dataforsize(size)

        im = channels.get("RGBA", None)
        if im:
            return im

        im = channels.get("RGB").copy()
        try:
            im.putalpha(channels["A"])
        except KeyError:
            pass
        return im


##
# Image plugin for Mac OS icons.


class IcnsImageFile(ImageFile.ImageFile):
    """
    PIL image support for Mac OS .icns files.
    Chooses the best resolution, but will possibly load
    a different size image if you mutate the size attribute
    before calling 'load'.

    The info dictionary has a key 'sizes' that is a list
    of sizes that the icns file has.
    """

    format = "ICNS"
    format_description = "Mac OS icns resource"

    def _open(self):
        self.icns = IcnsFile(self.fp)
        self.mode = "RGBA"
        self.info["sizes"] = self.icns.itersizes()
        self.best_size = self.icns.bestsize()
        self.size = (
            self.best_size[0] * self.best_size[2],
            self.best_size[1] * self.best_size[2],
        )

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        info_size = value
        if info_size not in self.info["sizes"] and len(info_size) == 2:
            info_size = (info_size[0], info_size[1], 1)
        if (
            info_size not in self.info["sizes"]
            and len(info_size) == 3
            and info_size[2] == 1
        ):
            simple_sizes = [
                (size[0] * size[2], size[1] * size[2]) for size in self.info["sizes"]
            ]
            if value in simple_sizes:
                info_size = self.info["sizes"][simple_sizes.index(value)]
        if info_size not in self.info["sizes"]:
            msg = "This is not one of the allowed sizes of this image"
            raise ValueError(msg)
        self._size = value

    def load(self):
        if len(self.size) == 3:
            self.best_size = self.size
            self.size = (
                self.best_size[0] * self.best_size[2],
                self.best_size[1] * self.best_size[2],
            )

        px = Image.Image.load(self)
        if self.im is not None and self.im.size == self.size:
            # Already loaded
            return px
        self.load_prepare()
        # This is likely NOT the best way to do it, but whatever.
        im = self.icns.getimage(self.best_size)

        # If this is a PNG or JPEG 2000, it won't be loaded yet
        px = im.load()

        self.im = im.im
        self.mode = im.mode
        self.size = im.size

        return px


def _save(im, fp, filename):
    """
    Saves the image as a series of PNG files,
    that are then combined into a .icns file.
    """
    if hasattr(fp, "flush"):
        fp.flush()

    sizes = {
        b"ic07": 128,
        b"ic08": 256,
        b"ic09": 512,
        b"ic10": 1024,
        b"ic11": 32,
        b"ic12": 64,
        b"ic13": 256,
        b"ic14": 512,
    }
    provided_images = {im.width: im for im in im.encoderinfo.get("append_images", [])}
    size_streams = {}
    for size in set(sizes.values()):
        image = (
            provided_images[size]
            if size in provided_images
            else im.resize((size, size))
        )

        temp = io.BytesIO()
        image.save(temp, "png")
        size_streams[size] = temp.getvalue()

    entries = []
    for type, size in sizes.items():
        stream = size_streams[size]
        entries.append(
            {"type": type, "size": HEADERSIZE + len(stream), "stream": stream}
        )

    # Header
    fp.write(MAGIC)
    file_length = HEADERSIZE  # Header
    file_length += HEADERSIZE + 8 * len(entries)  # TOC
    file_length += sum(entry["size"] for entry in entries)
    fp.write(struct.pack(">i", file_length))

    # TOC
    fp.write(b"TOC ")
    fp.write(struct.pack(">i", HEADERSIZE + len(entries) * HEADERSIZE))
    for entry in entries:
        fp.write(entry["type"])
        fp.write(struct.pack(">i", entry["size"]))

    # Data
    for entry in entries:
        fp.write(entry["type"])
        fp.write(struct.pack(">i", entry["size"]))
        fp.write(entry["stream"])

    if hasattr(fp, "flush"):
        fp.flush()


def _accept(prefix):
    return prefix[:4] == MAGIC


Image.register_open(IcnsImageFile.format, IcnsImageFile, _accept)
Image.register_extension(IcnsImageFile.format, ".icns")

Image.register_save(IcnsImageFile.format, _save)
Image.register_mime(IcnsImageFile.format, "image/icns")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Syntax: python3 IcnsImagePlugin.py [file]")
        sys.exit()

    with open(sys.argv[1], "rb") as fp:
        imf = IcnsImageFile(fp)
        for size in imf.info["sizes"]:
            imf.size = size
            imf.save("out-%s-%s-%s.png" % size)
        with Image.open(sys.argv[1]) as im:
            im.save("out.png")
        if sys.platform == "windows":
            os.startfile("out.png")
