#
# The Python Imaging Library.
# $Id$
#
# IFUNC IM file handling for PIL
#
# history:
# 1995-09-01 fl   Created.
# 1997-01-03 fl   Save palette images
# 1997-01-08 fl   Added sequence support
# 1997-01-23 fl   Added P and RGB save support
# 1997-05-31 fl   Read floating point images
# 1997-06-22 fl   Save floating point images
# 1997-08-27 fl   Read and save 1-bit images
# 1998-06-25 fl   Added support for RGB+LUT images
# 1998-07-02 fl   Added support for YCC images
# 1998-07-15 fl   Renamed offset attribute to avoid name clash
# 1998-12-29 fl   Added I;16 support
# 2001-02-17 fl   Use 're' instead of 'regex' (Python 2.1) (0.7)
# 2003-09-26 fl   Added LA/PA support
#
# Copyright (c) 1997-2003 by Secret Labs AB.
# Copyright (c) 1995-2001 by Fredrik Lundh.
#
# See the README file for information on usage and redistribution.
#


import os
import re

from . import Image, ImageFile, ImagePalette

# --------------------------------------------------------------------
# Standard tags

COMMENT = "Comment"
DATE = "Date"
EQUIPMENT = "Digitalization equipment"
FRAMES = "File size (no of images)"
LUT = "Lut"
NAME = "Name"
SCALE = "Scale (x,y)"
SIZE = "Image size (x*y)"
MODE = "Image type"

TAGS = {
    COMMENT: 0,
    DATE: 0,
    EQUIPMENT: 0,
    FRAMES: 0,
    LUT: 0,
    NAME: 0,
    SCALE: 0,
    SIZE: 0,
    MODE: 0,
}

OPEN = {
    # ifunc93/p3cfunc formats
    "0 1 image": ("1", "1"),
    "L 1 image": ("1", "1"),
    "Greyscale image": ("L", "L"),
    "Grayscale image": ("L", "L"),
    "RGB image": ("RGB", "RGB;L"),
    "RLB image": ("RGB", "RLB"),
    "RYB image": ("RGB", "RLB"),
    "B1 image": ("1", "1"),
    "B2 image": ("P", "P;2"),
    "B4 image": ("P", "P;4"),
    "X 24 image": ("RGB", "RGB"),
    "L 32 S image": ("I", "I;32"),
    "L 32 F image": ("F", "F;32"),
    # old p3cfunc formats
    "RGB3 image": ("RGB", "RGB;T"),
    "RYB3 image": ("RGB", "RYB;T"),
    # extensions
    "LA image": ("LA", "LA;L"),
    "PA image": ("LA", "PA;L"),
    "RGBA image": ("RGBA", "RGBA;L"),
    "RGBX image": ("RGBX", "RGBX;L"),
    "CMYK image": ("CMYK", "CMYK;L"),
    "YCC image": ("YCbCr", "YCbCr;L"),
}

# ifunc95 extensions
for i in ["8", "8S", "16", "16S", "32", "32F"]:
    OPEN[f"L {i} image"] = ("F", f"F;{i}")
    OPEN[f"L*{i} image"] = ("F", f"F;{i}")
for i in ["16", "16L", "16B"]:
    OPEN[f"L {i} image"] = (f"I;{i}", f"I;{i}")
    OPEN[f"L*{i} image"] = (f"I;{i}", f"I;{i}")
for i in ["32S"]:
    OPEN[f"L {i} image"] = ("I", f"I;{i}")
    OPEN[f"L*{i} image"] = ("I", f"I;{i}")
for i in range(2, 33):
    OPEN[f"L*{i} image"] = ("F", f"F;{i}")


# --------------------------------------------------------------------
# Read IM directory

split = re.compile(rb"^([A-Za-z][^:]*):[ \t]*(.*)[ \t]*$")


def number(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


##
# Image plugin for the IFUNC IM file format.


class ImImageFile(ImageFile.ImageFile):
    format = "IM"
    format_description = "IFUNC Image Memory"
    _close_exclusive_fp_after_loading = False

    def _open(self):
        # Quick rejection: if there's not an LF among the first
        # 100 bytes, this is (probably) not a text header.

        if b"\n" not in self.fp.read(100):
            msg = "not an IM file"
            raise SyntaxError(msg)
        self.fp.seek(0)

        n = 0

        # Default values
        self.info[MODE] = "L"
        self.info[SIZE] = (512, 512)
        self.info[FRAMES] = 1

        self.rawmode = "L"

        while True:
            s = self.fp.read(1)

            # Some versions of IFUNC uses \n\r instead of \r\n...
            if s == b"\r":
                continue

            if not s or s == b"\0" or s == b"\x1A":
                break

            # FIXME: this may read whole file if not a text file
            s = s + self.fp.readline()

            if len(s) > 100:
                msg = "not an IM file"
                raise SyntaxError(msg)

            if s[-2:] == b"\r\n":
                s = s[:-2]
            elif s[-1:] == b"\n":
                s = s[:-1]

            try:
                m = split.match(s)
            except re.error as e:
                msg = "not an IM file"
                raise SyntaxError(msg) from e

            if m:
                k, v = m.group(1, 2)

                # Don't know if this is the correct encoding,
                # but a decent guess (I guess)
                k = k.decode("latin-1", "replace")
                v = v.decode("latin-1", "replace")

                # Convert value as appropriate
                if k in [FRAMES, SCALE, SIZE]:
                    v = v.replace("*", ",")
                    v = tuple(map(number, v.split(",")))
                    if len(v) == 1:
                        v = v[0]
                elif k == MODE and v in OPEN:
                    v, self.rawmode = OPEN[v]

                # Add to dictionary. Note that COMMENT tags are
                # combined into a list of strings.
                if k == COMMENT:
                    if k in self.info:
                        self.info[k].append(v)
                    else:
                        self.info[k] = [v]
                else:
                    self.info[k] = v

                if k in TAGS:
                    n += 1

            else:
                msg = "Syntax error in IM header: " + s.decode("ascii", "replace")
                raise SyntaxError(msg)

        if not n:
            msg = "Not an IM file"
            raise SyntaxError(msg)

        # Basic attributes
        self._size = self.info[SIZE]
        self.mode = self.info[MODE]

        # Skip forward to start of image data
        while s and s[:1] != b"\x1A":
            s = self.fp.read(1)
        if not s:
            msg = "File truncated"
            raise SyntaxError(msg)

        if LUT in self.info:
            # convert lookup table to palette or lut attribute
            palette = self.fp.read(768)
            greyscale = 1  # greyscale palette
            linear = 1  # linear greyscale palette
            for i in range(256):
                if palette[i] == palette[i + 256] == palette[i + 512]:
                    if palette[i] != i:
                        linear = 0
                else:
                    greyscale = 0
            if self.mode in ["L", "LA", "P", "PA"]:
                if greyscale:
                    if not linear:
                        self.lut = list(palette[:256])
                else:
                    if self.mode in ["L", "P"]:
                        self.mode = self.rawmode = "P"
                    elif self.mode in ["LA", "PA"]:
                        self.mode = "PA"
                        self.rawmode = "PA;L"
                    self.palette = ImagePalette.raw("RGB;L", palette)
            elif self.mode == "RGB":
                if not greyscale or not linear:
                    self.lut = list(palette)

        self.frame = 0

        self.__offset = offs = self.fp.tell()

        self._fp = self.fp  # FIXME: hack

        if self.rawmode[:2] == "F;":
            # ifunc95 formats
            try:
                # use bit decoder (if necessary)
                bits = int(self.rawmode[2:])
                if bits not in [8, 16, 32]:
                    self.tile = [("bit", (0, 0) + self.size, offs, (bits, 8, 3, 0, -1))]
                    return
            except ValueError:
                pass

        if self.rawmode in ["RGB;T", "RYB;T"]:
            # Old LabEye/3PC files.  Would be very surprised if anyone
            # ever stumbled upon such a file ;-)
            size = self.size[0] * self.size[1]
            self.tile = [
                ("raw", (0, 0) + self.size, offs, ("G", 0, -1)),
                ("raw", (0, 0) + self.size, offs + size, ("R", 0, -1)),
                ("raw", (0, 0) + self.size, offs + 2 * size, ("B", 0, -1)),
            ]
        else:
            # LabEye/IFUNC files
            self.tile = [("raw", (0, 0) + self.size, offs, (self.rawmode, 0, -1))]

    @property
    def n_frames(self):
        return self.info[FRAMES]

    @property
    def is_animated(self):
        return self.info[FRAMES] > 1

    def seek(self, frame):
        if not self._seek_check(frame):
            return

        self.frame = frame

        if self.mode == "1":
            bits = 1
        else:
            bits = 8 * len(self.mode)

        size = ((self.size[0] * bits + 7) // 8) * self.size[1]
        offs = self.__offset + frame * size

        self.fp = self._fp

        self.tile = [("raw", (0, 0) + self.size, offs, (self.rawmode, 0, -1))]

    def tell(self):
        return self.frame


#
# --------------------------------------------------------------------
# Save IM files


SAVE = {
    # mode: (im type, raw mode)
    "1": ("0 1", "1"),
    "L": ("Greyscale", "L"),
    "LA": ("LA", "LA;L"),
    "P": ("Greyscale", "P"),
    "PA": ("LA", "PA;L"),
    "I": ("L 32S", "I;32S"),
    "I;16": ("L 16", "I;16"),
    "I;16L": ("L 16L", "I;16L"),
    "I;16B": ("L 16B", "I;16B"),
    "F": ("L 32F", "F;32F"),
    "RGB": ("RGB", "RGB;L"),
    "RGBA": ("RGBA", "RGBA;L"),
    "RGBX": ("RGBX", "RGBX;L"),
    "CMYK": ("CMYK", "CMYK;L"),
    "YCbCr": ("YCC", "YCbCr;L"),
}


def _save(im, fp, filename):
    try:
        image_type, rawmode = SAVE[im.mode]
    except KeyError as e:
        msg = f"Cannot save {im.mode} images as IM"
        raise ValueError(msg) from e

    frames = im.encoderinfo.get("frames", 1)

    fp.write(f"Image type: {image_type} image\r\n".encode("ascii"))
    if filename:
        # Each line must be 100 characters or less,
        # or: SyntaxError("not an IM file")
        # 8 characters are used for "Name: " and "\r\n"
        # Keep just the filename, ditch the potentially overlong path
        name, ext = os.path.splitext(os.path.basename(filename))
        name = "".join([name[: 92 - len(ext)], ext])

        fp.write(f"Name: {name}\r\n".encode("ascii"))
    fp.write(("Image size (x*y): %d*%d\r\n" % im.size).encode("ascii"))
    fp.write(f"File size (no of images): {frames}\r\n".encode("ascii"))
    if im.mode in ["P", "PA"]:
        fp.write(b"Lut: 1\r\n")
    fp.write(b"\000" * (511 - fp.tell()) + b"\032")
    if im.mode in ["P", "PA"]:
        im_palette = im.im.getpalette("RGB", "RGB;L")
        colors = len(im_palette) // 3
        palette = b""
        for i in range(3):
            palette += im_palette[colors * i : colors * (i + 1)]
            palette += b"\x00" * (256 - colors)
        fp.write(palette)  # 768 bytes
    ImageFile._save(im, fp, [("raw", (0, 0) + im.size, 0, (rawmode, 0, -1))])


#
# --------------------------------------------------------------------
# Registry


Image.register_open(ImImageFile.format, ImImageFile)
Image.register_save(ImImageFile.format, _save)

Image.register_extension(ImImageFile.format, ".im")
