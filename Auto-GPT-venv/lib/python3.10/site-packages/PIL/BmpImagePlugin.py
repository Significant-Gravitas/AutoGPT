#
# The Python Imaging Library.
# $Id$
#
# BMP file handler
#
# Windows (and OS/2) native bitmap storage format.
#
# history:
# 1995-09-01 fl   Created
# 1996-04-30 fl   Added save
# 1997-08-27 fl   Fixed save of 1-bit images
# 1998-03-06 fl   Load P images as L where possible
# 1998-07-03 fl   Load P images as 1 where possible
# 1998-12-29 fl   Handle small palettes
# 2002-12-30 fl   Fixed load of 1-bit palette images
# 2003-04-21 fl   Fixed load of 1-bit monochrome images
# 2003-04-23 fl   Added limited support for BI_BITFIELDS compression
#
# Copyright (c) 1997-2003 by Secret Labs AB
# Copyright (c) 1995-2003 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#


import os

from . import Image, ImageFile, ImagePalette
from ._binary import i16le as i16
from ._binary import i32le as i32
from ._binary import o8
from ._binary import o16le as o16
from ._binary import o32le as o32

#
# --------------------------------------------------------------------
# Read BMP file

BIT2MODE = {
    # bits => mode, rawmode
    1: ("P", "P;1"),
    4: ("P", "P;4"),
    8: ("P", "P"),
    16: ("RGB", "BGR;15"),
    24: ("RGB", "BGR"),
    32: ("RGB", "BGRX"),
}


def _accept(prefix):
    return prefix[:2] == b"BM"


def _dib_accept(prefix):
    return i32(prefix) in [12, 40, 64, 108, 124]


# =============================================================================
# Image plugin for the Windows BMP format.
# =============================================================================
class BmpImageFile(ImageFile.ImageFile):
    """Image plugin for the Windows Bitmap format (BMP)"""

    # ------------------------------------------------------------- Description
    format_description = "Windows Bitmap"
    format = "BMP"

    # -------------------------------------------------- BMP Compression values
    COMPRESSIONS = {"RAW": 0, "RLE8": 1, "RLE4": 2, "BITFIELDS": 3, "JPEG": 4, "PNG": 5}
    for k, v in COMPRESSIONS.items():
        vars()[k] = v

    def _bitmap(self, header=0, offset=0):
        """Read relevant info about the BMP"""
        read, seek = self.fp.read, self.fp.seek
        if header:
            seek(header)
        # read bmp header size @offset 14 (this is part of the header size)
        file_info = {"header_size": i32(read(4)), "direction": -1}

        # -------------------- If requested, read header at a specific position
        # read the rest of the bmp header, without its size
        header_data = ImageFile._safe_read(self.fp, file_info["header_size"] - 4)

        # -------------------------------------------------- IBM OS/2 Bitmap v1
        # ----- This format has different offsets because of width/height types
        if file_info["header_size"] == 12:
            file_info["width"] = i16(header_data, 0)
            file_info["height"] = i16(header_data, 2)
            file_info["planes"] = i16(header_data, 4)
            file_info["bits"] = i16(header_data, 6)
            file_info["compression"] = self.RAW
            file_info["palette_padding"] = 3

        # --------------------------------------------- Windows Bitmap v2 to v5
        # v3, OS/2 v2, v4, v5
        elif file_info["header_size"] in (40, 64, 108, 124):
            file_info["y_flip"] = header_data[7] == 0xFF
            file_info["direction"] = 1 if file_info["y_flip"] else -1
            file_info["width"] = i32(header_data, 0)
            file_info["height"] = (
                i32(header_data, 4)
                if not file_info["y_flip"]
                else 2**32 - i32(header_data, 4)
            )
            file_info["planes"] = i16(header_data, 8)
            file_info["bits"] = i16(header_data, 10)
            file_info["compression"] = i32(header_data, 12)
            # byte size of pixel data
            file_info["data_size"] = i32(header_data, 16)
            file_info["pixels_per_meter"] = (
                i32(header_data, 20),
                i32(header_data, 24),
            )
            file_info["colors"] = i32(header_data, 28)
            file_info["palette_padding"] = 4
            self.info["dpi"] = tuple(x / 39.3701 for x in file_info["pixels_per_meter"])
            if file_info["compression"] == self.BITFIELDS:
                if len(header_data) >= 52:
                    for idx, mask in enumerate(
                        ["r_mask", "g_mask", "b_mask", "a_mask"]
                    ):
                        file_info[mask] = i32(header_data, 36 + idx * 4)
                else:
                    # 40 byte headers only have the three components in the
                    # bitfields masks, ref:
                    # https://msdn.microsoft.com/en-us/library/windows/desktop/dd183376(v=vs.85).aspx
                    # See also
                    # https://github.com/python-pillow/Pillow/issues/1293
                    # There is a 4th component in the RGBQuad, in the alpha
                    # location, but it is listed as a reserved component,
                    # and it is not generally an alpha channel
                    file_info["a_mask"] = 0x0
                    for mask in ["r_mask", "g_mask", "b_mask"]:
                        file_info[mask] = i32(read(4))
                file_info["rgb_mask"] = (
                    file_info["r_mask"],
                    file_info["g_mask"],
                    file_info["b_mask"],
                )
                file_info["rgba_mask"] = (
                    file_info["r_mask"],
                    file_info["g_mask"],
                    file_info["b_mask"],
                    file_info["a_mask"],
                )
        else:
            msg = f"Unsupported BMP header type ({file_info['header_size']})"
            raise OSError(msg)

        # ------------------ Special case : header is reported 40, which
        # ---------------------- is shorter than real size for bpp >= 16
        self._size = file_info["width"], file_info["height"]

        # ------- If color count was not found in the header, compute from bits
        file_info["colors"] = (
            file_info["colors"]
            if file_info.get("colors", 0)
            else (1 << file_info["bits"])
        )
        if offset == 14 + file_info["header_size"] and file_info["bits"] <= 8:
            offset += 4 * file_info["colors"]

        # ---------------------- Check bit depth for unusual unsupported values
        self.mode, raw_mode = BIT2MODE.get(file_info["bits"], (None, None))
        if self.mode is None:
            msg = f"Unsupported BMP pixel depth ({file_info['bits']})"
            raise OSError(msg)

        # ---------------- Process BMP with Bitfields compression (not palette)
        decoder_name = "raw"
        if file_info["compression"] == self.BITFIELDS:
            SUPPORTED = {
                32: [
                    (0xFF0000, 0xFF00, 0xFF, 0x0),
                    (0xFF000000, 0xFF0000, 0xFF00, 0x0),
                    (0xFF000000, 0xFF0000, 0xFF00, 0xFF),
                    (0xFF, 0xFF00, 0xFF0000, 0xFF000000),
                    (0xFF0000, 0xFF00, 0xFF, 0xFF000000),
                    (0x0, 0x0, 0x0, 0x0),
                ],
                24: [(0xFF0000, 0xFF00, 0xFF)],
                16: [(0xF800, 0x7E0, 0x1F), (0x7C00, 0x3E0, 0x1F)],
            }
            MASK_MODES = {
                (32, (0xFF0000, 0xFF00, 0xFF, 0x0)): "BGRX",
                (32, (0xFF000000, 0xFF0000, 0xFF00, 0x0)): "XBGR",
                (32, (0xFF000000, 0xFF0000, 0xFF00, 0xFF)): "ABGR",
                (32, (0xFF, 0xFF00, 0xFF0000, 0xFF000000)): "RGBA",
                (32, (0xFF0000, 0xFF00, 0xFF, 0xFF000000)): "BGRA",
                (32, (0x0, 0x0, 0x0, 0x0)): "BGRA",
                (24, (0xFF0000, 0xFF00, 0xFF)): "BGR",
                (16, (0xF800, 0x7E0, 0x1F)): "BGR;16",
                (16, (0x7C00, 0x3E0, 0x1F)): "BGR;15",
            }
            if file_info["bits"] in SUPPORTED:
                if (
                    file_info["bits"] == 32
                    and file_info["rgba_mask"] in SUPPORTED[file_info["bits"]]
                ):
                    raw_mode = MASK_MODES[(file_info["bits"], file_info["rgba_mask"])]
                    self.mode = "RGBA" if "A" in raw_mode else self.mode
                elif (
                    file_info["bits"] in (24, 16)
                    and file_info["rgb_mask"] in SUPPORTED[file_info["bits"]]
                ):
                    raw_mode = MASK_MODES[(file_info["bits"], file_info["rgb_mask"])]
                else:
                    msg = "Unsupported BMP bitfields layout"
                    raise OSError(msg)
            else:
                msg = "Unsupported BMP bitfields layout"
                raise OSError(msg)
        elif file_info["compression"] == self.RAW:
            if file_info["bits"] == 32 and header == 22:  # 32-bit .cur offset
                raw_mode, self.mode = "BGRA", "RGBA"
        elif file_info["compression"] in (self.RLE8, self.RLE4):
            decoder_name = "bmp_rle"
        else:
            msg = f"Unsupported BMP compression ({file_info['compression']})"
            raise OSError(msg)

        # --------------- Once the header is processed, process the palette/LUT
        if self.mode == "P":  # Paletted for 1, 4 and 8 bit images
            # ---------------------------------------------------- 1-bit images
            if not (0 < file_info["colors"] <= 65536):
                msg = f"Unsupported BMP Palette size ({file_info['colors']})"
                raise OSError(msg)
            else:
                padding = file_info["palette_padding"]
                palette = read(padding * file_info["colors"])
                greyscale = True
                indices = (
                    (0, 255)
                    if file_info["colors"] == 2
                    else list(range(file_info["colors"]))
                )

                # ----------------- Check if greyscale and ignore palette if so
                for ind, val in enumerate(indices):
                    rgb = palette[ind * padding : ind * padding + 3]
                    if rgb != o8(val) * 3:
                        greyscale = False

                # ------- If all colors are grey, white or black, ditch palette
                if greyscale:
                    self.mode = "1" if file_info["colors"] == 2 else "L"
                    raw_mode = self.mode
                else:
                    self.mode = "P"
                    self.palette = ImagePalette.raw(
                        "BGRX" if padding == 4 else "BGR", palette
                    )

        # ---------------------------- Finally set the tile data for the plugin
        self.info["compression"] = file_info["compression"]
        args = [raw_mode]
        if decoder_name == "bmp_rle":
            args.append(file_info["compression"] == self.RLE4)
        else:
            args.append(((file_info["width"] * file_info["bits"] + 31) >> 3) & (~3))
        args.append(file_info["direction"])
        self.tile = [
            (
                decoder_name,
                (0, 0, file_info["width"], file_info["height"]),
                offset or self.fp.tell(),
                tuple(args),
            )
        ]

    def _open(self):
        """Open file, check magic number and read header"""
        # read 14 bytes: magic number, filesize, reserved, header final offset
        head_data = self.fp.read(14)
        # choke if the file does not have the required magic bytes
        if not _accept(head_data):
            msg = "Not a BMP file"
            raise SyntaxError(msg)
        # read the start position of the BMP image data (u32)
        offset = i32(head_data, 10)
        # load bitmap information (offset=raster info)
        self._bitmap(offset=offset)


class BmpRleDecoder(ImageFile.PyDecoder):
    _pulls_fd = True

    def decode(self, buffer):
        rle4 = self.args[1]
        data = bytearray()
        x = 0
        while len(data) < self.state.xsize * self.state.ysize:
            pixels = self.fd.read(1)
            byte = self.fd.read(1)
            if not pixels or not byte:
                break
            num_pixels = pixels[0]
            if num_pixels:
                # encoded mode
                if x + num_pixels > self.state.xsize:
                    # Too much data for row
                    num_pixels = max(0, self.state.xsize - x)
                if rle4:
                    first_pixel = o8(byte[0] >> 4)
                    second_pixel = o8(byte[0] & 0x0F)
                    for index in range(num_pixels):
                        if index % 2 == 0:
                            data += first_pixel
                        else:
                            data += second_pixel
                else:
                    data += byte * num_pixels
                x += num_pixels
            else:
                if byte[0] == 0:
                    # end of line
                    while len(data) % self.state.xsize != 0:
                        data += b"\x00"
                    x = 0
                elif byte[0] == 1:
                    # end of bitmap
                    break
                elif byte[0] == 2:
                    # delta
                    bytes_read = self.fd.read(2)
                    if len(bytes_read) < 2:
                        break
                    right, up = self.fd.read(2)
                    data += b"\x00" * (right + up * self.state.xsize)
                    x = len(data) % self.state.xsize
                else:
                    # absolute mode
                    if rle4:
                        # 2 pixels per byte
                        byte_count = byte[0] // 2
                        bytes_read = self.fd.read(byte_count)
                        for byte_read in bytes_read:
                            data += o8(byte_read >> 4)
                            data += o8(byte_read & 0x0F)
                    else:
                        byte_count = byte[0]
                        bytes_read = self.fd.read(byte_count)
                        data += bytes_read
                    if len(bytes_read) < byte_count:
                        break
                    x += byte[0]

                    # align to 16-bit word boundary
                    if self.fd.tell() % 2 != 0:
                        self.fd.seek(1, os.SEEK_CUR)
        rawmode = "L" if self.mode == "L" else "P"
        self.set_as_raw(bytes(data), (rawmode, 0, self.args[-1]))
        return -1, 0


# =============================================================================
# Image plugin for the DIB format (BMP alias)
# =============================================================================
class DibImageFile(BmpImageFile):
    format = "DIB"
    format_description = "Windows Bitmap"

    def _open(self):
        self._bitmap()


#
# --------------------------------------------------------------------
# Write BMP file


SAVE = {
    "1": ("1", 1, 2),
    "L": ("L", 8, 256),
    "P": ("P", 8, 256),
    "RGB": ("BGR", 24, 0),
    "RGBA": ("BGRA", 32, 0),
}


def _dib_save(im, fp, filename):
    _save(im, fp, filename, False)


def _save(im, fp, filename, bitmap_header=True):
    try:
        rawmode, bits, colors = SAVE[im.mode]
    except KeyError as e:
        msg = f"cannot write mode {im.mode} as BMP"
        raise OSError(msg) from e

    info = im.encoderinfo

    dpi = info.get("dpi", (96, 96))

    # 1 meter == 39.3701 inches
    ppm = tuple(map(lambda x: int(x * 39.3701 + 0.5), dpi))

    stride = ((im.size[0] * bits + 7) // 8 + 3) & (~3)
    header = 40  # or 64 for OS/2 version 2
    image = stride * im.size[1]

    if im.mode == "1":
        palette = b"".join(o8(i) * 4 for i in (0, 255))
    elif im.mode == "L":
        palette = b"".join(o8(i) * 4 for i in range(256))
    elif im.mode == "P":
        palette = im.im.getpalette("RGB", "BGRX")
        colors = len(palette) // 4
    else:
        palette = None

    # bitmap header
    if bitmap_header:
        offset = 14 + header + colors * 4
        file_size = offset + image
        if file_size > 2**32 - 1:
            msg = "File size is too large for the BMP format"
            raise ValueError(msg)
        fp.write(
            b"BM"  # file type (magic)
            + o32(file_size)  # file size
            + o32(0)  # reserved
            + o32(offset)  # image data offset
        )

    # bitmap info header
    fp.write(
        o32(header)  # info header size
        + o32(im.size[0])  # width
        + o32(im.size[1])  # height
        + o16(1)  # planes
        + o16(bits)  # depth
        + o32(0)  # compression (0=uncompressed)
        + o32(image)  # size of bitmap
        + o32(ppm[0])  # resolution
        + o32(ppm[1])  # resolution
        + o32(colors)  # colors used
        + o32(colors)  # colors important
    )

    fp.write(b"\0" * (header - 40))  # padding (for OS/2 format)

    if palette:
        fp.write(palette)

    ImageFile._save(im, fp, [("raw", (0, 0) + im.size, 0, (rawmode, stride, -1))])


#
# --------------------------------------------------------------------
# Registry


Image.register_open(BmpImageFile.format, BmpImageFile, _accept)
Image.register_save(BmpImageFile.format, _save)

Image.register_extension(BmpImageFile.format, ".bmp")

Image.register_mime(BmpImageFile.format, "image/bmp")

Image.register_decoder("bmp_rle", BmpRleDecoder)

Image.register_open(DibImageFile.format, DibImageFile, _dib_accept)
Image.register_save(DibImageFile.format, _dib_save)

Image.register_extension(DibImageFile.format, ".dib")

Image.register_mime(DibImageFile.format, "image/bmp")
