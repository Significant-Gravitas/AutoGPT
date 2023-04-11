"""
Blizzard Mipmap Format (.blp)
Jerome Leclanche <jerome@leclan.ch>

The contents of this file are hereby released in the public domain (CC0)
Full text of the CC0 license:
  https://creativecommons.org/publicdomain/zero/1.0/

BLP1 files, used mostly in Warcraft III, are not fully supported.
All types of BLP2 files used in World of Warcraft are supported.

The BLP file structure consists of a header, up to 16 mipmaps of the
texture

Texture sizes must be powers of two, though the two dimensions do
not have to be equal; 512x256 is valid, but 512x200 is not.
The first mipmap (mipmap #0) is the full size image; each subsequent
mipmap halves both dimensions. The final mipmap should be 1x1.

BLP files come in many different flavours:
* JPEG-compressed (type == 0) - only supported for BLP1.
* RAW images (type == 1, encoding == 1). Each mipmap is stored as an
  array of 8-bit values, one per pixel, left to right, top to bottom.
  Each value is an index to the palette.
* DXT-compressed (type == 1, encoding == 2):
- DXT1 compression is used if alpha_encoding == 0.
  - An additional alpha bit is used if alpha_depth == 1.
  - DXT3 compression is used if alpha_encoding == 1.
  - DXT5 compression is used if alpha_encoding == 7.
"""

import os
import struct
from enum import IntEnum
from io import BytesIO

from . import Image, ImageFile
from ._deprecate import deprecate


class Format(IntEnum):
    JPEG = 0


class Encoding(IntEnum):
    UNCOMPRESSED = 1
    DXT = 2
    UNCOMPRESSED_RAW_BGRA = 3


class AlphaEncoding(IntEnum):
    DXT1 = 0
    DXT3 = 1
    DXT5 = 7


def __getattr__(name):
    for enum, prefix in {
        Format: "BLP_FORMAT_",
        Encoding: "BLP_ENCODING_",
        AlphaEncoding: "BLP_ALPHA_ENCODING_",
    }.items():
        if name.startswith(prefix):
            name = name[len(prefix) :]
            if name in enum.__members__:
                deprecate(f"{prefix}{name}", 10, f"{enum.__name__}.{name}")
                return enum[name]
    msg = f"module '{__name__}' has no attribute '{name}'"
    raise AttributeError(msg)


def unpack_565(i):
    return ((i >> 11) & 0x1F) << 3, ((i >> 5) & 0x3F) << 2, (i & 0x1F) << 3


def decode_dxt1(data, alpha=False):
    """
    input: one "row" of data (i.e. will produce 4*width pixels)
    """

    blocks = len(data) // 8  # number of blocks in row
    ret = (bytearray(), bytearray(), bytearray(), bytearray())

    for block in range(blocks):
        # Decode next 8-byte block.
        idx = block * 8
        color0, color1, bits = struct.unpack_from("<HHI", data, idx)

        r0, g0, b0 = unpack_565(color0)
        r1, g1, b1 = unpack_565(color1)

        # Decode this block into 4x4 pixels
        # Accumulate the results onto our 4 row accumulators
        for j in range(4):
            for i in range(4):
                # get next control op and generate a pixel

                control = bits & 3
                bits = bits >> 2

                a = 0xFF
                if control == 0:
                    r, g, b = r0, g0, b0
                elif control == 1:
                    r, g, b = r1, g1, b1
                elif control == 2:
                    if color0 > color1:
                        r = (2 * r0 + r1) // 3
                        g = (2 * g0 + g1) // 3
                        b = (2 * b0 + b1) // 3
                    else:
                        r = (r0 + r1) // 2
                        g = (g0 + g1) // 2
                        b = (b0 + b1) // 2
                elif control == 3:
                    if color0 > color1:
                        r = (2 * r1 + r0) // 3
                        g = (2 * g1 + g0) // 3
                        b = (2 * b1 + b0) // 3
                    else:
                        r, g, b, a = 0, 0, 0, 0

                if alpha:
                    ret[j].extend([r, g, b, a])
                else:
                    ret[j].extend([r, g, b])

    return ret


def decode_dxt3(data):
    """
    input: one "row" of data (i.e. will produce 4*width pixels)
    """

    blocks = len(data) // 16  # number of blocks in row
    ret = (bytearray(), bytearray(), bytearray(), bytearray())

    for block in range(blocks):
        idx = block * 16
        block = data[idx : idx + 16]
        # Decode next 16-byte block.
        bits = struct.unpack_from("<8B", block)
        color0, color1 = struct.unpack_from("<HH", block, 8)

        (code,) = struct.unpack_from("<I", block, 12)

        r0, g0, b0 = unpack_565(color0)
        r1, g1, b1 = unpack_565(color1)

        for j in range(4):
            high = False  # Do we want the higher bits?
            for i in range(4):
                alphacode_index = (4 * j + i) // 2
                a = bits[alphacode_index]
                if high:
                    high = False
                    a >>= 4
                else:
                    high = True
                    a &= 0xF
                a *= 17  # We get a value between 0 and 15

                color_code = (code >> 2 * (4 * j + i)) & 0x03

                if color_code == 0:
                    r, g, b = r0, g0, b0
                elif color_code == 1:
                    r, g, b = r1, g1, b1
                elif color_code == 2:
                    r = (2 * r0 + r1) // 3
                    g = (2 * g0 + g1) // 3
                    b = (2 * b0 + b1) // 3
                elif color_code == 3:
                    r = (2 * r1 + r0) // 3
                    g = (2 * g1 + g0) // 3
                    b = (2 * b1 + b0) // 3

                ret[j].extend([r, g, b, a])

    return ret


def decode_dxt5(data):
    """
    input: one "row" of data (i.e. will produce 4 * width pixels)
    """

    blocks = len(data) // 16  # number of blocks in row
    ret = (bytearray(), bytearray(), bytearray(), bytearray())

    for block in range(blocks):
        idx = block * 16
        block = data[idx : idx + 16]
        # Decode next 16-byte block.
        a0, a1 = struct.unpack_from("<BB", block)

        bits = struct.unpack_from("<6B", block, 2)
        alphacode1 = bits[2] | (bits[3] << 8) | (bits[4] << 16) | (bits[5] << 24)
        alphacode2 = bits[0] | (bits[1] << 8)

        color0, color1 = struct.unpack_from("<HH", block, 8)

        (code,) = struct.unpack_from("<I", block, 12)

        r0, g0, b0 = unpack_565(color0)
        r1, g1, b1 = unpack_565(color1)

        for j in range(4):
            for i in range(4):
                # get next control op and generate a pixel
                alphacode_index = 3 * (4 * j + i)

                if alphacode_index <= 12:
                    alphacode = (alphacode2 >> alphacode_index) & 0x07
                elif alphacode_index == 15:
                    alphacode = (alphacode2 >> 15) | ((alphacode1 << 1) & 0x06)
                else:  # alphacode_index >= 18 and alphacode_index <= 45
                    alphacode = (alphacode1 >> (alphacode_index - 16)) & 0x07

                if alphacode == 0:
                    a = a0
                elif alphacode == 1:
                    a = a1
                elif a0 > a1:
                    a = ((8 - alphacode) * a0 + (alphacode - 1) * a1) // 7
                elif alphacode == 6:
                    a = 0
                elif alphacode == 7:
                    a = 255
                else:
                    a = ((6 - alphacode) * a0 + (alphacode - 1) * a1) // 5

                color_code = (code >> 2 * (4 * j + i)) & 0x03

                if color_code == 0:
                    r, g, b = r0, g0, b0
                elif color_code == 1:
                    r, g, b = r1, g1, b1
                elif color_code == 2:
                    r = (2 * r0 + r1) // 3
                    g = (2 * g0 + g1) // 3
                    b = (2 * b0 + b1) // 3
                elif color_code == 3:
                    r = (2 * r1 + r0) // 3
                    g = (2 * g1 + g0) // 3
                    b = (2 * b1 + b0) // 3

                ret[j].extend([r, g, b, a])

    return ret


class BLPFormatError(NotImplementedError):
    pass


def _accept(prefix):
    return prefix[:4] in (b"BLP1", b"BLP2")


class BlpImageFile(ImageFile.ImageFile):
    """
    Blizzard Mipmap Format
    """

    format = "BLP"
    format_description = "Blizzard Mipmap Format"

    def _open(self):
        self.magic = self.fp.read(4)

        self.fp.seek(5, os.SEEK_CUR)
        (self._blp_alpha_depth,) = struct.unpack("<b", self.fp.read(1))

        self.fp.seek(2, os.SEEK_CUR)
        self._size = struct.unpack("<II", self.fp.read(8))

        if self.magic in (b"BLP1", b"BLP2"):
            decoder = self.magic.decode()
        else:
            msg = f"Bad BLP magic {repr(self.magic)}"
            raise BLPFormatError(msg)

        self.mode = "RGBA" if self._blp_alpha_depth else "RGB"
        self.tile = [(decoder, (0, 0) + self.size, 0, (self.mode, 0, 1))]


class _BLPBaseDecoder(ImageFile.PyDecoder):
    _pulls_fd = True

    def decode(self, buffer):
        try:
            self._read_blp_header()
            self._load()
        except struct.error as e:
            msg = "Truncated BLP file"
            raise OSError(msg) from e
        return -1, 0

    def _read_blp_header(self):
        self.fd.seek(4)
        (self._blp_compression,) = struct.unpack("<i", self._safe_read(4))

        (self._blp_encoding,) = struct.unpack("<b", self._safe_read(1))
        (self._blp_alpha_depth,) = struct.unpack("<b", self._safe_read(1))
        (self._blp_alpha_encoding,) = struct.unpack("<b", self._safe_read(1))
        self.fd.seek(1, os.SEEK_CUR)  # mips

        self.size = struct.unpack("<II", self._safe_read(8))

        if isinstance(self, BLP1Decoder):
            # Only present for BLP1
            (self._blp_encoding,) = struct.unpack("<i", self._safe_read(4))
            self.fd.seek(4, os.SEEK_CUR)  # subtype

        self._blp_offsets = struct.unpack("<16I", self._safe_read(16 * 4))
        self._blp_lengths = struct.unpack("<16I", self._safe_read(16 * 4))

    def _safe_read(self, length):
        return ImageFile._safe_read(self.fd, length)

    def _read_palette(self):
        ret = []
        for i in range(256):
            try:
                b, g, r, a = struct.unpack("<4B", self._safe_read(4))
            except struct.error:
                break
            ret.append((b, g, r, a))
        return ret

    def _read_bgra(self, palette):
        data = bytearray()
        _data = BytesIO(self._safe_read(self._blp_lengths[0]))
        while True:
            try:
                (offset,) = struct.unpack("<B", _data.read(1))
            except struct.error:
                break
            b, g, r, a = palette[offset]
            d = (r, g, b)
            if self._blp_alpha_depth:
                d += (a,)
            data.extend(d)
        return data


class BLP1Decoder(_BLPBaseDecoder):
    def _load(self):
        if self._blp_compression == Format.JPEG:
            self._decode_jpeg_stream()

        elif self._blp_compression == 1:
            if self._blp_encoding in (4, 5):
                palette = self._read_palette()
                data = self._read_bgra(palette)
                self.set_as_raw(bytes(data))
            else:
                msg = f"Unsupported BLP encoding {repr(self._blp_encoding)}"
                raise BLPFormatError(msg)
        else:
            msg = f"Unsupported BLP compression {repr(self._blp_encoding)}"
            raise BLPFormatError(msg)

    def _decode_jpeg_stream(self):
        from .JpegImagePlugin import JpegImageFile

        (jpeg_header_size,) = struct.unpack("<I", self._safe_read(4))
        jpeg_header = self._safe_read(jpeg_header_size)
        self._safe_read(self._blp_offsets[0] - self.fd.tell())  # What IS this?
        data = self._safe_read(self._blp_lengths[0])
        data = jpeg_header + data
        data = BytesIO(data)
        image = JpegImageFile(data)
        Image._decompression_bomb_check(image.size)
        if image.mode == "CMYK":
            decoder_name, extents, offset, args = image.tile[0]
            image.tile = [(decoder_name, extents, offset, (args[0], "CMYK"))]
        r, g, b = image.convert("RGB").split()
        image = Image.merge("RGB", (b, g, r))
        self.set_as_raw(image.tobytes())


class BLP2Decoder(_BLPBaseDecoder):
    def _load(self):
        palette = self._read_palette()

        self.fd.seek(self._blp_offsets[0])

        if self._blp_compression == 1:
            # Uncompressed or DirectX compression

            if self._blp_encoding == Encoding.UNCOMPRESSED:
                data = self._read_bgra(palette)

            elif self._blp_encoding == Encoding.DXT:
                data = bytearray()
                if self._blp_alpha_encoding == AlphaEncoding.DXT1:
                    linesize = (self.size[0] + 3) // 4 * 8
                    for yb in range((self.size[1] + 3) // 4):
                        for d in decode_dxt1(
                            self._safe_read(linesize), alpha=bool(self._blp_alpha_depth)
                        ):
                            data += d

                elif self._blp_alpha_encoding == AlphaEncoding.DXT3:
                    linesize = (self.size[0] + 3) // 4 * 16
                    for yb in range((self.size[1] + 3) // 4):
                        for d in decode_dxt3(self._safe_read(linesize)):
                            data += d

                elif self._blp_alpha_encoding == AlphaEncoding.DXT5:
                    linesize = (self.size[0] + 3) // 4 * 16
                    for yb in range((self.size[1] + 3) // 4):
                        for d in decode_dxt5(self._safe_read(linesize)):
                            data += d
                else:
                    msg = f"Unsupported alpha encoding {repr(self._blp_alpha_encoding)}"
                    raise BLPFormatError(msg)
            else:
                msg = f"Unknown BLP encoding {repr(self._blp_encoding)}"
                raise BLPFormatError(msg)

        else:
            msg = f"Unknown BLP compression {repr(self._blp_compression)}"
            raise BLPFormatError(msg)

        self.set_as_raw(bytes(data))


class BLPEncoder(ImageFile.PyEncoder):
    _pushes_fd = True

    def _write_palette(self):
        data = b""
        palette = self.im.getpalette("RGBA", "RGBA")
        for i in range(256):
            r, g, b, a = palette[i * 4 : (i + 1) * 4]
            data += struct.pack("<4B", b, g, r, a)
        return data

    def encode(self, bufsize):
        palette_data = self._write_palette()

        offset = 20 + 16 * 4 * 2 + len(palette_data)
        data = struct.pack("<16I", offset, *((0,) * 15))

        w, h = self.im.size
        data += struct.pack("<16I", w * h, *((0,) * 15))

        data += palette_data

        for y in range(h):
            for x in range(w):
                data += struct.pack("<B", self.im.getpixel((x, y)))

        return len(data), 0, data


def _save(im, fp, filename, save_all=False):
    if im.mode != "P":
        msg = "Unsupported BLP image mode"
        raise ValueError(msg)

    magic = b"BLP1" if im.encoderinfo.get("blp_version") == "BLP1" else b"BLP2"
    fp.write(magic)

    fp.write(struct.pack("<i", 1))  # Uncompressed or DirectX compression
    fp.write(struct.pack("<b", Encoding.UNCOMPRESSED))
    fp.write(struct.pack("<b", 1 if im.palette.mode == "RGBA" else 0))
    fp.write(struct.pack("<b", 0))  # alpha encoding
    fp.write(struct.pack("<b", 0))  # mips
    fp.write(struct.pack("<II", *im.size))
    if magic == b"BLP1":
        fp.write(struct.pack("<i", 5))
        fp.write(struct.pack("<i", 0))

    ImageFile._save(im, fp, [("BLP", (0, 0) + im.size, 0, im.mode)])


Image.register_open(BlpImageFile.format, BlpImageFile, _accept)
Image.register_extension(BlpImageFile.format, ".blp")
Image.register_decoder("BLP1", BLP1Decoder)
Image.register_decoder("BLP2", BLP2Decoder)

Image.register_save(BlpImageFile.format, _save)
Image.register_encoder("BLP", BLPEncoder)
