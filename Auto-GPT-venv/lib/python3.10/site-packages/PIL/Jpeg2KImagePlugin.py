#
# The Python Imaging Library
# $Id$
#
# JPEG2000 file handling
#
# History:
# 2014-03-12 ajh  Created
# 2021-06-30 rogermb  Extract dpi information from the 'resc' header box
#
# Copyright (c) 2014 Coriolis Systems Limited
# Copyright (c) 2014 Alastair Houghton
#
# See the README file for information on usage and redistribution.
#
import io
import os
import struct

from . import Image, ImageFile, _binary


class BoxReader:
    """
    A small helper class to read fields stored in JPEG2000 header boxes
    and to easily step into and read sub-boxes.
    """

    def __init__(self, fp, length=-1):
        self.fp = fp
        self.has_length = length >= 0
        self.length = length
        self.remaining_in_box = -1

    def _can_read(self, num_bytes):
        if self.has_length and self.fp.tell() + num_bytes > self.length:
            # Outside box: ensure we don't read past the known file length
            return False
        if self.remaining_in_box >= 0:
            # Inside box contents: ensure read does not go past box boundaries
            return num_bytes <= self.remaining_in_box
        else:
            return True  # No length known, just read

    def _read_bytes(self, num_bytes):
        if not self._can_read(num_bytes):
            msg = "Not enough data in header"
            raise SyntaxError(msg)

        data = self.fp.read(num_bytes)
        if len(data) < num_bytes:
            msg = f"Expected to read {num_bytes} bytes but only got {len(data)}."
            raise OSError(msg)

        if self.remaining_in_box > 0:
            self.remaining_in_box -= num_bytes
        return data

    def read_fields(self, field_format):
        size = struct.calcsize(field_format)
        data = self._read_bytes(size)
        return struct.unpack(field_format, data)

    def read_boxes(self):
        size = self.remaining_in_box
        data = self._read_bytes(size)
        return BoxReader(io.BytesIO(data), size)

    def has_next_box(self):
        if self.has_length:
            return self.fp.tell() + self.remaining_in_box < self.length
        else:
            return True

    def next_box_type(self):
        # Skip the rest of the box if it has not been read
        if self.remaining_in_box > 0:
            self.fp.seek(self.remaining_in_box, os.SEEK_CUR)
        self.remaining_in_box = -1

        # Read the length and type of the next box
        lbox, tbox = self.read_fields(">I4s")
        if lbox == 1:
            lbox = self.read_fields(">Q")[0]
            hlen = 16
        else:
            hlen = 8

        if lbox < hlen or not self._can_read(lbox - hlen):
            msg = "Invalid header length"
            raise SyntaxError(msg)

        self.remaining_in_box = lbox - hlen
        return tbox


def _parse_codestream(fp):
    """Parse the JPEG 2000 codestream to extract the size and component
    count from the SIZ marker segment, returning a PIL (size, mode) tuple."""

    hdr = fp.read(2)
    lsiz = _binary.i16be(hdr)
    siz = hdr + fp.read(lsiz - 2)
    lsiz, rsiz, xsiz, ysiz, xosiz, yosiz, _, _, _, _, csiz = struct.unpack_from(
        ">HHIIIIIIIIH", siz
    )
    ssiz = [None] * csiz
    xrsiz = [None] * csiz
    yrsiz = [None] * csiz
    for i in range(csiz):
        ssiz[i], xrsiz[i], yrsiz[i] = struct.unpack_from(">BBB", siz, 36 + 3 * i)

    size = (xsiz - xosiz, ysiz - yosiz)
    if csiz == 1:
        if (yrsiz[0] & 0x7F) > 8:
            mode = "I;16"
        else:
            mode = "L"
    elif csiz == 2:
        mode = "LA"
    elif csiz == 3:
        mode = "RGB"
    elif csiz == 4:
        mode = "RGBA"
    else:
        mode = None

    return size, mode


def _res_to_dpi(num, denom, exp):
    """Convert JPEG2000's (numerator, denominator, exponent-base-10) resolution,
    calculated as (num / denom) * 10^exp and stored in dots per meter,
    to floating-point dots per inch."""
    if denom != 0:
        return (254 * num * (10**exp)) / (10000 * denom)


def _parse_jp2_header(fp):
    """Parse the JP2 header box to extract size, component count,
    color space information, and optionally DPI information,
    returning a (size, mode, mimetype, dpi) tuple."""

    # Find the JP2 header box
    reader = BoxReader(fp)
    header = None
    mimetype = None
    while reader.has_next_box():
        tbox = reader.next_box_type()

        if tbox == b"jp2h":
            header = reader.read_boxes()
            break
        elif tbox == b"ftyp":
            if reader.read_fields(">4s")[0] == b"jpx ":
                mimetype = "image/jpx"

    size = None
    mode = None
    bpc = None
    nc = None
    dpi = None  # 2-tuple of DPI info, or None

    while header.has_next_box():
        tbox = header.next_box_type()

        if tbox == b"ihdr":
            height, width, nc, bpc = header.read_fields(">IIHB")
            size = (width, height)
            if nc == 1 and (bpc & 0x7F) > 8:
                mode = "I;16"
            elif nc == 1:
                mode = "L"
            elif nc == 2:
                mode = "LA"
            elif nc == 3:
                mode = "RGB"
            elif nc == 4:
                mode = "RGBA"
        elif tbox == b"res ":
            res = header.read_boxes()
            while res.has_next_box():
                tres = res.next_box_type()
                if tres == b"resc":
                    vrcn, vrcd, hrcn, hrcd, vrce, hrce = res.read_fields(">HHHHBB")
                    hres = _res_to_dpi(hrcn, hrcd, hrce)
                    vres = _res_to_dpi(vrcn, vrcd, vrce)
                    if hres is not None and vres is not None:
                        dpi = (hres, vres)
                    break

    if size is None or mode is None:
        msg = "Malformed JP2 header"
        raise SyntaxError(msg)

    return size, mode, mimetype, dpi


##
# Image plugin for JPEG2000 images.


class Jpeg2KImageFile(ImageFile.ImageFile):
    format = "JPEG2000"
    format_description = "JPEG 2000 (ISO 15444)"

    def _open(self):
        sig = self.fp.read(4)
        if sig == b"\xff\x4f\xff\x51":
            self.codec = "j2k"
            self._size, self.mode = _parse_codestream(self.fp)
        else:
            sig = sig + self.fp.read(8)

            if sig == b"\x00\x00\x00\x0cjP  \x0d\x0a\x87\x0a":
                self.codec = "jp2"
                header = _parse_jp2_header(self.fp)
                self._size, self.mode, self.custom_mimetype, dpi = header
                if dpi is not None:
                    self.info["dpi"] = dpi
                if self.fp.read(12).endswith(b"jp2c\xff\x4f\xff\x51"):
                    self._parse_comment()
            else:
                msg = "not a JPEG 2000 file"
                raise SyntaxError(msg)

        if self.size is None or self.mode is None:
            msg = "unable to determine size/mode"
            raise SyntaxError(msg)

        self._reduce = 0
        self.layers = 0

        fd = -1
        length = -1

        try:
            fd = self.fp.fileno()
            length = os.fstat(fd).st_size
        except Exception:
            fd = -1
            try:
                pos = self.fp.tell()
                self.fp.seek(0, io.SEEK_END)
                length = self.fp.tell()
                self.fp.seek(pos)
            except Exception:
                length = -1

        self.tile = [
            (
                "jpeg2k",
                (0, 0) + self.size,
                0,
                (self.codec, self._reduce, self.layers, fd, length),
            )
        ]

    def _parse_comment(self):
        hdr = self.fp.read(2)
        length = _binary.i16be(hdr)
        self.fp.seek(length - 2, os.SEEK_CUR)

        while True:
            marker = self.fp.read(2)
            if not marker:
                break
            typ = marker[1]
            if typ in (0x90, 0xD9):
                # Start of tile or end of codestream
                break
            hdr = self.fp.read(2)
            length = _binary.i16be(hdr)
            if typ == 0x64:
                # Comment
                self.info["comment"] = self.fp.read(length - 2)[2:]
                break
            else:
                self.fp.seek(length - 2, os.SEEK_CUR)

    @property
    def reduce(self):
        # https://github.com/python-pillow/Pillow/issues/4343 found that the
        # new Image 'reduce' method was shadowed by this plugin's 'reduce'
        # property. This attempts to allow for both scenarios
        return self._reduce or super().reduce

    @reduce.setter
    def reduce(self, value):
        self._reduce = value

    def load(self):
        if self.tile and self._reduce:
            power = 1 << self._reduce
            adjust = power >> 1
            self._size = (
                int((self.size[0] + adjust) / power),
                int((self.size[1] + adjust) / power),
            )

            # Update the reduce and layers settings
            t = self.tile[0]
            t3 = (t[3][0], self._reduce, self.layers, t[3][3], t[3][4])
            self.tile = [(t[0], (0, 0) + self.size, t[2], t3)]

        return ImageFile.ImageFile.load(self)


def _accept(prefix):
    return (
        prefix[:4] == b"\xff\x4f\xff\x51"
        or prefix[:12] == b"\x00\x00\x00\x0cjP  \x0d\x0a\x87\x0a"
    )


# ------------------------------------------------------------
# Save support


def _save(im, fp, filename):
    # Get the keyword arguments
    info = im.encoderinfo

    if filename.endswith(".j2k") or info.get("no_jp2", False):
        kind = "j2k"
    else:
        kind = "jp2"

    offset = info.get("offset", None)
    tile_offset = info.get("tile_offset", None)
    tile_size = info.get("tile_size", None)
    quality_mode = info.get("quality_mode", "rates")
    quality_layers = info.get("quality_layers", None)
    if quality_layers is not None and not (
        isinstance(quality_layers, (list, tuple))
        and all(
            [
                isinstance(quality_layer, (int, float))
                for quality_layer in quality_layers
            ]
        )
    ):
        msg = "quality_layers must be a sequence of numbers"
        raise ValueError(msg)

    num_resolutions = info.get("num_resolutions", 0)
    cblk_size = info.get("codeblock_size", None)
    precinct_size = info.get("precinct_size", None)
    irreversible = info.get("irreversible", False)
    progression = info.get("progression", "LRCP")
    cinema_mode = info.get("cinema_mode", "no")
    mct = info.get("mct", 0)
    signed = info.get("signed", False)
    comment = info.get("comment")
    if isinstance(comment, str):
        comment = comment.encode()
    plt = info.get("plt", False)

    fd = -1
    if hasattr(fp, "fileno"):
        try:
            fd = fp.fileno()
        except Exception:
            fd = -1

    im.encoderconfig = (
        offset,
        tile_offset,
        tile_size,
        quality_mode,
        quality_layers,
        num_resolutions,
        cblk_size,
        precinct_size,
        irreversible,
        progression,
        cinema_mode,
        mct,
        signed,
        fd,
        comment,
        plt,
    )

    ImageFile._save(im, fp, [("jpeg2k", (0, 0) + im.size, 0, kind)])


# ------------------------------------------------------------
# Registry stuff


Image.register_open(Jpeg2KImageFile.format, Jpeg2KImageFile, _accept)
Image.register_save(Jpeg2KImageFile.format, _save)

Image.register_extensions(
    Jpeg2KImageFile.format, [".jp2", ".j2k", ".jpc", ".jpf", ".jpx", ".j2c"]
)

Image.register_mime(Jpeg2KImageFile.format, "image/jp2")
