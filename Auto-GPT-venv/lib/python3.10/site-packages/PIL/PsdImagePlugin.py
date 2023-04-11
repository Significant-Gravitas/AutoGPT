#
# The Python Imaging Library
# $Id$
#
# Adobe PSD 2.5/3.0 file handling
#
# History:
# 1995-09-01 fl   Created
# 1997-01-03 fl   Read most PSD images
# 1997-01-18 fl   Fixed P and CMYK support
# 2001-10-21 fl   Added seek/tell support (for layers)
#
# Copyright (c) 1997-2001 by Secret Labs AB.
# Copyright (c) 1995-2001 by Fredrik Lundh
#
# See the README file for information on usage and redistribution.
#

import io

from . import Image, ImageFile, ImagePalette
from ._binary import i8
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._binary import si16be as si16

MODES = {
    # (photoshop mode, bits) -> (pil mode, required channels)
    (0, 1): ("1", 1),
    (0, 8): ("L", 1),
    (1, 8): ("L", 1),
    (2, 8): ("P", 1),
    (3, 8): ("RGB", 3),
    (4, 8): ("CMYK", 4),
    (7, 8): ("L", 1),  # FIXME: multilayer
    (8, 8): ("L", 1),  # duotone
    (9, 8): ("LAB", 3),
}


# --------------------------------------------------------------------.
# read PSD images


def _accept(prefix):
    return prefix[:4] == b"8BPS"


##
# Image plugin for Photoshop images.


class PsdImageFile(ImageFile.ImageFile):
    format = "PSD"
    format_description = "Adobe Photoshop"
    _close_exclusive_fp_after_loading = False

    def _open(self):
        read = self.fp.read

        #
        # header

        s = read(26)
        if not _accept(s) or i16(s, 4) != 1:
            msg = "not a PSD file"
            raise SyntaxError(msg)

        psd_bits = i16(s, 22)
        psd_channels = i16(s, 12)
        psd_mode = i16(s, 24)

        mode, channels = MODES[(psd_mode, psd_bits)]

        if channels > psd_channels:
            msg = "not enough channels"
            raise OSError(msg)
        if mode == "RGB" and psd_channels == 4:
            mode = "RGBA"
            channels = 4

        self.mode = mode
        self._size = i32(s, 18), i32(s, 14)

        #
        # color mode data

        size = i32(read(4))
        if size:
            data = read(size)
            if mode == "P" and size == 768:
                self.palette = ImagePalette.raw("RGB;L", data)

        #
        # image resources

        self.resources = []

        size = i32(read(4))
        if size:
            # load resources
            end = self.fp.tell() + size
            while self.fp.tell() < end:
                read(4)  # signature
                id = i16(read(2))
                name = read(i8(read(1)))
                if not (len(name) & 1):
                    read(1)  # padding
                data = read(i32(read(4)))
                if len(data) & 1:
                    read(1)  # padding
                self.resources.append((id, name, data))
                if id == 1039:  # ICC profile
                    self.info["icc_profile"] = data

        #
        # layer and mask information

        self.layers = []

        size = i32(read(4))
        if size:
            end = self.fp.tell() + size
            size = i32(read(4))
            if size:
                _layer_data = io.BytesIO(ImageFile._safe_read(self.fp, size))
                self.layers = _layerinfo(_layer_data, size)
            self.fp.seek(end)
        self.n_frames = len(self.layers)
        self.is_animated = self.n_frames > 1

        #
        # image descriptor

        self.tile = _maketile(self.fp, mode, (0, 0) + self.size, channels)

        # keep the file open
        self._fp = self.fp
        self.frame = 1
        self._min_frame = 1

    def seek(self, layer):
        if not self._seek_check(layer):
            return

        # seek to given layer (1..max)
        try:
            name, mode, bbox, tile = self.layers[layer - 1]
            self.mode = mode
            self.tile = tile
            self.frame = layer
            self.fp = self._fp
            return name, bbox
        except IndexError as e:
            msg = "no such layer"
            raise EOFError(msg) from e

    def tell(self):
        # return layer number (0=image, 1..max=layers)
        return self.frame


def _layerinfo(fp, ct_bytes):
    # read layerinfo block
    layers = []

    def read(size):
        return ImageFile._safe_read(fp, size)

    ct = si16(read(2))

    # sanity check
    if ct_bytes < (abs(ct) * 20):
        msg = "Layer block too short for number of layers requested"
        raise SyntaxError(msg)

    for _ in range(abs(ct)):
        # bounding box
        y0 = i32(read(4))
        x0 = i32(read(4))
        y1 = i32(read(4))
        x1 = i32(read(4))

        # image info
        mode = []
        ct_types = i16(read(2))
        types = list(range(ct_types))
        if len(types) > 4:
            continue

        for _ in types:
            type = i16(read(2))

            if type == 65535:
                m = "A"
            else:
                m = "RGBA"[type]

            mode.append(m)
            read(4)  # size

        # figure out the image mode
        mode.sort()
        if mode == ["R"]:
            mode = "L"
        elif mode == ["B", "G", "R"]:
            mode = "RGB"
        elif mode == ["A", "B", "G", "R"]:
            mode = "RGBA"
        else:
            mode = None  # unknown

        # skip over blend flags and extra information
        read(12)  # filler
        name = ""
        size = i32(read(4))  # length of the extra data field
        if size:
            data_end = fp.tell() + size

            length = i32(read(4))
            if length:
                fp.seek(length - 16, io.SEEK_CUR)

            length = i32(read(4))
            if length:
                fp.seek(length, io.SEEK_CUR)

            length = i8(read(1))
            if length:
                # Don't know the proper encoding,
                # Latin-1 should be a good guess
                name = read(length).decode("latin-1", "replace")

            fp.seek(data_end)
        layers.append((name, mode, (x0, y0, x1, y1)))

    # get tiles
    for i, (name, mode, bbox) in enumerate(layers):
        tile = []
        for m in mode:
            t = _maketile(fp, m, bbox, 1)
            if t:
                tile.extend(t)
        layers[i] = name, mode, bbox, tile

    return layers


def _maketile(file, mode, bbox, channels):
    tile = None
    read = file.read

    compression = i16(read(2))

    xsize = bbox[2] - bbox[0]
    ysize = bbox[3] - bbox[1]

    offset = file.tell()

    if compression == 0:
        #
        # raw compression
        tile = []
        for channel in range(channels):
            layer = mode[channel]
            if mode == "CMYK":
                layer += ";I"
            tile.append(("raw", bbox, offset, layer))
            offset = offset + xsize * ysize

    elif compression == 1:
        #
        # packbits compression
        i = 0
        tile = []
        bytecount = read(channels * ysize * 2)
        offset = file.tell()
        for channel in range(channels):
            layer = mode[channel]
            if mode == "CMYK":
                layer += ";I"
            tile.append(("packbits", bbox, offset, layer))
            for y in range(ysize):
                offset = offset + i16(bytecount, i)
                i += 2

    file.seek(offset)

    if offset & 1:
        read(1)  # padding

    return tile


# --------------------------------------------------------------------
# registry


Image.register_open(PsdImageFile.format, PsdImageFile, _accept)

Image.register_extension(PsdImageFile.format, ".psd")

Image.register_mime(PsdImageFile.format, "image/vnd.adobe.photoshop")
